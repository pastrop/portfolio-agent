"""
HTTP routes for the portfolio-agent server.

Endpoints:

  POST   /runs                       — start a run (returns immediately).
  GET    /runs                       — list known runs (newest first).
  GET    /runs/{id}                  — full status + result summary.
  DELETE /runs/{id}                  — drop from registry (+ files).
  GET    /runs/{id}/trace            — harness_output.json.
  GET    /runs/{id}/report           — harness_output.md.
  GET    /runs/{id}/stdout           — stdout.log (full captured output).
  GET    /runs/{id}/artifacts/{name} — generic per-run file fetch.
  GET    /healthz                    — liveness + key/runs-dir/queue info.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from .jobs import JobManager, JobRecord
from .schemas import HealthResponse, JobView, RunCreated, RunRequest
from .storage import ARTIFACT_NAMES


router = APIRouter()


def get_manager(request: Request) -> JobManager:
    """FastAPI dependency: pull the JobManager off app.state."""
    return request.app.state.job_manager


# --- Helpers ----------------------------------------------------------------


def _present_artifacts(rec: JobRecord) -> dict[str, str]:
    """Return ``{key: filename}`` for every artifact that exists on disk."""
    out: dict[str, str] = {}
    for key, name in ARTIFACT_NAMES.items():
        if (rec.artifacts_dir / name).exists():
            out[key] = name
    return out


def _to_view(rec: JobRecord) -> JobView:
    return JobView(
        run_id=rec.run_id,
        status=rec.status,  # type: ignore[arg-type]
        created_at=rec.created_at,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
        artifacts_dir=str(rec.artifacts_dir),
        artifacts=_present_artifacts(rec),
        request=rec.request,
        error=rec.error,
        result_summary=rec.result_summary,
    )


def _require_run(rec: JobRecord | None) -> JobRecord:
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    return rec


def _serve_artifact(rec: JobRecord, key: str, media_type: str) -> FileResponse:
    name = ARTIFACT_NAMES.get(key)
    if name is None:
        raise HTTPException(status_code=404, detail=f"unknown artifact key '{key}'")
    path = rec.artifacts_dir / name
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"artifact '{name}' not yet available "
                f"(run status: {rec.status})"
            ),
        )
    return FileResponse(path, media_type=media_type, filename=name)


# --- Routes -----------------------------------------------------------------


@router.get("/healthz", response_model=HealthResponse)
def healthz(manager: JobManager = Depends(get_manager)) -> HealthResponse:
    running, pending = manager.counts()
    return HealthResponse(
        ok=True,
        anthropic_key_present=bool(os.environ.get("ANTHROPIC_API_KEY")),
        runs_dir=str(manager.runs_root),
        queue_max=manager.queue_max,
        active_runs=running,
        pending_runs=pending,
    )


@router.post(
    "/runs",
    response_model=RunCreated,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_run(
    req: RunRequest, manager: JobManager = Depends(get_manager)
) -> RunCreated:
    try:
        rec = manager.submit(req)
    except asyncio.QueueFull:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Run queue is full (max {manager.queue_max} pending). "
                f"Wait for in-flight runs to drain and retry."
            ),
        )
    return RunCreated(
        run_id=rec.run_id,
        status=rec.status,  # type: ignore[arg-type]
        created_at=rec.created_at,
        artifacts_dir=str(rec.artifacts_dir),
    )


@router.get("/runs", response_model=list[JobView])
def list_runs(manager: JobManager = Depends(get_manager)) -> list[JobView]:
    return [_to_view(r) for r in manager.list_jobs()]


@router.get("/runs/{run_id}", response_model=JobView)
def get_run(run_id: str, manager: JobManager = Depends(get_manager)) -> JobView:
    return _to_view(_require_run(manager.get(run_id)))


@router.delete("/runs/{run_id}")
def delete_run(
    run_id: str,
    keep_files: bool = False,
    manager: JobManager = Depends(get_manager),
) -> dict[str, object]:
    rec = _require_run(manager.get(run_id))
    try:
        manager.delete(run_id, keep_files=keep_files)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"deleted": run_id, "keep_files": keep_files, "had_status": rec.status}


@router.get("/runs/{run_id}/trace")
def get_trace(
    run_id: str, manager: JobManager = Depends(get_manager)
) -> FileResponse:
    return _serve_artifact(_require_run(manager.get(run_id)), "trace_json", "application/json")


@router.get("/runs/{run_id}/report")
def get_report(
    run_id: str, manager: JobManager = Depends(get_manager)
) -> FileResponse:
    return _serve_artifact(_require_run(manager.get(run_id)), "report_md", "text/markdown")


@router.get("/runs/{run_id}/stdout")
def get_stdout(
    run_id: str, manager: JobManager = Depends(get_manager)
) -> FileResponse:
    return _serve_artifact(_require_run(manager.get(run_id)), "stdout_log", "text/plain")


@router.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(
    run_id: str, name: str, manager: JobManager = Depends(get_manager)
) -> FileResponse:
    rec = _require_run(manager.get(run_id))
    # Defensive against path traversal: reject any name that contains
    # path separators or starts with a dot.  We only ever serve files
    # directly inside the run's own directory.
    if "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid artifact name")
    path = rec.artifacts_dir / name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"artifact '{name}' not found")
    return FileResponse(path, filename=name)
