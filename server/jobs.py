"""
In-memory job registry + single-worker execution for the server.

Design:

  * One process-global ``JobManager`` lives on ``app.state.job_manager``.
  * ``submit(req)`` is synchronous: creates the per-run directory,
    writes ``request.json``, registers the job, enqueues it.  Returns
    immediately so ``POST /runs`` doesn't block.
  * A background worker coroutine pulls run-ids off the asyncio
    queue one at a time and dispatches them to a dedicated
    single-thread ``ThreadPoolExecutor`` via ``loop.run_in_executor``.
    The single-worker executor matches the "single run at a time"
    decision and guarantees ``contextlib.redirect_stdout`` in
    ``runner.execute_run_sync`` can't leak across runs.
  * Pending queue is bounded (default 8).  When full, ``submit`` raises
    ``asyncio.QueueFull`` which the route handler translates to HTTP 429.

Restart loses the registry — artifacts on disk survive but
``GET /runs/{id}`` will 404 until the registry is rebuilt from disk
(Phase 3 work; out of scope for Phase 1).
"""

from __future__ import annotations

import asyncio
import json
import queue as queuelib
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .runner import execute_run_sync
from .schemas import RunRequest, ResultSummary
from .storage import ARTIFACT_NAMES, make_run_dir, new_run_id


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class JobRecord:
    """One run's in-memory state.  Mutated by the worker as it progresses."""

    run_id: str
    request: RunRequest
    status: str  # "queued" | "running" | "succeeded" | "failed"
    created_at: str
    artifacts_dir: Path
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result_summary: Optional[ResultSummary] = None
    # Phase 2's SSE consumer drains this.  Populated by QueueWriter in
    # runner.py; unbounded on purpose (lines are small, runs are
    # minutes-bounded, no consumer is fine).
    log_queue: queuelib.Queue = field(default_factory=queuelib.Queue)


class JobManager:
    """Owns the job registry, the pending asyncio queue, and the worker."""

    def __init__(self, runs_root: Path, queue_max: int) -> None:
        self.runs_root = runs_root
        self.queue_max = queue_max
        self.jobs: dict[str, JobRecord] = {}
        self.pending: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_max)
        # max_workers=1: serialise runs at the executor level so the
        # stdout-redirect can't be observed by anything else.
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="harness-run"
        )
        self._worker_task: Optional[asyncio.Task] = None
        self._stopped = False

    # --- Lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._worker_task = loop.create_task(self._worker_loop(), name="harness-worker")

    async def stop(self) -> None:
        self._stopped = True
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass
        self.executor.shutdown(wait=False, cancel_futures=True)

    # --- Public CRUD -------------------------------------------------------

    def submit(self, req: RunRequest) -> JobRecord:
        """
        Create + register a job.  Synchronous: no await.  Raises
        ``asyncio.QueueFull`` when the pending cap is reached.
        """
        run_id = new_run_id()
        artifacts_dir = make_run_dir(self.runs_root, run_id)
        # Persist the request body immediately so the run is reproducible
        # from disk even if the server crashes before the run starts.
        with open(artifacts_dir / ARTIFACT_NAMES["request_json"], "w") as f:
            json.dump(req.model_dump(), f, indent=2)
        rec = JobRecord(
            run_id=run_id,
            request=req,
            status="queued",
            created_at=_utcnow(),
            artifacts_dir=artifacts_dir,
        )
        # Enqueue BEFORE recording in self.jobs so a QueueFull leaves
        # no dangling registry entry.
        self.pending.put_nowait(run_id)
        self.jobs[run_id] = rec
        return rec

    def get(self, run_id: str) -> Optional[JobRecord]:
        return self.jobs.get(run_id)

    def list_jobs(self) -> list[JobRecord]:
        """Newest-first by created_at."""
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, run_id: str, *, keep_files: bool = False) -> bool:
        """
        Remove from the registry and (optionally) delete the on-disk
        artifacts directory.  Returns False if the id was unknown.
        Refuses to delete a run that's currently ``running``.
        """
        rec = self.jobs.get(run_id)
        if rec is None:
            return False
        if rec.status == "running":
            raise RuntimeError("cannot delete a run that is currently running")
        self.jobs.pop(run_id, None)
        if not keep_files:
            try:
                shutil.rmtree(rec.artifacts_dir)
            except Exception:
                # Best-effort: registry is gone either way.
                pass
        return True

    # --- Introspection (for /healthz) -------------------------------------

    def counts(self) -> tuple[int, int]:
        """Returns (active_running, queued_pending)."""
        running = sum(1 for j in self.jobs.values() if j.status == "running")
        return running, self.pending.qsize()

    # --- Worker -----------------------------------------------------------

    async def _worker_loop(self) -> None:
        while not self._stopped:
            try:
                run_id = await self.pending.get()
            except asyncio.CancelledError:
                break
            rec = self.jobs.get(run_id)
            if rec is None:
                # Submitted but deleted before pickup — skip.
                continue
            try:
                await self._execute(rec)
            except Exception as exc:  # noqa: BLE001 — defensive
                # Should be impossible (execute_run_sync swallows its
                # own exceptions) but a bare except keeps the worker
                # alive across surprises.
                rec.status = "failed"
                rec.error = f"{type(exc).__name__}: {exc}"
                rec.finished_at = _utcnow()

    async def _execute(self, rec: JobRecord) -> None:
        rec.status = "running"
        rec.started_at = _utcnow()
        loop = asyncio.get_running_loop()
        result, summary, error_str = await loop.run_in_executor(
            self.executor,
            execute_run_sync,
            rec.request,
            rec.artifacts_dir,
            rec.log_queue,
        )
        if error_str is not None:
            rec.status = "failed"
            rec.error = error_str
        else:
            rec.status = "succeeded"
            rec.result_summary = summary
        rec.finished_at = _utcnow()
