"""
Synchronous wrapper around ``harness.run_harness`` for the server.

Responsibilities:

  * Snapshot the project's mutable module globals (``api.MODEL`` /
    ``api.PLANNER_MODEL`` / ``api.ADVISOR_MODEL`` / ``harness.MAX_ITERATIONS``),
    patch them per the request, then restore on exit.  Mirrors the
    behaviour of the CLI's ``__main__`` block in ``harness.py`` and means
    out-of-band CLI runs in a separate process see no interference.

  * Redirect ``sys.stdout`` to a ``QueueWriter`` that simultaneously
    (a) tees every chunk to a per-run ``stdout.log`` file, (b) mirrors
    to the server's real terminal, and (c) pushes complete lines into a
    queue for SSE consumers (used in Phase 2 — the queue is populated
    now so no harness change is needed later).

  * Write ``harness_output.json`` and ``harness_output.md`` artifacts
    into the per-run directory using the same writers the CLI uses.

This module is BLOCKING — it's intended to be called from a worker
thread via ``loop.run_in_executor``.  Never call directly from an
async route handler.
"""

from __future__ import annotations

import contextlib
import io
import json
import queue as queuelib
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import api  # noqa: F401 — patched dynamically below
import harness  # noqa: F401 — patched dynamically below
from report import write_markdown_report

from .schemas import MODEL_ALIASES, RunRequest, ResultSummary
from .storage import ARTIFACT_NAMES


# Capture the real terminal stdout at import time, before any
# redirect_stdout call ever swaps sys.stdout out from under us.
# ``sys.__stdout__`` is the canonical "original stdout" but can be
# None under some embedded interpreters — fall back to whatever
# sys.stdout was at module load if so.
_REAL_STDOUT = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout


class QueueWriter(io.TextIOBase):
    """
    A file-like that tees writes to three sinks:

      * ``file`` — open file handle for ``stdout.log`` (raw chunks).
      * ``mirror`` — the server's real stdout (raw chunks), so the
        terminal that started uvicorn keeps showing harness output.
      * ``q`` — a ``queue.Queue`` that receives one item per LINE
        (newline-buffered).  Phase 2's SSE endpoint drains this.

    The line-buffering matters: ``print(x)`` issues two writes (the
    text + ``"\\n"``), and we want one queue item per logical line, not
    one per fragment.

    Single-threaded by construction: only the run worker thread writes
    here (because ``contextlib.redirect_stdout`` reassigns ``sys.stdout``
    process-wide for the duration of the run, and we serialise runs
    via a single-worker executor).  No internal locking needed.
    """

    def __init__(
        self,
        q: queuelib.Queue,
        file_handle: Optional[io.TextIOBase],
        mirror: Optional[io.TextIOBase],
    ) -> None:
        super().__init__()
        self.q = q
        self.file = file_handle
        self.mirror = mirror
        self._buf = ""

    def write(self, s: Any) -> int:
        if not isinstance(s, str):
            s = str(s)
        # Tee raw chunks to the mirror + file FIRST so even partial /
        # unflushed lines reach the terminal and the log as quickly as
        # possible (matches how print() normally feels).
        if self.mirror is not None:
            try:
                self.mirror.write(s)
            except Exception:
                pass
        if self.file is not None:
            try:
                self.file.write(s)
            except Exception:
                pass
        # Then line-buffer for the queue.
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            try:
                self.q.put_nowait(line)
            except Exception:
                # An unbounded Queue shouldn't raise on put, but if it
                # ever does we'd rather drop a log line than crash the run.
                pass
        return len(s)

    def flush(self) -> None:
        if self.mirror is not None:
            try:
                self.mirror.flush()
            except Exception:
                pass
        if self.file is not None:
            try:
                self.file.flush()
            except Exception:
                pass

    def writable(self) -> bool:
        return True


def _resolve_model_patches(req: RunRequest) -> Optional[str]:
    """
    Decide what (if any) single model string should override all three
    agent globals.  Returns the resolved model ID, or ``None`` when the
    per-agent defaults (Opus/Sonnet/Haiku) should stand.

      * ``test=true``   → Haiku for everything (mirrors CLI ``--test``).
      * ``model="..."`` → that model for everything (mirrors ``--model X``).
      * neither set     → ``None`` (no patching, per-agent mix used).
    """
    if req.test:
        return MODEL_ALIASES["haiku"]
    if req.model is not None:
        return MODEL_ALIASES.get(req.model, req.model)
    return None


def _build_result_summary(result: dict[str, Any]) -> ResultSummary:
    final_prop = result.get("final_proposal") or {}
    final_eval = result.get("final_evaluation") or {}
    return ResultSummary(
        selected_iteration=result.get("selected_iteration"),
        final_average_score=float(final_eval.get("average_score") or 0),
        final_expected_return=float(final_prop.get("expected_annual_return") or 0),
        final_expected_max_drawdown=float(final_prop.get("expected_max_drawdown") or 0),
        passed_qa=bool(final_eval.get("passed", False)),
    )


def execute_run_sync(
    req: RunRequest,
    artifacts_dir: Path,
    log_queue: queuelib.Queue,
) -> tuple[Optional[dict[str, Any]], Optional[ResultSummary], Optional[str]]:
    """
    Blocking entry point used by ``JobManager`` inside its single-worker
    executor.  Returns ``(result_dict_or_None, summary_or_None,
    error_str_or_None)``.  Never raises — all exceptions are captured
    into the error tuple slot so the caller can update the JobRecord
    cleanly.
    """
    # --- Snapshot mutable project globals so we can restore in finally ---
    orig_model = api.MODEL
    orig_planner = api.PLANNER_MODEL
    orig_advisor = api.ADVISOR_MODEL
    orig_max_iter = harness.MAX_ITERATIONS

    # --- Resolve effective flags (test mode overrides explicit flags) ---
    if req.test:
        refine = False
        advise = False
        price = False
        iterations: Optional[int] = 1
    else:
        refine = req.refine
        advise = req.advise
        price = req.price
        iterations = req.iterations

    # --- Apply patches ---
    override_model = _resolve_model_patches(req)
    if override_model is not None:
        api.MODEL = override_model
        api.PLANNER_MODEL = override_model
        api.ADVISOR_MODEL = override_model
    if iterations is not None:
        harness.MAX_ITERATIONS = iterations

    stdout_log = artifacts_dir / ARTIFACT_NAMES["stdout_log"]
    trace_json = artifacts_dir / ARTIFACT_NAMES["trace_json"]
    report_md = artifacts_dir / ARTIFACT_NAMES["report_md"]
    error_json = artifacts_dir / ARTIFACT_NAMES["error_json"]

    result: Optional[dict[str, Any]] = None
    summary: Optional[ResultSummary] = None
    error_str: Optional[str] = None
    error_traceback: Optional[str] = None

    try:
        # Open the log file in line-buffered text mode so anything we
        # tee into it is visible to ``tail -f`` immediately.
        with open(stdout_log, "w", buffering=1) as logf:
            writer = QueueWriter(log_queue, file_handle=logf, mirror=_REAL_STDOUT)
            # Print a header BEFORE the redirect so it lands on the server
            # terminal regardless of the mirror state.
            _REAL_STDOUT.write(
                f"\n[run {artifacts_dir.name}] starting "
                f"(model_override={override_model or '<per-agent mix>'}, "
                f"iterations={iterations if iterations is not None else harness.MAX_ITERATIONS}, "
                f"refine={refine}, advise={advise}, price={price})\n"
            )
            _REAL_STDOUT.flush()

            with contextlib.redirect_stdout(writer):
                try:
                    result = harness.run_harness(
                        req.goal,
                        refine=refine,
                        advise=advise,
                        price=price,
                        capital=req.capital,
                    )
                except Exception as exc:
                    error_traceback = traceback.format_exc()
                    error_str = f"{type(exc).__name__}: {exc}"
                    # Surface to the log + terminal too so the failure is
                    # visible in stdout.log and on the server's screen.
                    print(
                        f"\n!!! RUN FAILED !!!\n{error_str}\n{error_traceback}",
                        flush=True,
                    )
                finally:
                    writer.flush()

        # --- Out of redirect — write artifacts ---
        if result is not None:
            with open(trace_json, "w") as f:
                json.dump(result, f, indent=2, default=str)
            try:
                write_markdown_report(result, str(report_md))
            except Exception as exc:
                # Markdown rendering failure isn't fatal — the JSON
                # trace is the source of truth.  Record the error
                # alongside the artifacts for debugging.
                tb = traceback.format_exc()
                with open(artifacts_dir / "report_error.txt", "w") as f:
                    f.write(f"{type(exc).__name__}: {exc}\n{tb}")
            summary = _build_result_summary(result)

        if error_str is not None:
            with open(error_json, "w") as f:
                json.dump(
                    {"error": error_str, "traceback": error_traceback},
                    f,
                    indent=2,
                )

    finally:
        # --- Restore globals ---
        api.MODEL = orig_model
        api.PLANNER_MODEL = orig_planner
        api.ADVISOR_MODEL = orig_advisor
        harness.MAX_ITERATIONS = orig_max_iter
        # --- Sentinel for any SSE consumer draining the queue ---
        try:
            log_queue.put_nowait(None)
        except Exception:
            pass

    return result, summary, error_str
