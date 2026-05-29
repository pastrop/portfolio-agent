"""
FastAPI app entry point for the portfolio-agent server.

Boot:

    uv run uvicorn server.main:app --reload

Environment:

    ANTHROPIC_API_KEY        — required (enforced at import time by api.py).
    PORTFOLIO_AGENT_RUNS_DIR — where per-run artifact directories live.
                               Defaults to ``./runs`` relative to CWD.
    PORTFOLIO_AGENT_QUEUE_MAX — max pending runs before POST /runs → 429.
                                Defaults to 8.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from .jobs import JobManager
from .routes import router
from .storage import get_runs_dir


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    runs_dir = get_runs_dir()
    queue_max = int(os.environ.get("PORTFOLIO_AGENT_QUEUE_MAX", "8"))
    manager = JobManager(runs_root=runs_dir, queue_max=queue_max)
    await manager.start()
    app.state.job_manager = manager
    print(
        f"[server] portfolio-agent ready  "
        f"runs_dir={runs_dir}  queue_max={queue_max}",
        flush=True,
    )
    try:
        yield
    finally:
        await manager.stop()
        print("[server] portfolio-agent shut down", flush=True)


app = FastAPI(
    title="Portfolio Agent API",
    description=(
        "HTTP wrapper around the portfolio optimisation harness. "
        "POST /runs to start a run, GET /runs/{id} to poll, fetch "
        "artifacts under /runs/{id}/trace, /report, /stdout."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
