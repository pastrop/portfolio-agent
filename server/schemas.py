"""
Pydantic request / response models for the portfolio-agent server.

The shapes here are the public contract of the HTTP API.  Keep them
stable — clients depend on the field names.  Internal types
(``JobRecord`` etc.) live in ``jobs.py``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# Short aliases accepted in ``RunRequest.model``.  Mirrors harness.py's CLI.
MODEL_ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


class RunRequest(BaseModel):
    """Body of ``POST /runs``.  Mirrors the CLI flags in harness.py 1:1."""

    goal: str = Field(
        ...,
        min_length=1,
        description="Free-form investment goal handed to the Planner agent.",
    )
    model: Optional[str] = Field(
        None,
        description=(
            "Override ALL three agent models. Aliases: 'haiku' | 'sonnet' "
            "| 'opus', or a full Anthropic model ID. When omitted, the "
            "per-agent defaults are used (Opus for Generator/Evaluator/"
            "Refiner, Sonnet for Planner, Haiku for Advisor) — same as "
            "running `python harness.py` with no flags."
        ),
    )
    test: bool = Field(
        False,
        description=(
            "Test mode: forces all agents to Haiku, 1 iteration, and "
            "skips refine/advise/price.  Mirrors `--test` on the CLI."
        ),
    )
    iterations: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Override MAX_ITERATIONS (the Generator ↔ Evaluator round "
            "count).  Ignored when test=true (which forces iterations=1)."
        ),
    )
    max_loss: Optional[float] = Field(
        None,
        gt=0,
        lt=1,
        description=(
            "Override TARGET_MAX_LOSS (the max annual loss budget) as a "
            "fraction in (0, 1) — e.g. 0.10 for 10%. Drives the agent "
            "prompts, the selection target, and the auto-derived "
            "under-utilisation band. Mirrors `--max-loss` on the CLI; "
            "applies in test mode too. When omitted, the harness default "
            "is used."
        ),
    )
    refine: bool = Field(True, description="Run the post-selection Refiner pass.")
    advise: bool = Field(True, description="Run the Advisor passes (intra-loop + final).")
    price: bool = Field(True, description="Run the yfinance pricing / lot-size check.")
    capital: float = Field(
        100_000.0,
        gt=0,
        description="Assumed investable capital in USD for the pricing pass.",
    )


class RunCreated(BaseModel):
    """Response to ``POST /runs`` — returned immediately, before the run completes."""

    run_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    artifacts_dir: str


class ResultSummary(BaseModel):
    """Compact summary of a finished run.  Full trace is in the JSON artifact."""

    selected_iteration: Optional[int]
    final_average_score: float
    final_expected_return: float
    final_expected_max_drawdown: float
    passed_qa: bool


class JobView(BaseModel):
    """Response shape for ``GET /runs`` and ``GET /runs/{id}``."""

    run_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    artifacts_dir: str
    # Map of artifact-key → filename, only including files that exist on disk.
    artifacts: dict[str, str]
    request: RunRequest
    error: Optional[str]
    result_summary: Optional[ResultSummary]


class HealthResponse(BaseModel):
    ok: bool
    anthropic_key_present: bool
    runs_dir: str
    queue_max: int
    active_runs: int
    pending_runs: int
