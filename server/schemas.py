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
    horizon_years: int = Field(
        10,
        ge=1,
        description=(
            "Investment horizon in whole years. Mirrors `--horizon-years` "
            "on the CLI (default DEFAULT_HORIZON_YEARS=10, so omitting it "
            "reproduces today's output). Horizon = risk CAPACITY; it binds "
            "independently of max_loss and the MORE CONSERVATIVE wins. "
            ">=3y runs the full optimizer with a glide-path growth ceiling "
            "injected into the Planner; <3y short-circuits to a "
            "deterministic capital-preservation template (NO LLM agents "
            "run). Unlike max_loss/iterations this is a plain run_harness "
            "argument, not a patched global."
        ),
    )
    refine: bool = Field(True, description="Run the post-selection Refiner pass.")
    advise: bool = Field(True, description="Run the Advisor passes (intra-loop + final).")
    price: bool = Field(True, description="Run the yfinance pricing / lot-size check.")
    risk: bool = Field(
        True,
        description=(
            "Run the post-selection Monte-Carlo return-distribution profile "
            "(block-bootstrap with long-history proxies). Mirrors `--no-risk` "
            "on the CLI (set false to skip). Ignored when test=true."
        ),
    )
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
    """
    Compact summary of a finished run.  Full trace is in the JSON artifact.

    Phase 2: the summary is regime-aware.  In ``mode == "preservation"``
    (horizon_years < 3) NO LLM agents run, so the optimizer-centric fields
    (``selected_iteration``, ``final_average_score``, ``final_expected_*``)
    are naturally null/zero and ``passed_qa`` is ``None`` rather than a
    misleading ``true``/``false`` — the deterministic template was never
    put through the Generator↔Evaluator QA loop.  ``annualized_return`` is
    surfaced from the (no-LLM) Monte-Carlo risk profile so preservation
    runs still report a meaningful expected return instead of a flat zero.
    """

    mode: Literal["optimized", "preservation"]
    selected_iteration: Optional[int]
    final_average_score: float
    final_expected_return: float
    final_expected_max_drawdown: float
    # Realized geometric annualized return from the Monte-Carlo risk
    # profile (``risk_profile.annualized_return``).  None when the risk
    # pass was skipped (e.g. test mode / risk=false).
    annualized_return: Optional[float]
    # None in preservation mode: the template never went through QA.  A
    # bool only when an actual Evaluator pass/fail was produced.
    passed_qa: Optional[bool]


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
