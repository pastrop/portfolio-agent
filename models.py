"""
Shared dataclasses for the Portfolio Optimization Harness.

These types are passed between the agents (in ``agents.py``) and the
orchestrator (in ``harness.py``).  The module has no dependencies on
other project modules — both ``agents.py`` and ``harness.py`` import
from here, so keeping it dep-free avoids circular imports.

Every dataclass carries the model's raw text response in ``raw_text``
so the markdown report can include it in collapsible ``<details>``
blocks for transparency / debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InvestmentSpec:
    """Produced by the Planner.  Consumed by Generator & Evaluator."""
    objective: str = ""
    constraints: str = ""
    asset_universe: str = ""
    risk_budget: str = ""
    evaluation_criteria: str = ""
    raw_text: str = ""


@dataclass
class PortfolioProposal:
    """Produced by the Generator.  Consumed by the Evaluator."""
    allocations: dict[str, float] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    expected_annual_return: float = 0.0
    expected_max_drawdown: float = 0.0
    methodology: str = ""
    rationale: str = ""
    raw_text: str = ""


@dataclass
class EvaluationResult:
    """Produced by the Evaluator.  Fed back to Generator on failure."""
    passed: bool = False
    scores: dict[str, float] = field(default_factory=dict)
    average_score: float = 0.0
    critique: str = ""
    raw_text: str = ""


@dataclass
class AdvisorOutput:
    """
    Produced by the Advisor.  Pure advisory — never changes the portfolio.
    Surfaces highly correlated holdings in a portfolio and offers concrete
    consolidation ideas with explicit trade-offs.

    Used in two places:
      • Per-iteration (intra-loop) — its correlation_pairs (filtered to
        |ρ| ≥ ADVISOR_FEEDBACK_RHO_THRESHOLD) become Generator feedback
        for the next round.
      • Final pass (after the orchestrator picks the final portfolio) —
        suggestions and the full correlation snapshot are rendered into
        the markdown report's "Simplification Suggestions" section.
    """
    # List of {"merge_from": [tickers], "merge_into": ticker,
    #          "rationale": str, "tradeoff": str}
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    # List of {"a": ticker_a, "b": ticker_b, "rho": float,
    #          "note": optional str}
    correlation_pairs: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    raw_text: str = ""
