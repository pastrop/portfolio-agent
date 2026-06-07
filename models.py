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
