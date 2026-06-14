"""
Portfolio Optimization Harness — orchestrator + CLI
====================================================
A multi-agent harness inspired by Anthropic's "Harness design for
long-running application development" blog post, applied to financial
portfolio optimisation.

Composition (Planner → (Generator ↔ Evaluator) × N → Selector → Refiner
→ Re-evaluate → Pricing → Risk → Correlation) lives in this file.  The
four LLM agents themselves are in ``agents.py``; the Anthropic client
wrapper and JSON parser are in ``api.py``; the dataclasses passed between
agents are in ``models.py``; the markdown report writer is in
``report.py``; the yfinance-backed lot-size step is in ``pricing.py``;
the no-LLM pairwise-correlation snapshot is in ``correlation.py``.

This is a *concept exploration* — most data is model-recalled, not
computed from market feeds (except for the Pricing step's yfinance
spot-price lookup).
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from typing import Any

import api
from agents import (
    run_evaluator,
    run_generator,
    run_planner,
    run_refiner,
)
from models import (
    EvaluationResult,
    InvestmentSpec,
    PortfolioProposal,
)
from correlation import CORRELATION_DISCLAIMER, run_correlation
from loss_floor import LOSS_FLOOR_DISCLAIMER, run_loss_floor_check
from pricing import DEFAULT_CAPITAL, PRICING_DISCLAIMER, run_pricing
from report import write_markdown_report
from risk import RISK_DISCLAIMER, RISK_PROXY_MAP, run_risk_profile

# ---------------------------------------------------------------------------
# Orchestrator-only policy constants
# ---------------------------------------------------------------------------
# (Model / token / retry config lives in api.py; per-agent model selection
# is api.MODEL / api.PLANNER_MODEL.  The CLI block at the bottom of this
# file patches those module-level globals when --test or --model X is
# provided.)
MAX_ITERATIONS = 3          # generator ↔ evaluator rounds
PASS_THRESHOLD = 7          # minimum average score (out of 10) to pass QA
TARGET_MAX_LOSS = 0.05      # default loss budget we want expected_max_drawdown to land on (overridable via --max-loss)
# The under-utilisation band auto-derives from TARGET_MAX_LOSS: a drawdown
# below this fraction of the budget is "wasting risk capacity".  Whenever
# TARGET_MAX_LOSS is patched (CLI --max-loss, server runner), recompute
# UNDER_UTILISATION_BAND = UNDER_UTILISATION_RATIO * TARGET_MAX_LOSS.
UNDER_UTILISATION_RATIO = 0.8
UNDER_UTILISATION_BAND = UNDER_UTILISATION_RATIO * TARGET_MAX_LOSS

# ---------------------------------------------------------------------------
# Horizon policy (Phase 2) — horizon = risk CAPACITY
# ---------------------------------------------------------------------------
# `--horizon-years N` makes the recommended portfolio horizon-aware.  The
# mental model: horizon is risk *capacity*; `--max-loss` is risk *tolerance*.
# They bind independently and the MORE CONSERVATIVE wins.  At long horizons
# the max-loss budget binds (≈ today's output); at short horizons the
# horizon ceiling binds.
#
# DEFAULT_HORIZON_YEARS is 10 deliberately: a no-flag run lands in the
# topmost glide-path band (90% growth ceiling, which never bites a sane
# max-loss-constrained book) and so reproduces today's output byte-for-byte.
# This is the Phase 2 acceptance gate — do NOT change it without re-baselining
# harness_output.json.
DEFAULT_HORIZON_YEARS = 10
# Hard floor: anything below this short-circuits to a deterministic
# capital-preservation template and the optimizer NEVER runs (no LLM agents).
# There is no escape hatch — a 2-year horizon has no business holding an
# equity-heavy book regardless of stated risk tolerance.
HORIZON_FLOOR_YEARS = 3

# Glide path (optimizer regime, horizon_years >= 3): a ceiling on the
# combined weight of GROWTH assets (equity + REIT + commodity + high-yield).
# The remainder must be high-grade bonds / TIPS / cash.  Bands are
# HALF-OPEN on the right: [low, high).  An ordered list so the first band
# whose [low, high) contains the horizon wins; the final band runs to
# infinity (encoded as ``None`` high).
#   horizon 3 -> 0.50,  horizon 5 -> 0.75,  horizon 10 -> 0.90.
HORIZON_GLIDE_PATH: list[dict[str, Any]] = [
    {"low": 3,  "high": 5,    "label": "Conservative balanced", "band": "3-5y",  "growth_ceiling": 0.50},
    {"low": 5,  "high": 10,   "label": "Balanced growth",       "band": "5-10y", "growth_ceiling": 0.75},
    {"low": 10, "high": None, "label": "Growth",                "band": "10y+",  "growth_ceiling": 0.90},
]

# Preservation templates (short-circuit regime, horizon_years < 3): fixed,
# deterministic allocations — no LLM judgement, no optimisation.  These ARE
# the answer below the floor.
#   1 <= horizon < 3 -> "1-3y";  horizon < 1 -> "<1y".
PRESERVATION_TEMPLATES: dict[str, dict[str, float]] = {
    "1-3y": {"SGOV": 0.40, "SHY": 0.40, "VTIP": 0.20},
    "<1y":  {"SGOV": 1.00},
}
# Posture label reported for any preservation-mode run.
PRESERVATION_POSTURE = "Capital preservation"

# ---------------------------------------------------------------------------
# Growth-asset classifier (shared with the Evaluator's deterministic ceiling
# check in agents.py via `growth_asset_weight`)
# ---------------------------------------------------------------------------
# The glide path caps GROWTH-asset weight, but PortfolioProposal has NO
# structured asset-class field (allocations is just ticker -> weight), so we
# must classify each holding ourselves.  This is a heuristic, FAIL-SOFT
# classifier — it errs toward counting an *unclassifiable* holding AS growth
# (the conservative choice for a ceiling check: an unknown sleeve can only
# tighten the gate, never loosen it, so the deterministic Evaluator check can
# never wave through a genuinely over-the-ceiling book on a classification
# miss).
#
# "Growth" = equity (US / international / EM / factor / dividend / small-cap),
# REITs, broad commodities, gold, managed futures, and HIGH-YIELD credit —
# i.e. the spec's growth bucket plus the risk sleeves that behave like it.
# "Defensive" = high-grade bonds (Treasuries, IG credit, aggregate, munis),
# TIPS / inflation-linked, and cash / T-bills.  Options overlays and explicit
# CASH/USD sleeves are treated as defensive (they don't consume the growth
# budget).
#
# Three classification signals, in priority order:
#   1. Exact ticker membership in the curated growth / defensive sets below
#      (built from the Generator prompt's overlap groups + risk.RISK_PROXY_MAP
#      so every proxy'd young ETF resolves too).
#   2. The risk.RISK_PROXY_MAP long-history proxy (e.g. SCHD -> VIG): if the
#      proxy is classifiable, inherit its class.
#   3. Keyword scan of the holding's one-line description (e.g. "equity",
#      "REIT", "high yield", "treasury", "TIPS", "cash") as a last resort.
# Anything still unresolved -> counted as growth (fail-soft, see above).

# Defensive tickers: high-grade bonds, TIPS, cash.  (Drawn from the Generator
# overlap groups + the bond/cash/TIPS entries of RISK_PROXY_MAP's domain.)
_DEFENSIVE_TICKERS: frozenset[str] = frozenset({
    # intermediate Treasuries
    "IEF", "GOVT", "VGIT", "SCHR",
    # long Treasuries
    "TLT", "EDV", "VGLT", "SPTL",
    # short Treasuries / cash / T-bills
    "SHV", "SHY", "BIL", "SGOV", "GBIL",
    # investment-grade credit
    "LQD", "VCIT", "VCSH", "IGIB", "IGSB",
    # US aggregate bonds
    "AGG", "BND", "SCHZ", "IUSB",
    # TIPS / inflation-linked
    "TIP", "SCHP", "VTIP", "STIP", "LTPZ",
    # municipal bonds
    "MUB", "VTEB", "TFI", "SUB", "VWITX",
})
# Growth tickers: equity (broad / factor / dividend / small-cap / int'l / EM),
# REITs, gold, broad commodities, managed futures, high-yield credit.
_GROWTH_TICKERS: frozenset[str] = frozenset({
    # US broad equity
    "VOO", "VTI", "SPY", "IVV", "SPLG", "ITOT", "SCHB",
    # US large-cap factor tilts
    "QUAL", "MTUM", "VLUE", "USMV", "SPHQ", "SPLV",
    # US dividend tilts
    "SCHD", "DGRO", "VYM", "HDV", "NOBL", "DVY", "VIG",
    # US small-cap
    "IWM", "VB", "IJR", "SCHA", "VTWO",
    # int'l developed equity
    "VEA", "IEFA", "VXUS", "SCHF", "IDEV", "EFA",
    # emerging-market equity
    "VWO", "IEMG", "EEM", "SCHE", "SPEM",
    # high-yield credit (behaves like equity in stress -> growth bucket)
    "HYG", "JNK", "USHY", "SHYG",
    # gold
    "GLD", "IAU", "GLDM", "SGOL", "BAR",
    # broad commodities
    "DBC", "PDBC", "GSG", "BCI", "COMT",
    # US REITs
    "VNQ", "IYR", "SCHH", "XLRE", "RWR",
    # managed futures
    "DBMF", "KMLM", "CTA", "WTMF",
})
# Description keywords, scanned only when ticker + proxy both miss.  Order
# matters: a defensive keyword wins over a growth one ONLY when no growth
# keyword is present (so "high-yield bond" -> growth, "treasury bond" ->
# defensive).
_GROWTH_KEYWORDS: tuple[str, ...] = (
    "equity", "stock", "s&p", "nasdaq", "reit", "real estate", "commodit",
    "gold", "managed futures", "high yield", "high-yield", "emerging",
    "small-cap", "small cap", "dividend",
)
_DEFENSIVE_KEYWORDS: tuple[str, ...] = (
    "treasury", "t-bill", "bill", "tips", "inflation-linked", "inflation linked",
    "aggregate bond", "investment-grade", "investment grade", "municipal",
    "muni", "cash", "money market", "money-market", "government bond",
)


def _classify_holding(ticker: str, description: str = "") -> str:
    """
    Classify a single holding as ``"growth"`` or ``"defensive"``.

    FAIL-SOFT: an unclassifiable holding returns ``"growth"`` so that the
    Evaluator's ceiling check can only ever be tightened — never loosened —
    by a classification miss (an unknown sleeve must not silently buy room
    under the growth ceiling).  See the module-level note above for the
    full rationale and the three-signal priority order.
    """
    t = (ticker or "").strip().upper()
    if t in _DEFENSIVE_TICKERS:
        return "defensive"
    if t in _GROWTH_TICKERS:
        return "growth"

    # Signal 2: inherit the long-history proxy's class, if it resolves.
    proxy = RISK_PROXY_MAP.get(t)
    if proxy is not None:
        p = proxy.upper()
        if p in _DEFENSIVE_TICKERS:
            return "defensive"
        if p in _GROWTH_TICKERS:
            return "growth"

    # Signal 3: keyword scan of the description.  A growth keyword wins over
    # a defensive one (so "high-yield bond" lands in growth).
    desc = (description or "").lower()
    if any(kw in desc for kw in _GROWTH_KEYWORDS):
        return "growth"
    if any(kw in desc for kw in _DEFENSIVE_KEYWORDS):
        return "defensive"

    # Explicit cash sleeves the keyword list might miss.
    if t in {"CASH", "USD", "MONEY_MARKET"}:
        return "defensive"
    # Options overlays / hedges named like SPX_PUT_SPREAD: a hedge is not a
    # growth asset, so treat any holding whose name screams "hedge/put/option"
    # as defensive rather than fail-soft growth.
    if any(kw in t for kw in ("PUT", "HEDGE", "OPTION", "COLLAR")):
        return "defensive"

    # Fail-soft default: count the unknown holding as growth.
    return "growth"


def growth_asset_weight(proposal: PortfolioProposal) -> float:
    """
    Return the combined weight of GROWTH assets in ``proposal``'s
    allocations, as a fraction in [0, 1] (normalised by the total weight so
    a book whose weights don't sum to exactly 1.0 still yields a comparable
    fraction).

    This is the shared classifier behind the glide-path ceiling: the harness
    uses it for diagnostics and the Evaluator (in ``agents.py``) imports it
    for its deterministic, hard pass/fail growth-ceiling check.  See the
    classifier note above for the heuristic + its fail-soft contract.
    """
    allocations = proposal.allocations or {}
    descriptions = proposal.descriptions or {}
    total = 0.0
    growth = 0.0
    for ticker, weight in allocations.items():
        try:
            w = abs(float(weight))
        except (TypeError, ValueError):
            continue
        total += w
        if _classify_holding(ticker, descriptions.get(ticker, "")) == "growth":
            growth += w
    if total <= 0:
        return 0.0
    return growth / total


# ---------------------------------------------------------------------------
# Risk-factor budget (shared with the Evaluator's deterministic factor gate
# in agents.py via ``factor_budget_violations``)
# ---------------------------------------------------------------------------
# Coarser than the growth/defensive split AND coarser than product category:
# it collapses each macro risk factor's many wrappers into one bucket, because
# daily returns are dominated by the FACTOR, not the wrapper.  The two buckets
# below are the ones most prone to redundant "cousins" — instruments in
# different product categories that nonetheless load the SAME factor (and so
# run ~0.8-0.95 correlated):
#   • _RATES_TICKERS — Treasuries (any maturity), IG corporate credit,
#     aggregate bonds, munis.  IG credit / agg are ~90% duration in daily
#     moves, so they are NOT independent of Treasuries.  (TIPS are the separate
#     INFLATION factor; short-Tsy / T-bills are the separate CASH factor —
#     neither is counted here.)
#   • _EQUITY_TILT — equity-beta sleeves BEYOND the core US-broad + intl + EM
#     trio: factor, dividend, small-cap, REIT, high-yield.  All ~0.8-0.95
#     correlated with broad equity (→ ~1 in a crash), so they are tilts, not
#     diversifiers.
# The gate caps each bucket at ONE sleeve (see ``factor_budget_violations``);
# the other factors (inflation, commodities, gold, trend, cash, the equity
# core) are left to the Generator prompt's per-factor "one sleeve" guidance.
_RATES_TICKERS: frozenset[str] = frozenset({
    # intermediate treasuries
    "IEF", "GOVT", "VGIT", "SCHR",
    # long treasuries
    "TLT", "EDV", "VGLT", "SPTL",
    # investment-grade corporate credit (~ duration daily; NOT independent)
    "LQD", "VCIT", "VCSH", "IGIB", "IGSB",
    # US aggregate bonds (~ duration + a thin credit/MBS tilt)
    "AGG", "BND", "SCHZ", "IUSB",
    # municipal bonds (duration)
    "MUB", "VTEB", "TFI", "SUB", "VWITX",
})
_EQUITY_TILT: frozenset[str] = frozenset({
    # US large-cap factor tilts
    "QUAL", "MTUM", "VLUE", "USMV", "SPHQ", "SPLV",
    # US dividend tilts
    "SCHD", "DGRO", "VYM", "HDV", "NOBL", "DVY", "VIG",
    # US small-cap
    "IWM", "VB", "IJR", "SCHA", "VTWO",
    # US REITs (equity-beta sector tilt)
    "VNQ", "IYR", "SCHH", "XLRE", "RWR",
    # high-yield credit (equity-beta, especially in stress)
    "HYG", "JNK", "USHY", "SHYG",
})


def _factor_bucket(ticker: str) -> str | None:
    """
    Map a ticker to a capped factor bucket — ``"rates"``, ``"equity_tilt"``,
    or ``None`` (everything the gate does not cap: equity core / intl / EM,
    TIPS, commodities, gold, trend, cash, and unknowns).  Matches the raw
    ticker first, then its ``RISK_PROXY_MAP`` long-history proxy, so proxied
    young ETFs resolve too.
    """
    t = (ticker or "").strip().upper()
    if t in _RATES_TICKERS:
        return "rates"
    if t in _EQUITY_TILT:
        return "equity_tilt"
    proxy = RISK_PROXY_MAP.get(t)
    if proxy is not None:
        p = proxy.upper()
        if p in _RATES_TICKERS:
            return "rates"
        if p in _EQUITY_TILT:
            return "equity_tilt"
    return None


def factor_budget_violations(proposal: PortfolioProposal) -> list[str]:
    """
    Deterministic factor-budget check for the construction taxonomy.  Returns
    a list of human-readable violation strings (empty list -> no violation).

    Enforces the two caps the Generator prompt declares binding (see its
    DIVERSIFICATION block):
      • AT MOST ONE pure-rates sleeve (Treasuries / IG credit / agg / muni all
        load the same duration factor),
      • AT MOST ONE equity tilt beyond core US + intl + EM (factor / dividend /
        small-cap / REIT / high-yield are all equity beta).

    The Evaluator (in ``agents.py``) calls this and forces a QA FAIL on any
    violation — like the growth-ceiling gate, it can only turn a pass into a
    fail, never the reverse.  Positive-weight holdings only; classification is
    via :func:`_factor_bucket` (raw ticker, then proxy), so unknown tickers are
    simply ignored by this gate.
    """
    allocations = proposal.allocations or {}
    rates: list[str] = []
    tilts: list[str] = []
    for ticker, weight in allocations.items():
        try:
            if float(weight) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        bucket = _factor_bucket(ticker)
        if bucket == "rates":
            rates.append(ticker)
        elif bucket == "equity_tilt":
            tilts.append(ticker)

    violations: list[str] = []
    if len(rates) > 1:
        violations.append(
            f"RATES/DURATION over-budget: {len(rates)} sleeves "
            f"({', '.join(rates)}) all load the same duration factor "
            f"(Treasuries / IG credit / aggregate / muni run ~0.8-0.95 "
            f"correlated daily). Hold AT MOST ONE; TIPS and cash are separate "
            f"factors and do not count here."
        )
    if len(tilts) > 1:
        violations.append(
            f"EQUITY-BETA over-budget: {len(tilts)} tilts ({', '.join(tilts)}) "
            f"are all equity beta (factor / dividend / small-cap / REIT / "
            f"high-yield run ~0.8-0.95 correlated with broad equity). Keep AT "
            f"MOST ONE tilt beyond core US + intl + EM."
        )
    return violations


# ---------------------------------------------------------------------------
# Horizon helpers
# ---------------------------------------------------------------------------
def posture_for_horizon(horizon_years: int) -> dict[str, Any]:
    """
    Resolve the glide-path posture for an OPTIMIZED-regime horizon
    (``horizon_years >= HORIZON_FLOOR_YEARS``).

    Returns ``{"label", "growth_ceiling", "band", "horizon_years"}`` — the
    first band whose half-open ``[low, high)`` interval contains the
    horizon.  Horizons at or above the last band's ``low`` land in that
    final (open-ended) band.

    Caller contract: only call this for ``horizon_years >= HORIZON_FLOOR_YEARS``
    (the harness short-circuits below the floor before reaching here).
    """
    for band in HORIZON_GLIDE_PATH:
        low = band["low"]
        high = band["high"]
        if horizon_years >= low and (high is None or horizon_years < high):
            return {
                "label": band["label"],
                "growth_ceiling": band["growth_ceiling"],
                "band": band["band"],
                "horizon_years": horizon_years,
            }
    # Unreachable for horizon_years >= HORIZON_FLOOR_YEARS given the bands
    # above start at 3, but fail-soft to the most conservative band.
    first = HORIZON_GLIDE_PATH[0]
    return {
        "label": first["label"],
        "growth_ceiling": first["growth_ceiling"],
        "band": first["band"],
        "horizon_years": horizon_years,
    }


def preservation_template(horizon_years: int) -> dict[str, float]:
    """
    Resolve the fixed capital-preservation allocation for a SHORT-circuit
    horizon (``horizon_years < HORIZON_FLOOR_YEARS``).

    ``horizon_years < 1`` -> the ``"<1y"`` template (pure T-bills);
    ``1 <= horizon_years < 3`` -> the ``"1-3y"`` template (T-bills + short
    Treasury + short TIPS).  Returns a fresh copy so callers can mutate
    freely.
    """
    key = "<1y" if horizon_years < 1 else "1-3y"
    return dict(PRESERVATION_TEMPLATES[key])


def _refinement_improvements(
    orig_eval: EvaluationResult,
    refined_eval: EvaluationResult,
    orig_proposal: PortfolioProposal,
    refined_proposal: PortfolioProposal,
) -> dict[str, Any]:
    """Compute a structured before/after diff for the markdown report."""
    score_deltas: dict[str, dict[str, float]] = {}
    all_keys = set(orig_eval.scores) | set(refined_eval.scores)
    for k in all_keys:
        before = orig_eval.scores.get(k, 0)
        after = refined_eval.scores.get(k, 0)
        score_deltas[k] = {
            "before": before,
            "after": after,
            "delta": round(after - before, 2),
        }

    orig_w = orig_proposal.allocations
    new_w = refined_proposal.allocations
    all_tickers = set(orig_w) | set(new_w)
    allocation_changes: list[dict[str, Any]] = []
    for t in sorted(all_tickers, key=lambda x: -abs(new_w.get(x, 0) - orig_w.get(x, 0))):
        before = float(orig_w.get(t, 0))
        after = float(new_w.get(t, 0))
        if abs(after - before) < 1e-9:
            continue  # unchanged
        kind = "added" if before == 0 else "removed" if after == 0 else "changed"
        allocation_changes.append({
            "ticker": t,
            "before": before,
            "after": after,
            "delta": round(after - before, 4),
            "kind": kind,
        })

    return {
        "score_deltas": score_deltas,
        "average_score_before": orig_eval.average_score,
        "average_score_after": refined_eval.average_score,
        "average_score_delta": round(refined_eval.average_score - orig_eval.average_score, 2),
        "expected_return_before": orig_proposal.expected_annual_return,
        "expected_return_after": refined_proposal.expected_annual_return,
        "expected_max_drawdown_before": orig_proposal.expected_max_drawdown,
        "expected_max_drawdown_after": refined_proposal.expected_max_drawdown,
        "passed_before": orig_eval.passed,
        "passed_after": refined_eval.passed,
        "allocation_changes": allocation_changes,
    }


# ---------------------------------------------------------------------------
# ORCHESTRATOR — the harness loop
# ---------------------------------------------------------------------------
def _build_feedback(
    evaluation: EvaluationResult,
    proposal: PortfolioProposal,
) -> str:
    """
    Produce the next round's feedback string based on the current evaluation.

    Four regimes (selects the feedback text):
      • Failed QA              → pass the critique through unchanged (today's behaviour).
      • Passed but over target → tell the generator to bring drawdown down to
                                 ~TARGET_MAX_LOSS without sacrificing scores.
      • Passed but well
        under target           → tell the generator the risk budget is being
                                 wasted and to push drawdown closer to (but
                                 under) TARGET_MAX_LOSS.
      • Passed and near target → still ask for a refinement attempt — the
                                 selector will keep the best across iterations.
    """
    if not evaluation.passed:
        base = evaluation.critique
    else:
        dd = proposal.expected_max_drawdown
        if dd > TARGET_MAX_LOSS:
            overshoot_pct = (dd - TARGET_MAX_LOSS) * 100
            base = (
                f"QA PASSED with average score {evaluation.average_score}, "
                f"BUT expected_max_drawdown is {dd:.2%}, which exceeds the "
                f"{TARGET_MAX_LOSS:.0%} loss target by {overshoot_pct:.1f} "
                f"percentage points. Your top priority this round is to "
                f"REDUCE expected_max_drawdown as close to {TARGET_MAX_LOSS:.0%} "
                f"as possible WITHOUT crossing it, while keeping every QA "
                f"score ≥ {PASS_THRESHOLD}. Accept lower expected return if "
                f"necessary. Prior critique for reference:\n{evaluation.critique}"
            )
        elif dd < UNDER_UTILISATION_BAND:
            base = (
                f"QA PASSED with average score {evaluation.average_score} and "
                f"expected_max_drawdown of {dd:.2%}, which is well UNDER the "
                f"{TARGET_MAX_LOSS:.0%} loss budget. You are leaving return on "
                f"the table. This round, deploy more risk-bearing exposure to "
                f"raise expected_max_drawdown closer to (but still under) "
                f"{TARGET_MAX_LOSS:.0%}, while keeping every QA score "
                f"≥ {PASS_THRESHOLD}. Prior critique for reference:\n"
                f"{evaluation.critique}"
            )
        else:
            base = (
                f"QA PASSED with average score {evaluation.average_score} and "
                f"expected_max_drawdown of {dd:.2%}, which is close to the "
                f"{TARGET_MAX_LOSS:.0%} target. Attempt one more refinement: try "
                f"to either raise expected_annual_return without exceeding "
                f"{TARGET_MAX_LOSS:.0%} drawdown, or improve the QA scores while "
                f"holding drawdown near {TARGET_MAX_LOSS:.0%}. Prior critique:\n"
                f"{evaluation.critique}"
            )

    return base


def _select_best_iteration(history: list[dict]) -> int | None:
    """
    Return the 1-based iteration index of the best passing iteration, or None
    if no iteration passed.  The orchestrator's no-pass fallback is to keep
    the FIRST iteration (see ``run_harness``): the critique-driven feedback
    loop tends to push the generator toward increasingly conservative
    portfolios (lower returns at similar drawdown), so when nothing passes,
    the unbiased first attempt is usually the most balanced result.

    Selection key (smaller is better):
      1. |expected_max_drawdown - TARGET_MAX_LOSS|   — closeness to the loss target
      2. expected_max_drawdown                       — prefer smaller drawdown on tie
      3. -average_score                              — prefer higher score on tie
    """
    passing = [h for h in history if h["passed"]]
    if not passing:
        return None
    best = min(
        passing,
        key=lambda h: (
            abs(h["expected_max_drawdown"] - TARGET_MAX_LOSS),
            h["expected_max_drawdown"],
            -h["average_score"],
        ),
    )
    return best["iteration"]


def _print_loss_floor_verdict(block: dict[str, Any], max_loss: float) -> None:
    """Print a clear console verdict for the deterministic loss-floor check."""
    if not block.get("performed"):
        reason = block.get("skipped_reason") or block.get("error") or "n/a"
        print(f"\n(Loss-floor check inconclusive — {reason}.)\n")
        return
    worst = block.get("worst_gross_annual_loss")
    wy = block.get("worst_year")
    if block.get("relies_on_unheld_mechanism"):
        print(
            f"\n❌  LOSS-FLOOR: delivered book lost {worst:.1%} GROSS in {wy} "
            f"(cap {max_loss:.0%}) and holds NO hedge leg — its ≤{max_loss:.0%} "
            f"claim relies on a mechanism that is NOT in the portfolio.\n"
        )
    elif block.get("organic_pass"):
        print(
            f"\n✅  LOSS-FLOOR: delivered book is ORGANICALLY within "
            f"{max_loss:.0%} (worst gross {worst:.1%} in {wy}).\n"
        )
    else:
        legs = ", ".join(block.get("hedge_legs") or [])
        print(
            f"\n⚠️  LOSS-FLOOR: delivered book breaches {max_loss:.0%} gross "
            f"({worst:.1%} in {wy}) but holds hedge leg(s): {legs}.\n"
        )


def _run_preservation(
    horizon_years: int,
    *,
    price: bool,
    risk: bool,
    correlation: bool,
    loss_floor: bool,
    capital: float,
) -> dict[str, Any]:
    """
    Build the SHORT-circuit (capital-preservation) result for a sub-floor
    horizon (``horizon_years < HORIZON_FLOOR_YEARS``).

    NO LLM agents run here — not the Planner, Generator, Evaluator, or
    Refiner.  We hand back the deterministic preservation template plus a
    redirect message, then run the NO-LLM steps (pricing, risk profile, and
    correlation snapshot) on the template's allocations so the user still
    gets a feasibility check, a downside picture, and a correlation view.
    The result dict mirrors the optimized-mode schema so every downstream
    reader (report, server, viz) can consume it uniformly: stubs / skipped
    blocks fill the LLM-only slots.
    """
    allocations = preservation_template(horizon_years)
    band = "<1y" if horizon_years < 1 else "1-3y"
    redirect = (
        f"Horizon of {horizon_years} year(s) is below the {HORIZON_FLOOR_YEARS}-year "
        f"floor for return optimisation. At this horizon, capital preservation "
        f"dominates: there is not enough time to recover from a drawdown, so the "
        f"optimizer is short-circuited and a fixed high-grade cash / short-bond "
        f"template is returned instead. To have a portfolio optimised for return, "
        f"use a horizon of at least {HORIZON_FLOOR_YEARS} years."
    )
    descriptions = {
        "SGOV": "iShares 0-3 Month Treasury ETF — T-bills / cash equivalent",
        "SHY":  "iShares 1-3 Year Treasury ETF — short-duration US government bonds",
        "VTIP": "Vanguard Short-Term TIPS ETF — short inflation-protected Treasuries",
    }
    descriptions = {t: descriptions.get(t, "") for t in allocations}

    print("\n" + "=" * 60)
    print(f"PRESERVATION SHORT-CIRCUIT — horizon {horizon_years}y < "
          f"{HORIZON_FLOOR_YEARS}y floor (no LLM agents run)")
    print("=" * 60)
    print(redirect + "\n")
    for ticker, weight in allocations.items():
        print(f"  {ticker:20s}  {weight:6.1%}")
    print()

    # final_proposal carries the template; expected_* are NULL — these are a
    # deterministic template, not a modeled / optimised book, so any return /
    # drawdown number would be fabricated.
    final_proposal = PortfolioProposal(
        allocations=allocations,
        descriptions=descriptions,
        expected_annual_return=None,
        expected_max_drawdown=None,
        methodology=(
            "Deterministic capital-preservation template (no optimisation). "
            "Selected by horizon band, not by the LLM pipeline."
        ),
        rationale=redirect,
        raw_text="",
    )
    spec_stub = InvestmentSpec(
        objective=(
            f"Capital preservation over a {horizon_years}-year horizon "
            f"(below the {HORIZON_FLOOR_YEARS}-year optimisation floor)."
        ),
        constraints="Preserve principal; no return optimisation at this horizon.",
        asset_universe="High-grade Treasuries, short TIPS, T-bills / cash.",
        risk_budget="Minimise drawdown; do not deploy growth-asset risk.",
        evaluation_criteria="N/A — deterministic template, not LLM-evaluated.",
        raw_text="",
    )
    final_evaluation_stub = EvaluationResult(
        passed=True,
        scores={},
        average_score=0.0,
        critique=(
            "Not QA-evaluated: capital-preservation template returned "
            "deterministically below the horizon floor."
        ),
        raw_text="",
    )

    # --- Pricing & risk are NO-LLM steps — run them on the template ---
    pricing_block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "capital": capital,
        "total_invested": 0.0,
        "leftover_cash": 0.0,
        "max_abs_drift": 0.0,
        "rows": [],
        "failed_tickers": [],
        "disclaimer": PRICING_DISCLAIMER,
        "source": "yfinance",
        "fetched_at": "",
        "error": None,
    }
    if not price:
        pricing_block["skipped_reason"] = "disabled via --no-prices / --test"
    else:
        pricing_result = run_pricing(allocations, capital)
        pricing_block = {
            "performed": True,
            "skipped_reason": None,
            **asdict(pricing_result),
        }

    risk_block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "disclaimer": RISK_DISCLAIMER,
        "error": None,
    }
    if not risk:
        risk_block["skipped_reason"] = "disabled via --no-risk / --test"
    else:
        risk_result = run_risk_profile(allocations, horizon_years=horizon_years)
        risk_block = {
            "performed": True,
            "skipped_reason": None,
            **asdict(risk_result),
        }

    correlation_block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "disclaimer": CORRELATION_DISCLAIMER,
        "error": None,
    }
    if not correlation:
        correlation_block["skipped_reason"] = "disabled via --no-correlation / --test"
    else:
        correlation_result = run_correlation(allocations)
        correlation_block = {
            "performed": True,
            "skipped_reason": None,
            **asdict(correlation_result),
        }

    loss_floor_block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "disclaimer": LOSS_FLOOR_DISCLAIMER,
        "error": None,
    }
    if not loss_floor:
        loss_floor_block["skipped_reason"] = "disabled via --no-loss-floor / --test"
    elif not allocations:
        loss_floor_block["skipped_reason"] = "no allocations to check"
    else:
        loss_floor_block = run_loss_floor_check(
            allocations, max_loss=TARGET_MAX_LOSS, descriptions=descriptions,
        )

    return {
        "model": api.MODEL,
        "max_iterations": MAX_ITERATIONS,
        "pass_threshold": PASS_THRESHOLD,
        "target_max_loss": TARGET_MAX_LOSS,
        "mode": "preservation",
        "horizon_years": horizon_years,
        "horizon_posture": PRESERVATION_POSTURE,
        "preservation_band": band,
        "redirect_message": redirect,
        "spec": asdict(spec_stub),
        "final_proposal": asdict(final_proposal),
        "final_evaluation": asdict(final_evaluation_stub),
        "selected_iteration": None,
        "selected_proposal": asdict(final_proposal),
        "selected_evaluation": asdict(final_evaluation_stub),
        "iteration_history": [],
        "refinement": {
            "performed": False,
            "skipped_reason": "preservation short-circuit — no optimisation loop",
            "promoted": False,
            "refined_proposal": None,
            "refined_evaluation": None,
            "improvements": None,
        },
        "pricing": pricing_block,
        "risk_profile": risk_block,
        "correlation": correlation_block,
        "loss_floor": loss_floor_block,
    }


def run_harness(
    user_goal: str,
    *,
    refine: bool = True,
    price: bool = True,
    risk: bool = True,
    correlation: bool = True,
    loss_floor: bool = True,
    capital: float = DEFAULT_CAPITAL,
    horizon_years: int = DEFAULT_HORIZON_YEARS,
) -> dict[str, Any]:
    """
    Main entry point.  Runs the full Planner → Generator ↔ Evaluator loop,
    then (by default) a post-selection REFINER pass, a PRICING / lot-size
    feasibility pass, a Monte-Carlo RISK profile, and a no-LLM CORRELATION
    snapshot.

    Flow:
      1. Planner expands the goal into a full investment spec.
      2. MAX_ITERATIONS rounds of Generator ↔ Evaluator.  Always runs
         all rounds; never breaks early.
      3. Select the best PASSING iteration whose expected_max_drawdown
         is closest to TARGET_MAX_LOSS.  Falls back to the first iteration
         if nothing passed.
      4. If `refine=True`, run a single Refiner pass on the selected
         portfolio: Refiner addresses every critique point, then the
         Evaluator re-scores.  If the refined version passes QA, it is
         PROMOTED to `final_proposal`; otherwise the selected one stays
         as the final answer.
      5. If `price=True`, fetch the latest price for each ticker in the
         final portfolio via yfinance and compute a whole-share lot-size
         feasibility check against `capital` USD.  Per-ticker failures
         (unknown ticker, network blip, model-invented pseudo-ticker) are
         recorded gracefully and never abort the pipeline.
      6. If `risk=True`, build a Monte-Carlo return-distribution profile
         (block-bootstrap of historical returns, long-history asset-class
         proxies for young ETFs) reporting median outcome / chance of
         ending down / unlucky tails over several holding horizons.  Also
         yfinance-backed and fail-soft.
      7. If `correlation=True`, compute a pairwise daily-return correlation
         snapshot (yfinance, no LLM) on the final portfolio, flagging
         highly redundant holdings (|ρ| >= 0.85).  Fail-soft.

    HORIZON (Phase 2):
      ``horizon_years`` is risk *capacity*.  Below ``HORIZON_FLOOR_YEARS``
      the whole optimisation loop is short-circuited to a deterministic
      capital-preservation template (mode="preservation", NO LLM agents).
      At or above the floor (mode="optimized") the glide-path posture for
      the horizon is computed deterministically, injected into the Planner
      as a growth-asset ceiling, and enforced by the Evaluator's
      deterministic ceiling check.  ``DEFAULT_HORIZON_YEARS`` (10) lands in
      the topmost band and reproduces today's output.
    """
    # --- Horizon gate: short-circuit below the floor (no LLM agents) ---
    if horizon_years < HORIZON_FLOOR_YEARS:
        return _run_preservation(
            horizon_years,
            price=price, risk=risk, correlation=correlation,
            loss_floor=loss_floor, capital=capital,
        )

    # --- Optimized regime: derive the glide-path posture deterministically ---
    posture = posture_for_horizon(horizon_years)
    growth_ceiling = posture["growth_ceiling"]
    print("\n" + "=" * 60)
    print(
        f"HORIZON {horizon_years}y → posture '{posture['label']}' "
        f"(band {posture['band']}, growth ceiling {growth_ceiling:.0%})"
    )
    print("=" * 60)

    # --- Step 1: Plan (horizon-aware via the injected INVESTOR PROFILE) ---
    spec = run_planner(user_goal, horizon_years=horizon_years, posture=posture)

    history: list[dict] = []
    proposals: list[PortfolioProposal] = []
    evaluations: list[EvaluationResult] = []
    feedback: str | None = None
    previous_proposal: PortfolioProposal | None = None

    for i in range(1, MAX_ITERATIONS + 1):
        # --- Step 2: Generate ---
        proposal = run_generator(
            spec, feedback=feedback, iteration=i, max_loss=TARGET_MAX_LOSS,
            previous_proposal=previous_proposal,
        )

        # --- Step 3: Evaluate ---
        evaluation = run_evaluator(
            spec, proposal,
            pass_threshold=PASS_THRESHOLD, max_loss=TARGET_MAX_LOSS,
            growth_ceiling=growth_ceiling,
        )

        proposals.append(proposal)
        evaluations.append(evaluation)

        history.append({
            "iteration": i,
            "allocations": proposal.allocations,
            "descriptions": proposal.descriptions,
            "expected_return": proposal.expected_annual_return,
            "expected_max_drawdown": proposal.expected_max_drawdown,
            "scores": evaluation.scores,
            "average_score": evaluation.average_score,
            "passed": evaluation.passed,
            "critique_snippet": evaluation.critique[:300],
            "selected": False,
        })

        print(f"\n--- Iteration {i} result ---")
        print(f"  Scores              : {evaluation.scores}")
        print(f"  Average             : {evaluation.average_score}")
        print(f"  Passed              : {evaluation.passed}")
        print(f"  Expected max loss   : {proposal.expected_max_drawdown:.2%}")
        print(f"  Target max loss     : {TARGET_MAX_LOSS:.0%}")

        if i == MAX_ITERATIONS:
            # No need to compute feedback after the last round.
            continue

        if evaluation.passed:
            dd = proposal.expected_max_drawdown
            if dd > TARGET_MAX_LOSS:
                print("✅  Passed QA but drawdown OVER target — pushing for lower drawdown next round.\n")
            elif dd < UNDER_UTILISATION_BAND:
                print("✅  Passed QA but drawdown WELL UNDER target — pushing for more risk budget use next round.\n")
            else:
                print("✅  Passed QA and drawdown near target — attempting one more refinement.\n")
        else:
            print("❌  Failed QA — feeding critique back to generator.\n")

        # Carry BOTH the critique and the portfolio it refers to into the next
        # round (set together so they always describe the same iteration).
        feedback = _build_feedback(evaluation, proposal)
        previous_proposal = proposal

    # --- Step 4: Select the best passing iteration ---
    selected_idx = _select_best_iteration(history)

    if selected_idx is not None:
        history[selected_idx - 1]["selected"] = True
        selected_proposal = proposals[selected_idx - 1]
        selected_evaluation = evaluations[selected_idx - 1]
        print(
            f"\n🎯  Selected iteration {selected_idx} of {MAX_ITERATIONS}: "
            f"passing portfolio with expected_max_drawdown "
            f"{selected_proposal.expected_max_drawdown:.2%} "
            f"(closest to {TARGET_MAX_LOSS:.0%} target).\n"
        )
    else:
        # Fall back to the FIRST iteration when nothing passed.  The
        # critique-driven feedback loop tends to push later iterations
        # toward increasingly conservative portfolios (lower returns at
        # similar drawdown).  When no iteration passes QA there is no
        # "objectively better" one — so we keep the unbiased first
        # attempt, which is usually the most balanced result.
        history[0]["selected"] = True
        selected_proposal = proposals[0]
        selected_evaluation = evaluations[0]
        print(
            f"\n⚠️  Reached max iterations ({MAX_ITERATIONS}) without any "
            f"passing portfolio. Returning iteration 1 (first attempt — "
            f"least biased by feedback-driven over-correction).\n"
        )

    # --- Step 5: Post-selection Refinement ---
    final_proposal = selected_proposal
    final_evaluation = selected_evaluation
    refinement_block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "promoted": False,
        "refined_proposal": None,
        "refined_evaluation": None,
        "improvements": None,
    }

    if not refine:
        refinement_block["skipped_reason"] = "disabled via --no-refine / --test"
        print("\n(Refinement step skipped.)\n")
    elif not selected_evaluation.critique:
        refinement_block["skipped_reason"] = "no critique available to refine against"
        print("\n(Refinement step skipped — no critique available.)\n")
    else:
        refined_proposal = run_refiner(
            spec, selected_proposal, selected_evaluation,
            max_iterations=MAX_ITERATIONS, max_loss=TARGET_MAX_LOSS,
        )
        refined_evaluation = run_evaluator(
            spec, refined_proposal,
            pass_threshold=PASS_THRESHOLD, max_loss=TARGET_MAX_LOSS,
            growth_ceiling=growth_ceiling,
        )

        print(f"\n--- Refinement result ---")
        print(f"  Scores              : {refined_evaluation.scores}")
        print(f"  Average             : {refined_evaluation.average_score}")
        print(f"  Passed              : {refined_evaluation.passed}")
        print(f"  Expected max loss   : {refined_proposal.expected_max_drawdown:.2%}")

        improvements = _refinement_improvements(
            selected_evaluation, refined_evaluation,
            selected_proposal, refined_proposal,
        )

        # Promote the refined version only if it passes QA AND its drawdown
        # is still within the target (we don't want refinement to wander out
        # of the risk budget while "fixing" things).
        refined_dd = float(refined_proposal.expected_max_drawdown or 0)
        promote = (
            refined_evaluation.passed
            and refined_dd <= TARGET_MAX_LOSS
        )

        refinement_block.update({
            "performed": True,
            "promoted": promote,
            "refined_proposal": asdict(refined_proposal),
            "refined_evaluation": asdict(refined_evaluation),
            "improvements": improvements,
        })

        if promote:
            final_proposal = refined_proposal
            final_evaluation = refined_evaluation
            print(
                f"\n✨  Refined portfolio PROMOTED to final "
                f"(passed QA, drawdown {refined_dd:.2%} ≤ {TARGET_MAX_LOSS:.0%}).\n"
            )
        else:
            reasons = []
            if not refined_evaluation.passed:
                reasons.append("failed QA")
            if refined_dd > TARGET_MAX_LOSS:
                reasons.append(f"drawdown {refined_dd:.2%} > target")
            print(
                f"\n🔒  Refined version NOT promoted ({', '.join(reasons)}). "
                f"Keeping selected iteration {selected_idx} as final.\n"
            )

    # --- Step 6: Pricing & lot-size feasibility (yfinance) ---
    pricing_block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "capital": capital,
        "total_invested": 0.0,
        "leftover_cash": 0.0,
        "max_abs_drift": 0.0,
        "rows": [],
        "failed_tickers": [],
        "disclaimer": PRICING_DISCLAIMER,
        "source": "yfinance",
        "fetched_at": "",
        "error": None,
    }
    if not price:
        pricing_block["skipped_reason"] = "disabled via --no-prices / --test"
        print("\n(Pricing step skipped.)\n")
    elif not final_proposal.allocations:
        pricing_block["skipped_reason"] = "no allocations to price"
        print("\n(Pricing step skipped — no allocations.)\n")
    else:
        pricing_result = run_pricing(final_proposal.allocations, capital)
        pricing_block = {
            "performed": True,
            "skipped_reason": None,
            **asdict(pricing_result),
        }

    # --- Step 7: Return-distribution risk profile (yfinance + Monte-Carlo) ---
    risk_block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "disclaimer": RISK_DISCLAIMER,
        "error": None,
    }
    if not risk:
        risk_block["skipped_reason"] = "disabled via --no-risk / --test"
        print("\n(Risk-profile step skipped.)\n")
    elif not final_proposal.allocations:
        risk_block["skipped_reason"] = "no allocations to model"
        print("\n(Risk-profile step skipped — no allocations.)\n")
    else:
        risk_result = run_risk_profile(final_proposal.allocations, horizon_years=horizon_years)
        risk_block = {
            "performed": True,
            "skipped_reason": None,
            **asdict(risk_result),
        }

    # --- Step 8: Pairwise-correlation snapshot (yfinance, no LLM) ---
    correlation_block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "disclaimer": CORRELATION_DISCLAIMER,
        "error": None,
    }
    if not correlation:
        correlation_block["skipped_reason"] = "disabled via --no-correlation / --test"
        print("\n(Correlation step skipped.)\n")
    elif not final_proposal.allocations:
        correlation_block["skipped_reason"] = "no allocations to correlate"
        print("\n(Correlation step skipped — no allocations.)\n")
    else:
        correlation_result = run_correlation(final_proposal.allocations)
        correlation_block = {
            "performed": True,
            "skipped_reason": None,
            **asdict(correlation_result),
        }

    # --- Step 9: Loss-floor compliance (deterministic backtest of the
    # DELIVERED book; flags a ≤cap claim that rests on an un-held hedge) ---
    loss_floor_block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "disclaimer": LOSS_FLOOR_DISCLAIMER,
        "error": None,
    }
    if not loss_floor:
        loss_floor_block["skipped_reason"] = "disabled via --no-loss-floor / --test"
        print("\n(Loss-floor check skipped.)\n")
    elif not final_proposal.allocations:
        loss_floor_block["skipped_reason"] = "no allocations to check"
        print("\n(Loss-floor check skipped — no allocations.)\n")
    else:
        loss_floor_block = run_loss_floor_check(
            final_proposal.allocations,
            max_loss=TARGET_MAX_LOSS,
            descriptions=final_proposal.descriptions,
        )
        _print_loss_floor_verdict(loss_floor_block, TARGET_MAX_LOSS)

    return {
        "model": api.MODEL,
        "max_iterations": MAX_ITERATIONS,
        "pass_threshold": PASS_THRESHOLD,
        "target_max_loss": TARGET_MAX_LOSS,
        "mode": "optimized",
        "horizon_years": horizon_years,
        "horizon_posture": posture["label"],
        "spec": asdict(spec),
        "final_proposal": asdict(final_proposal),
        "final_evaluation": asdict(final_evaluation),
        "selected_iteration": selected_idx,
        "selected_proposal": asdict(selected_proposal),
        "selected_evaluation": asdict(selected_evaluation),
        "iteration_history": history,
        "refinement": refinement_block,
        "pricing": pricing_block,
        "risk_profile": risk_block,
        "correlation": correlation_block,
        "loss_floor": loss_floor_block,
    }


# ---------------------------------------------------------------------------
# Human-readable Markdown report writer — moved to report.py
# ---------------------------------------------------------------------------
# The writer lives in `report.py` and is imported at the top of this file.
# It reads everything it needs from the result dict (including model,
# max_iterations, pass_threshold, target_max_loss) so it has zero
# dependency on this module's mutable globals.



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    _MODEL_ALIASES: dict[str, str] = {
        "haiku":  "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-6",
        "opus":   "claude-opus-4-7",
        "fable":  "claude-fable-5",
    }

    parser = argparse.ArgumentParser(
        description="Portfolio Optimization Harness — Planner → Generator ↔ Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              uv run python harness.py
              uv run python harness.py --test
              uv run python harness.py --model sonnet
              uv run python harness.py --model haiku --iterations 1
              uv run python harness.py --reasoning-model fable
              uv run python harness.py --no-refine
              uv run python harness.py --no-prices
              uv run python harness.py --no-risk
              uv run python harness.py --no-correlation
              uv run python harness.py --capital 250000
              uv run python harness.py --max-loss 0.10
              uv run python harness.py --horizon-years 5
              uv run python harness.py --horizon-years 2
        """),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Test mode: use claude-haiku-4-5-20251001 with 1 iteration to smoke-test "
            "the full pipeline quickly and cheaply. Useful when Opus is overloaded or "
            "you just want to verify the plumbing."
        ),
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help=(
            "Override the model for ALL agents. Accepts short aliases "
            "(haiku, sonnet, opus, fable) or a full Anthropic model ID. "
            "Ignored when --test is also set."
        ),
    )
    parser.add_argument(
        "--reasoning-model",
        metavar="MODEL",
        help=(
            "Override the model for ONLY the heavy reasoning agents "
            "(Generator / Evaluator / Refiner), leaving the Planner on its "
            "own model. Same aliases as --model (haiku, sonnet, opus, fable) "
            "or a full model ID. Use this to A/B a reasoning model "
            "(e.g. --reasoning-model fable) without dragging the Planner into "
            "it. Ignored under --test; if combined with --model, this wins "
            "for the heavy agents."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        metavar="N",
        help=(
            "Override the number of generator ↔ evaluator rounds (default "
            f"{MAX_ITERATIONS}). Ignored when --test is also set."
        ),
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help=(
            "Skip the post-selection Refiner pass. By default, after the best "
            "iteration is selected, a Refiner agent addresses every critique "
            "point and the Evaluator re-scores; --no-refine disables that. "
            "--test implies --no-refine."
        ),
    )
    parser.add_argument(
        "--no-correlation",
        action="store_true",
        help=(
            "Skip the post-selection Correlation snapshot. By default, after "
            "the final portfolio is determined, a no-LLM step computes the "
            "pairwise daily-return correlations of the holdings from Yahoo "
            "Finance history and flags highly redundant pairs (|ρ| >= 0.85). "
            "yfinance-backed and fail-soft. --test implies --no-correlation."
        ),
    )
    parser.add_argument(
        "--no-prices",
        action="store_true",
        help=(
            "Skip the post-selection Pricing pass. By default, after the "
            "final portfolio is determined, the latest prices for each "
            "ticker are fetched from Yahoo Finance via yfinance and a "
            "whole-share lot-size feasibility check is performed using "
            "--capital. Per-ticker failures degrade gracefully. "
            "--test implies --no-prices."
        ),
    )
    parser.add_argument(
        "--no-risk",
        action="store_true",
        help=(
            "Skip the post-selection Risk-profile pass. By default, after "
            "the final portfolio is determined, a Monte-Carlo "
            "return-distribution profile is built (block-bootstrap of "
            "historical returns, with long-history asset-class proxies for "
            "young ETFs) reporting median outcome / chance of ending down / "
            "unlucky tails per holding horizon. yfinance-backed and "
            "fail-soft. --test implies --no-risk."
        ),
    )
    parser.add_argument(
        "--no-loss-floor",
        action="store_true",
        help=(
            "Skip the post-selection Loss-floor compliance check. By default, "
            "after the final portfolio is determined, a no-LLM step backtests "
            "the DELIVERED holdings over the calendar-year stress windows "
            "(2008/2020/2022, proxy-substituted) and flags when the book's "
            "≤max-loss claim relies on a hedge it does not actually hold. "
            "yfinance-backed and fail-soft. --test implies --no-loss-floor."
        ),
    )
    parser.add_argument(
        "--max-loss",
        type=float,
        metavar="FRACTION",
        help=(
            "Override the max annual loss budget, as a fraction in (0, 1) "
            f"— e.g. 0.10 for 10%% (default {TARGET_MAX_LOSS * 100:.0f}%%). "
            "Drives the Generator/Evaluator/Refiner prompts, the selection "
            "target, and the under-utilisation band (which auto-derives as "
            f"{UNDER_UTILISATION_RATIO * 100:.0f}%% of this value). Applies "
            "in --test mode too."
        ),
    )
    parser.add_argument(
        "--horizon-years",
        type=int,
        default=DEFAULT_HORIZON_YEARS,
        metavar="N",
        help=(
            f"Investment horizon in whole years (default {DEFAULT_HORIZON_YEARS}). "
            f"Horizon is risk CAPACITY and binds independently of --max-loss "
            f"(risk tolerance) — the more conservative of the two wins. At or "
            f"above the {HORIZON_FLOOR_YEARS}-year floor the glide-path posture "
            f"caps growth-asset weight (3-5y→50%%, 5-10y→75%%, 10y+→90%%) and the "
            f"full pipeline runs. Below the floor the optimizer is short-circuited "
            f"to a deterministic capital-preservation template (no LLM agents). "
            f"Must be >= 1."
        ),
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        metavar="USD",
        help=(
            f"Assumed investable capital in USD for the lot-size "
            f"feasibility check (default ${DEFAULT_CAPITAL:,.0f}). Only "
            f"used when pricing is enabled."
        ),
    )
    args = parser.parse_args()

    # --horizon-years validates in both --test and normal modes (it changes
    # the recommended portfolio, not just a model/iteration knob).  Reject
    # < 1 with a clear error; the run_harness floor handles the < 3 case.
    if args.horizon_years < 1:
        parser.error("--horizon-years must be >= 1 (whole years)")
    horizon_years = args.horizon_years

    # Apply --test first (it takes precedence), then individual overrides.
    refine = not args.no_refine
    price = not args.no_prices
    risk = not args.no_risk
    correlation = not args.no_correlation
    loss_floor = not args.no_loss_floor
    capital = args.capital
    if args.test:
        # --test forces ALL agents to haiku (and disables everything else).
        # Per-agent model globals live in api.py — patch them there so
        # call_claude picks them up on the next call.
        api.MODEL = _MODEL_ALIASES["haiku"]
        api.PLANNER_MODEL = _MODEL_ALIASES["haiku"]
        MAX_ITERATIONS = 1
        refine = False        # --test always implies --no-refine
        price = False         # --test always implies --no-prices
        risk = False          # --test always implies --no-risk
        correlation = False   # --test always implies --no-correlation
        loss_floor = False    # --test always implies --no-loss-floor
        print(
            f"[TEST MODE] model={api.MODEL}  iterations={MAX_ITERATIONS}  "
            f"refine={refine}  price={price}  risk={risk}  "
            f"correlation={correlation}  loss_floor={loss_floor}"
        )
    else:
        if args.model:
            # --model X overrides ALL agents to X (escape hatch — runs
            # the entire pipeline on one tier, useful for cost / quality
            # comparison or when one tier is overloaded).
            resolved = _MODEL_ALIASES.get(args.model, args.model)
            api.MODEL = resolved
            api.PLANNER_MODEL = resolved
            print(f"[model override — all agents] {api.MODEL}")
        else:
            # No override — show the per-agent assignments so the user
            # knows the pipeline isn't uniform.
            print(
                f"[per-agent models] generator/evaluator/refiner={api.MODEL}  "
                f"planner={api.PLANNER_MODEL}"
            )
        # --reasoning-model overrides ONLY the heavy agents (Generator /
        # Evaluator / Refiner) by patching api.MODEL, leaving api.PLANNER_MODEL
        # untouched — so a reasoning model (e.g. Fable 5) can be A/B'd without
        # dragging the Planner into it.  Applied AFTER --model so it wins for
        # the heavy agents.
        if args.reasoning_model:
            resolved_rm = _MODEL_ALIASES.get(args.reasoning_model, args.reasoning_model)
            api.MODEL = resolved_rm
            print(
                f"[reasoning-model override — generator/evaluator/refiner] "
                f"{api.MODEL}  (planner stays {api.PLANNER_MODEL})"
            )
        if args.iterations is not None:
            if args.iterations < 1:
                parser.error("--iterations must be ≥ 1")
            MAX_ITERATIONS = args.iterations
            print(f"[iterations override] {MAX_ITERATIONS}")
        if capital <= 0:
            parser.error("--capital must be > 0")
        if not refine:
            print("[refinement disabled]")
        if not price:
            print("[pricing disabled]")
        else:
            print(f"[pricing enabled  capital=${capital:,.0f}]")
        if not risk:
            print("[risk profile disabled]")
        if not correlation:
            print("[correlation disabled]")

    # --max-loss applies in both --test and normal modes (orthogonal to
    # model / iteration overrides).  Patching TARGET_MAX_LOSS here means
    # run_harness threads it into every agent prompt and the selection
    # target; recompute the auto-derived under-utilisation band to match.
    if args.max_loss is not None:
        if not 0 < args.max_loss < 1:
            parser.error("--max-loss must be a fraction in (0, 1), e.g. 0.10")
        TARGET_MAX_LOSS = args.max_loss
        UNDER_UTILISATION_BAND = UNDER_UTILISATION_RATIO * TARGET_MAX_LOSS
        print(
            f"[max-loss override] {TARGET_MAX_LOSS:.0%} "
            f"(under-utilisation band {UNDER_UTILISATION_BAND:.0%})"
        )

    goal = (
        "Optimise a portfolio for a US-based retail investor. "
        "Maximise annual return while ensuring the portfolio can lose "
        f"no more than {TARGET_MAX_LOSS:.0%} of its invested value in any "
        "given year. Use any instruments available to a typical retail "
        "investor (stocks, ETFs, bonds, options overlays, etc.)."
    )

    print(f"[horizon] {horizon_years} year(s)")

    result = run_harness(
        goal,
        refine=refine,
        price=price,
        risk=risk,
        correlation=correlation,
        loss_floor=loss_floor,
        capital=capital,
        horizon_years=horizon_years,
    )

    # Dump full trace to a JSON file for inspection
    with open("harness_output.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Also write a human-readable Markdown report
    write_markdown_report(result, "harness_output.md")

    print("\n" + "=" * 60)
    print("FINAL PORTFOLIO")
    print("=" * 60)
    mode = result.get("mode", "optimized")
    print(f"  Mode               : {mode}")
    print(
        f"  Horizon            : {result.get('horizon_years')} year(s) "
        f"— {result.get('horizon_posture')}"
    )
    if mode == "preservation":
        print("  (Capital-preservation short-circuit — no optimisation loop ran.)")
    else:
        sel = result.get("selected_iteration")
        if sel is not None:
            print(f"  Selected iteration : {sel} of {MAX_ITERATIONS} (best passing)")
        else:
            print(f"  Selected iteration : last of {MAX_ITERATIONS} (no iteration passed)")
    print(f"  Target max loss    : {result['target_max_loss']:.1%}\n")

    if result["final_proposal"]:
        for ticker, weight in result["final_proposal"]["allocations"].items():
            print(f"  {ticker:20s}  {weight:6.1%}")
        exp_ret = result["final_proposal"]["expected_annual_return"]
        exp_dd = result["final_proposal"]["expected_max_drawdown"]
        # Preservation mode leaves these NULL (a deterministic template, not
        # a modeled book) — print "n/a" rather than crash on the format spec.
        print(f"\n  Expected return    : {exp_ret:.1%}" if exp_ret is not None else "\n  Expected return    : n/a")
        print(f"  Expected max loss  : {exp_dd:.1%}" if exp_dd is not None else "  Expected max loss  : n/a")
    print(f"\n  Passed QA          : {result['final_evaluation']['passed']}")
    print(f"  Final avg score    : {result['final_evaluation']['average_score']}")

    print("\n  Iteration summary:")
    for h in result["iteration_history"]:
        mark = "★" if h["selected"] else " "
        passed_str = "PASS" if h["passed"] else "FAIL"
        print(
            f"   {mark} #{h['iteration']}  "
            f"avg={h['average_score']:.2f}  "
            f"max_loss={h['expected_max_drawdown']:.2%}  "
            f"{passed_str}"
        )

    pricing_summary = result.get("pricing") or {}
    if pricing_summary.get("performed"):
        n_failed = len(pricing_summary.get("failed_tickers") or [])
        print("\n  Lot-size feasibility (yfinance):")
        print(
            f"    capital=${pricing_summary.get('capital', 0):,.0f}  "
            f"invested=${pricing_summary.get('total_invested', 0):,.2f}  "
            f"leftover=${pricing_summary.get('leftover_cash', 0):,.2f}  "
            f"max|Δw|={pricing_summary.get('max_abs_drift', 0):.2%}  "
            f"unpriced={n_failed}"
        )

    print("\nFull trace saved to:")
    print("  - harness_output.json   (machine-readable)")
    print("  - harness_output.md     (human-readable)")
