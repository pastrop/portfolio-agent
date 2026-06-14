"""
loss_floor.py — deterministic check that the DELIVERED portfolio meets its
annual loss floor *on its own*, without crediting a hedge it does not hold.

Why this exists
---------------
The pipeline judges a portfolio against a hard annual loss floor (``--max-loss``,
default 5%).  When the Planner's spec defines an ``enforcement_mechanism`` (e.g.
a protective put overlay), the LLM Evaluator is allowed to credit that mechanism
when scoring constraint compliance (see ``agents.run_evaluator``).  That is fine
for mechanisms the book can actually execute with the assets it HOLDS (a dynamic
de-risking / rebalancing rule).  It is NOT fine when the mechanism is an
*instrument overlay the portfolio never buys* — a "SPY put overlay" that appears
only in the spec's prose and the Evaluator's arithmetic, while the delivered
``allocations`` are 100% long-only ETFs.  In that case the headline "≤5%" is
certified against a hedge that isn't in the deliverable, and a user who buys the
listed holdings is unprotected.

This module is the deterministic ground truth: it backtests the DELIVERED
holdings over the standard calendar-year stress windows (proxy-substituted so
pre-inception years have coverage), reports the worst gross annual loss, detects
whether any actual hedge leg is held, and flags ``relies_on_unheld_mechanism``
when the book breaches the floor gross AND holds no hedge.

Self-contained and FAIL-SOFT, like ``pricing.py`` / ``risk.py`` / ``correlation.py``:
it takes ``allocations`` directly, never raises, and degrades to a skipped block
on any error.  No LLM.

Public surface: ``run_loss_floor_check`` and ``LOSS_FLOOR_DISCLAIMER``.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from tools import compute_backtest

# Reuse risk.py's long-history proxy map so the 2008 window has coverage
# (young ETFs like SGOV/SCHP didn't exist in 2008).  Import the data constant
# only — keeps this module's dependency surface tiny.  Fail-soft on import.
try:
    from risk import RISK_PROXY_MAP
except Exception:                       # pragma: no cover - risk should import
    RISK_PROXY_MAP = {}

# Calendar-year stress windows.  The floor is an ANNUAL (Jan 1–Dec 31) loss
# cap, so we judge each year's total return — NOT the intra-year max drawdown.
STRESS_YEARS = (2008, 2020, 2022)
_MIN_COVERAGE = 0.90                    # a year gappier than this can't be judged

# A holding counts as a real hedge leg ONLY when it is an actual OPTIONS /
# derivative downside hedge — the kind a spec "put overlay" refers to.  The
# matching is deliberately strict because the SAFE direction is to UNDER-
# detect: a missed hedge only makes the gate flag MORE, while a false match
# would wrongly EXEMPT a book from the gate.  In particular, colloquial
# "hedges" must NOT count — gold is routinely called a "tail hedge"
# ("GLD ... geopolitical tail hedge"), commodities an "oil-shock hedge",
# TIPS an "inflation hedge"; none of those is an options overlay.
_HEDGE_TICKER_TOKENS = ("PUT", "CALL", "OPTION", "OVERLAY", "COLLAR",
                        "SPREAD", "OTM")
# Dedicated options-based downside-protection ETFs (they actually hold puts /
# structured protection).  Kept narrow on purpose.
_HEDGE_ETFS = frozenset({"TAIL", "SWAN", "PHDG"})
# Phrases that denote an actual OPTIONS position — each requires "put" or an
# explicit "options …" construct, so gold/commodity/TIPS "hedge" prose can't
# match.
_HEDGE_DESC_PHRASES = ("put option", "protective put", "put spread",
                       "put-spread", "put overlay", "index put", "spx put",
                       "spy put", "options overlay", "options collar")

LOSS_FLOOR_DISCLAIMER = (
    "Loss-floor compliance is a yfinance-backed backtest of the DELIVERED "
    "holdings (proxy-substituted for pre-inception years), judged on "
    "calendar-year total return vs the loss cap. It reflects the book AS "
    "LISTED — it does NOT credit any options overlay or de-risking rule the "
    "portfolio does not actually hold. 'relies_on_unheld_mechanism' = the "
    "book breaches the cap gross and holds no hedge leg, so its ≤cap claim "
    "depends on a mechanism that is not in the deliverable."
)


def _hedge_legs(allocations: dict, descriptions: dict | None) -> list[str]:
    """Tickers (positive weight) that look like an actual hedge/options leg."""
    legs: list[str] = []
    for tk, w in (allocations or {}).items():
        try:
            if float(w) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        upper = str(tk).upper()
        desc = str((descriptions or {}).get(tk, "")).lower()
        if (upper in _HEDGE_ETFS
                or any(tok in upper for tok in _HEDGE_TICKER_TOKENS)
                or any(p in desc for p in _HEDGE_DESC_PHRASES)):
            legs.append(tk)
    return legs


def _proxy_weights(allocations: dict) -> tuple[dict[str, float], list[dict]]:
    """Aggregate positive weights onto long-history proxy tickers for coverage."""
    weights: dict[str, float] = {}
    subs: list[dict] = []
    for tk, w in allocations.items():
        try:
            wf = float(w)
        except (TypeError, ValueError):
            continue
        if wf <= 0:
            continue
        proxy = RISK_PROXY_MAP.get(tk, tk)
        if proxy != tk:
            subs.append({"original": tk, "proxy": proxy})
        weights[proxy] = weights.get(proxy, 0.0) + wf
    return weights, subs


def run_loss_floor_check(
    allocations: dict,
    max_loss: float = 0.05,
    descriptions: dict | None = None,
) -> dict[str, Any]:
    """
    Backtest the DELIVERED ``allocations`` over the calendar-year stress
    windows and judge organic compliance with the ``max_loss`` annual floor.

    Returns a self-describing block (same shape family as ``risk_profile`` /
    ``correlation``).  Key fields:
      * ``worst_gross_annual_loss`` / ``worst_year`` — worst Jan-Dec total
        return across the reliably-covered stress years (the gross number).
      * ``organic_pass`` — worst gross annual loss is within the cap WITHOUT
        any hedge (i.e. the listed book meets the floor on its own).
      * ``holds_hedge_leg`` / ``hedge_legs`` — whether the book actually holds
        an options/hedge instrument.
      * ``relies_on_unheld_mechanism`` — True when the book breaches the cap
        gross AND holds no hedge: the ≤cap claim rests on a mechanism not in
        the deliverable. THIS IS THE BUG-FLAG.

    Never raises; on any failure returns the block with ``error`` set and the
    verdict fields left None so callers degrade gracefully.
    """
    block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "max_loss": max_loss,
        "stress_years": list(STRESS_YEARS),
        "per_year": [],
        "worst_gross_annual_loss": None,
        "worst_year": None,
        "holds_hedge_leg": False,
        "hedge_legs": [],
        "organic_pass": None,
        "relies_on_unheld_mechanism": None,
        "proxy_substitutions": [],
        "disclaimer": LOSS_FLOOR_DISCLAIMER,
        "source": "yfinance via tools.compute_backtest",
        "computed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "error": None,
    }
    try:
        if not allocations:
            block["skipped_reason"] = "no allocations to check"
            return block

        legs = _hedge_legs(allocations, descriptions)
        block["hedge_legs"] = legs
        block["holds_hedge_leg"] = bool(legs)

        weights, subs = _proxy_weights(allocations)
        block["proxy_substitutions"] = subs
        if not weights:
            block["skipped_reason"] = "no positive-weight holdings"
            return block

        per_year: list[dict[str, Any]] = []
        for year in STRESS_YEARS:
            try:
                res = compute_backtest({
                    "weights": weights,
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                })
            except Exception as exc:
                per_year.append({"year": year, "ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            cov = float(res.get("coverage_weight", 0.0) or 0.0)
            tr = res.get("total_return")
            per_year.append({
                "year": year,
                "ok": cov >= _MIN_COVERAGE and tr is not None,
                "gross_annual_return": tr,
                "gross_max_drawdown": res.get("max_drawdown"),
                "coverage_weight": cov,
                "breaches_cap": (tr is not None and tr < -abs(max_loss)),
            })
        block["per_year"] = per_year

        reliable = [p for p in per_year if p.get("ok")]
        if not reliable:
            block["performed"] = True
            block["skipped_reason"] = (
                "no stress year had adequate data coverage to judge"
            )
            return block

        worst = min(reliable, key=lambda p: p["gross_annual_return"])
        block["worst_gross_annual_loss"] = worst["gross_annual_return"]
        block["worst_year"] = worst["year"]
        organic_pass = worst["gross_annual_return"] >= -abs(max_loss)
        block["organic_pass"] = organic_pass
        block["relies_on_unheld_mechanism"] = bool(
            (not organic_pass) and (not block["holds_hedge_leg"])
        )
        block["performed"] = True
        return block
    except Exception as exc:                 # fail-soft: never raise
        block["error"] = f"{type(exc).__name__}: {exc}"
        block["skipped_reason"] = "loss-floor check failed (see error)"
        return block
