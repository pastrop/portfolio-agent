"""
whatif.py — deterministic portfolio risk analysis (no LLM, no API key).

Two modes:
  • EDIT mode    — take a finished ``harness_output.json``, edit the final
                   portfolio (substitute / set / remove a security), and report
                   the DETERMINISTIC before/after deltas (return, drawdown, and
                   the all-important loss-cap verdict).
  • DIRECT mode  — analyze a portfolio YOU supply (``--portfolio`` inline or
                   ``--portfolio-file``) and emit a standalone risk report. No
                   baseline to compare against; no comparison produced.

Standalone, like ``viz/ui_designer.py``: it reads its inputs as black boxes and
never touches the harness.  It DOES reuse the harness's allocation-first,
LLM-free eval core (``loss_floor`` / ``risk`` / ``correlation`` / ``pricing`` /
``tools.compute_backtest``) and ``report.py``'s section renderers.  Both modes
emit a STANDARD harness-schema ``.json`` so ``viz/ui_designer.py`` renders it
like any run.  No API key needed.

Design + locked decisions: see SESSION_NOTES.md → "What-If scenario tool".

Usage:
    # EDIT mode (before/after)
    uv run python whatif/whatif.py harness_output.json --substitute SPY=NVDA -o nvda
    uv run python whatif/whatif.py harness_output.json --set GLD=0.10 --remove VNQ
    # DIRECT mode (standalone) — inline or file
    uv run python whatif/whatif.py --portfolio "SCHD=0.24,VTV=0.20,SGOV=0.56" -o my_book
    uv run python whatif/whatif.py --portfolio-file my_book.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import asdict
from typing import Any

# --- bootstrap: this module lives in whatif/ but reuses the repo-root eval
# core, so put the repo root on sys.path (viz/ui_designer.py doesn't need this
# because it imports nothing from the harness; we do). ----------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from correlation import run_correlation                       # noqa: E402
from loss_floor import run_loss_floor_check                   # noqa: E402
from pricing import DEFAULT_CAPITAL, run_pricing              # noqa: E402
from report import (                                          # noqa: E402
    _push_correlation_section,
    _push_loss_floor_section,
    _push_pricing_section,
    _push_risk_section,
)
from risk import run_risk_profile                             # noqa: E402
from tools import compute_backtest                            # noqa: E402

_WEIGHT_EPS = 1e-9


# ===========================================================================
# Single-name (individual stock) detection — fail-soft yfinance metadata
# ===========================================================================
def _security_types(tickers: set[str]) -> dict[str, str]:
    """
    Map each ticker -> its yfinance ``quoteType`` (``EQUITY`` | ``ETF`` |
    ``MUTUALFUND`` | ``UNKNOWN``).  One lookup over the union of tickers; fully
    fail-soft (offline / unknown ticker -> ``UNKNOWN``).  This lives HERE, not
    in pricing.py — pricing uses ``fast_info`` (no quoteType), and a what-if-only
    concern shouldn't slow the harness.
    """
    tickers = {t for t in tickers if t}
    if not tickers:
        return {}
    try:
        import yfinance as yf
    except Exception:
        return {t: "UNKNOWN" for t in tickers}
    out: dict[str, str] = {}
    for t in sorted(tickers):
        try:
            info = yf.Ticker(t).info or {}
            out[t] = str(info.get("quoteType") or "UNKNOWN").upper()
        except Exception:
            out[t] = "UNKNOWN"
    return out


def _is_single_name(quote_type: str) -> bool:
    """An individual company stock (the thing a fund-for-stock swap introduces)."""
    return quote_type == "EQUITY"


# ===========================================================================
# Concentration (pure-python; effective-N is breadth, single-name is the
# fund-vs-stock story effective-N is blind to)
# ===========================================================================
def _concentration(allocations: dict, qtypes: dict[str, str]) -> dict[str, Any]:
    pos = {t: float(w) for t, w in allocations.items() if float(w) > _WEIGHT_EPS}
    hhi = sum(w * w for w in pos.values())
    max_t = max(pos, key=pos.get) if pos else None
    singles = {t: w for t, w in pos.items()
               if _is_single_name(qtypes.get(t, "UNKNOWN"))}
    return {
        "effective_n": round(1.0 / hhi, 2) if hhi > 0 else 0.0,
        "max_name": max_t,
        "max_name_weight": round(pos.get(max_t, 0.0), 4) if max_t else 0.0,
        "single_name_weight": round(sum(singles.values()), 4),
        "single_names": sorted(singles),
    }


# ===========================================================================
# 1. inputs — a baseline run (edit mode) OR a portfolio you supply (direct mode)
# ===========================================================================
def load_baseline(path: str) -> dict:
    """Read a completed ``harness_output.json`` (READ-ONLY)."""
    with open(path) as f:
        return json.load(f)


def _normalize_weights(raw: dict) -> tuple[dict, list[str]]:
    """Upper-case tickers, drop ≤0 weights, renormalize to 1.0 (never silent —
    any rescale lands in ``notes``).  Weights are fractions."""
    alloc: dict[str, float] = {}
    for t, w in raw.items():
        try:
            wf = float(w)
        except (TypeError, ValueError):
            raise ValueError(f"weight for {t!r} must be a number, got {w!r}")
        if wf < 0:
            raise ValueError(f"weight for {t!r} is negative ({wf})")
        if wf <= _WEIGHT_EPS:
            continue
        tk = str(t).strip().upper()
        alloc[tk] = alloc.get(tk, 0.0) + wf
    if not alloc:
        raise ValueError("portfolio has no positive-weight holdings")
    notes: list[str] = []
    s = sum(alloc.values())
    if abs(s - 1.0) > 1e-6:
        for t in alloc:
            alloc[t] /= s
        notes.append(f"renormalized to 100% (input summed to {s:.2f})")
    return {t: round(w, 6) for t, w in alloc.items()}, notes


def parse_portfolio_string(s: str) -> dict:
    """Parse ``'SCHD=0.24,VTV=0.20,…'`` → ``{ticker: weight}`` (fractions)."""
    raw: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--portfolio: expected TICKER=WEIGHT, got '{part}'")
        t, w = part.split("=", 1)
        try:
            raw[t.strip()] = float(w.strip())
        except ValueError:
            raise ValueError(f"--portfolio: weight for {t.strip()} must be a number, "
                             f"got '{w.strip()}'")
    if not raw:
        raise ValueError("--portfolio is empty")
    return raw


def load_portfolio_file(path: str) -> tuple[dict, dict]:
    """
    Load a portfolio JSON.  Two accepted shapes:
      • bare map:           ``{"SCHD": 0.24, "VTV": 0.20, …}``
      • self-contained scenario:
            ``{"allocations": {…}, "max_loss": 0.05, "horizon_years": 5,
               "capital": 100000}``

    Returns ``(raw_allocations, file_assumptions)``.  Assumptions present in the
    file are used unless a CLI flag overrides them (flag > file > default).
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    assumptions: dict = {}
    if "allocations" in data:
        raw = data.get("allocations")
        for k in ("max_loss", "horizon_years", "capital"):
            if data.get(k) is not None:
                assumptions[k] = data[k]
    else:
        raw = data
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: no allocations found")
    return raw, assumptions


# ===========================================================================
# 2. apply_edits
# ===========================================================================
def _parse_edits(raw_edits: list[tuple[str, str]]) -> list[tuple]:
    """Normalize CLI edit strings into typed ops, preserving order."""
    ops: list[tuple] = []
    for kind, raw in raw_edits:
        if kind == "substitute":
            if "=" not in raw:
                raise ValueError(f"--substitute expects A=B, got '{raw}'")
            a, b = raw.split("=", 1)
            ops.append(("substitute", a.strip().upper(), b.strip().upper()))
        elif kind == "set":
            if "=" not in raw:
                raise ValueError(f"--set expects TICKER=WEIGHT, got '{raw}'")
            x, w = raw.split("=", 1)
            try:
                wf = float(w)
            except ValueError:
                raise ValueError(f"--set weight must be a number, got '{w}'")
            if not (0 <= wf < 1):
                raise ValueError(f"--set weight must be in [0, 1), got {wf}")
            ops.append(("set", x.strip().upper(), wf))
        elif kind == "remove":
            ops.append(("remove", raw.strip().upper()))
        else:
            raise ValueError(f"unknown edit kind: {kind}")
    return ops


def apply_edits(baseline_alloc: dict, ops: list[tuple]) -> tuple[dict, list[str]]:
    """
    Apply substitute / set / remove ops in order, then renormalize ONCE so the
    book sums to 1.0.  Returns ``(new_allocations, notes)``; renormalization is
    NEVER silent — it always lands in ``notes``.

    Semantics (see SESSION_NOTES → What-If, decision 2):
      • substitute A=B  — move A's weight to B (sum preserved); add to B if held;
                          error if A absent.
      • set X=w         — X is PINNED at w; the others renormalize to fill (1-w).
      • remove A        — drop A; the rest renormalize.
    """
    alloc = {t: float(w) for t, w in baseline_alloc.items() if float(w) > _WEIGHT_EPS}
    pinned: set[str] = set()
    notes: list[str] = []

    for op in ops:
        if op[0] == "substitute":
            _, a, b = op
            if a not in alloc:
                raise ValueError(f"--substitute {a}={b}: '{a}' is not in the portfolio")
            w = alloc.pop(a)
            pinned.discard(a)
            if b in alloc:
                alloc[b] += w
                notes.append(f"{a} ({w:.1%}) reassigned to existing {b} → {alloc[b]:.1%}")
            else:
                alloc[b] = w
                notes.append(f"substituted {a} ({w:.1%}) → {b} ({w:.1%})")
        elif op[0] == "set":
            _, x, w = op
            prev = alloc.get(x)
            alloc[x] = w
            pinned.add(x)
            notes.append(f"set {x} = {w:.1%}" + (f" (was {prev:.1%})" if prev else " (new)"))
        elif op[0] == "remove":
            _, a = op
            if a not in alloc:
                raise ValueError(f"--remove {a}: '{a}' is not in the portfolio")
            w = alloc.pop(a)
            pinned.discard(a)
            notes.append(f"removed {a} ({w:.1%})")

    # drop set-to-zero / emptied legs
    for t in [t for t, w in list(alloc.items()) if w <= _WEIGHT_EPS]:
        del alloc[t]
        pinned.discard(t)
    if not alloc:
        raise ValueError("edits left an empty portfolio")

    raw_sum = sum(alloc.values())
    if pinned:
        pin_sum = sum(alloc[t] for t in pinned)
        unpinned = [t for t in alloc if t not in pinned]
        un_sum = sum(alloc[t] for t in unpinned)
        if (1.0 - pin_sum) > _WEIGHT_EPS and un_sum > _WEIGHT_EPS:
            scale = (1.0 - pin_sum) / un_sum
            for t in unpinned:
                alloc[t] *= scale
            notes.append(
                f"renormalized non-pinned legs to fill {1.0 - pin_sum:.1%} "
                f"(pinned {', '.join(sorted(pinned))} = {pin_sum:.1%})"
            )
        else:
            s = sum(alloc.values())
            for t in alloc:
                alloc[t] /= s
            notes.append(
                f"pinned weights summed to {pin_sum:.0%}; renormalized everything "
                f"proportionally"
            )
    elif abs(raw_sum - 1.0) > 1e-6:
        for t in alloc:
            alloc[t] /= raw_sum
        notes.append(f"renormalized proportionally (raw sum was {raw_sum:.2f})")

    return {t: round(w, 6) for t, w in alloc.items()}, notes


# ===========================================================================
# 3. evaluate — recompute the deterministic blocks for ONE book
# ===========================================================================
def _trailing_backtest(alloc: dict, years: int) -> dict | None:
    """compute_backtest over the trailing ``years`` (the one thing loss_floor /
    risk_profile don't already cover).  Fail-soft -> None."""
    today = _dt.date.today()
    start = (today - _dt.timedelta(days=365 * years)).isoformat()
    try:
        r = compute_backtest({
            "weights": alloc,
            "start_date": start,
            "end_date": today.isoformat(),
        })
    except Exception as exc:
        return {"ok": False, "years": years, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "years": years,
        "start": start,
        "total_return": r.get("total_return"),
        "max_drawdown": r.get("max_drawdown"),
        "annualised_vol": r.get("annualised_volatility"),
        "coverage_weight": r.get("coverage_weight"),
    }


def _reuse_or(precomputed: dict | None, key: str, compute):
    """Reuse a baseline block if it was actually performed; else recompute."""
    if precomputed:
        b = precomputed.get(key)
        if isinstance(b, dict) and b.get("performed"):
            return b
    return compute()


def evaluate(
    alloc: dict,
    *,
    descriptions: dict | None,
    target_max_loss: float,
    horizon_years: int | None,
    capital: float,
    qtypes: dict[str, str],
    trailing_years: int = 5,
    precomputed: dict | None = None,
) -> dict:
    """
    Recompute the deterministic eval blocks for ``alloc``.  When ``precomputed``
    (a baseline result dict) is supplied, its already-computed, seeded blocks
    are reused for the "before" side; the modified book passes ``precomputed=None``
    so everything is computed fresh.  Returns a dict with the harness-shaped
    blocks plus the trailing backtest and concentration.
    """
    loss_floor_block = _reuse_or(
        precomputed, "loss_floor",
        lambda: run_loss_floor_check(alloc, max_loss=target_max_loss,
                                     descriptions=descriptions),
    )
    risk_block = _reuse_or(
        precomputed, "risk_profile",
        lambda: {"performed": True, "skipped_reason": None,
                 **asdict(run_risk_profile(alloc, horizon_years=horizon_years))},
    )
    correlation_block = _reuse_or(
        precomputed, "correlation",
        lambda: {"performed": True, "skipped_reason": None,
                 **asdict(run_correlation(alloc))},
    )
    pricing_block = _reuse_or(
        precomputed, "pricing",
        lambda: {"performed": True, "skipped_reason": None,
                 **asdict(run_pricing(alloc, capital))},
    )
    return {
        "allocations": alloc,
        "blocks": {
            "loss_floor": loss_floor_block,
            "risk_profile": risk_block,
            "correlation": correlation_block,
            "pricing": pricing_block,
        },
        "trailing": _trailing_backtest(alloc, trailing_years),
        "concentration": _concentration(alloc, qtypes),
    }


# ===========================================================================
# 4. compare — before/after deltas + the headline loss-cap verdict
# ===========================================================================
def _horizon_row(risk_block: dict, horizon_years: int | None) -> dict | None:
    rows = risk_block.get("horizons") or []
    if not rows:
        return None
    if horizon_years is not None:
        for r in rows:
            if r.get("horizon_years") == horizon_years:
                return r
    return rows[-1]  # longest horizon as the default lens


def _high_pairs(corr_block: dict) -> set[frozenset]:
    return {
        frozenset((p.get("a"), p.get("b")))
        for p in (corr_block.get("pairs") or []) if p.get("high")
    }


def compare(baseline_eval: dict, modified_eval: dict, *,
            target_max_loss: float, horizon_years: int | None) -> dict:
    b, m = baseline_eval, modified_eval
    blf, mlf = b["blocks"]["loss_floor"], m["blocks"]["loss_floor"]

    def _within(lf: dict) -> str:
        if not lf.get("performed"):
            return "n/a"
        return "within cap" if lf.get("organic_pass") else "BREACHES cap"

    wb, wm = blf.get("worst_gross_annual_loss"), mlf.get("worst_gross_annual_loss")
    loss_cap_verdict = (
        f"Worst stress-year loss "
        f"{('%+.1f%%' % (wb * 100)) if wb is not None else 'n/a'} "
        f"({blf.get('worst_year')}) → "
        f"{('%+.1f%%' % (wm * 100)) if wm is not None else 'n/a'} "
        f"({mlf.get('worst_year')}); ≤{target_max_loss:.0%} cap: "
        f"{_within(blf)} → {_within(mlf)}"
    )

    br = _horizon_row(b["blocks"]["risk_profile"], horizon_years)
    mr = _horizon_row(m["blocks"]["risk_profile"], horizon_years)

    new_high = _high_pairs(m["blocks"]["correlation"]) - _high_pairs(b["blocks"]["correlation"])
    bc, mc = b["concentration"], m["concentration"]
    new_singles = sorted(set(mc["single_names"]) - set(bc["single_names"]))

    return {
        "loss_cap_verdict": loss_cap_verdict,
        "loss_cap": {
            "cap": target_max_loss,
            "baseline_worst": wb, "baseline_worst_year": blf.get("worst_year"),
            "modified_worst": wm, "modified_worst_year": mlf.get("worst_year"),
            "baseline_pass": blf.get("organic_pass"),
            "modified_pass": mlf.get("organic_pass"),
            "flipped_to_breach": bool(blf.get("organic_pass") and mlf.get("organic_pass") is False),
        },
        "risk_horizon": (None if not (br and mr) else {
            "horizon_years": mr.get("horizon_years"),
            "median": (br.get("median"), mr.get("median")),
            "prob_end_down": (br.get("prob_end_down"), mr.get("prob_end_down")),
            "bad_5th": (br.get("bad_5th"), mr.get("bad_5th")),
        }),
        "trailing": {"baseline": b["trailing"], "modified": m["trailing"]},
        "concentration": {
            "effective_n": (bc["effective_n"], mc["effective_n"]),
            "max_name": ((bc["max_name"], bc["max_name_weight"]),
                         (mc["max_name"], mc["max_name_weight"])),
            "single_name_weight": (bc["single_name_weight"], mc["single_name_weight"]),
            "new_single_names": new_singles,
        },
        "new_high_corr_pairs": [sorted(p) for p in new_high],
    }


# ===========================================================================
# 5. render — whatif_<label>.json (harness-schema-compatible) + .md
# ===========================================================================
def _fmt_pct(x, signed=True):
    if x is None or isinstance(x, str):
        return "n/a"
    return (f"{x:+.1%}" if signed else f"{x:.1%}")


def _delta(before, after):
    """after − before, None-safe (any missing/non-numeric side → None)."""
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def _schema_result(eval_result: dict, *, baseline: dict | None, notes: list[str],
                   target_max_loss: float, horizon_years: int | None,
                   methodology: str, rationale: str) -> dict:
    """
    Build a portfolio as a STANDARD harness-schema result, so
    ``viz/ui_designer.py`` renders it like any run with NO designer changes.
    Shared by both modes:
      • EDIT mode   → ``baseline`` is the source run (model / spec / posture copied).
      • DIRECT mode → ``baseline`` is None (model / spec / posture null; the
                       designer hides the spec / QA / iteration panels, since none
                       of those exist for a user-supplied book).

    ASSUMPTIONS (documented): the LLM-only fields (``final_evaluation``,
    ``iteration_history``, ``expected_annual_return`` / ``_max_drawdown``) are
    null/empty because no LLM ran — the designer is already told to hide
    missing/null sections gracefully.  ``mode="optimized"`` is only a rendering
    hint; provenance lives within-schema in ``final_proposal.methodology`` /
    ``rationale``.  The comparison (edit mode) lives in the CLI + .md, NOT here.
    """
    b = baseline or {}
    alloc = eval_result["allocations"]
    base_fp = b.get("final_proposal") or {}
    return {
        "model": b.get("model"),
        "max_iterations": b.get("max_iterations"),
        "pass_threshold": b.get("pass_threshold"),
        "target_max_loss": target_max_loss,
        "mode": "optimized",
        "horizon_years": horizon_years,
        "horizon_posture": b.get("horizon_posture"),
        "spec": b.get("spec"),
        "final_proposal": {
            "allocations": alloc,
            "descriptions": {t: (base_fp.get("descriptions") or {}).get(t, "")
                             for t in alloc},
            "expected_annual_return": None,   # LLM assertion — absent (no LLM ran)
            "expected_max_drawdown": None,
            "methodology": methodology,
            "rationale": rationale,
            "raw_text": "",
        },
        "final_evaluation": None,        # null → designer hides the QA section
        "selected_iteration": None,
        "selected_proposal": None,
        "selected_evaluation": None,
        "iteration_history": [],
        "refinement": {"performed": False,
                       "skipped_reason": "deterministic what-if/analysis — no optimisation loop",
                       "promoted": False, "refined_proposal": None,
                       "refined_evaluation": None, "improvements": None},
        "pricing": eval_result["blocks"]["pricing"],
        "risk_profile": eval_result["blocks"]["risk_profile"],
        "correlation": eval_result["blocks"]["correlation"],
        "loss_floor": eval_result["blocks"]["loss_floor"],
    }


def _comparison_md(push, comparison: dict, baseline_eval: dict, modified_eval: dict):
    cmp = comparison
    push("## What-If Comparison (baseline → modified)")
    push("")
    push(f"> **Loss-cap verdict:** {cmp['loss_cap_verdict']}"
         + ("  ⚠️ **FLIPPED TO BREACH**" if cmp["loss_cap"]["flipped_to_breach"] else ""))
    push("")
    push("| Metric | Baseline | Modified | Δ |")
    push("|---|---|---|---|")

    lc = cmp["loss_cap"]
    push(f"| Worst stress-year loss (gross) | {_fmt_pct(lc['baseline_worst'])} "
         f"({lc['baseline_worst_year']}) | {_fmt_pct(lc['modified_worst'])} "
         f"({lc['modified_worst_year']}) | "
         f"{_fmt_pct(_delta(lc['baseline_worst'], lc['modified_worst']))} |")

    tb, tm = cmp["trailing"]["baseline"], cmp["trailing"]["modified"]
    if tb and tm and tb.get("ok") and tm.get("ok"):
        push(f"| Trailing {tm['years']}y return | {_fmt_pct(tb['total_return'])} | "
             f"{_fmt_pct(tm['total_return'])} | "
             f"{_fmt_pct(_delta(tb['total_return'], tm['total_return']))} |")
        push(f"| Trailing {tm['years']}y volatility (ann.) | {_fmt_pct(tb['annualised_vol'], 0)} | "
             f"{_fmt_pct(tm['annualised_vol'], 0)} | "
             f"{_fmt_pct(_delta(tb['annualised_vol'], tm['annualised_vol']))} |")
        push(f"| Trailing {tm['years']}y max drawdown | {_fmt_pct(tb['max_drawdown'])} | "
             f"{_fmt_pct(tm['max_drawdown'])} | "
             f"{_fmt_pct(_delta(tb['max_drawdown'], tm['max_drawdown']))} |")

    rh = cmp["risk_horizon"]
    if rh:
        bm, mm = rh["median"]
        bp, mp = rh["prob_end_down"]
        push(f"| Monte-Carlo {rh['horizon_years']}y median return | {_fmt_pct(bm)} | "
             f"{_fmt_pct(mm)} | {_fmt_pct(_delta(bm, mm))} |")
        push(f"| {rh['horizon_years']}y P(ending down) | {_fmt_pct(bp, 0)} | {_fmt_pct(mp, 0)} | "
             f"{_fmt_pct(_delta(bp, mp))} |")

    con = cmp["concentration"]
    bn, mn = con["effective_n"]
    push(f"| Effective-N (1/HHI, breadth) | {bn} | {mn} | {round(mn - bn, 2):+} |")
    bsw, msw = con["single_name_weight"]
    push(f"| **Single-name equity weight** | {_fmt_pct(bsw, 0)} | {_fmt_pct(msw, 0)} | "
         f"{_fmt_pct(msw - bsw)} |")
    push("")

    if con["new_single_names"]:
        push(f"> ⚠️ **Introduces single-stock exposure:** "
             f"{', '.join('`%s`' % t for t in con['new_single_names'])} "
             f"(individual companies, not funds — effective-N alone misses this; "
             f"the volatility / loss-cap rows above carry the real risk).")
        push("")
    if cmp["new_high_corr_pairs"]:
        pairs = "; ".join(f"`{a}`↔`{b}`" for a, b in cmp["new_high_corr_pairs"])
        push(f"> ⚠️ **New |ρ| ≥ 0.85 pairs introduced:** {pairs}")
        push("")


def render(baseline: dict, modified_eval: dict, comparison: dict, *,
           edits_repr: list, notes: list[str],
           target_max_loss: float, horizon_years: int | None,
           label: str, out_dir: str = ".") -> tuple[str, str]:
    """Write whatif_<label>.{json,md}; return their paths.  Baseline untouched.

    The .json is a standard-schema modified portfolio (for the designer); the
    .md carries the before/after comparison."""
    result = _schema_result(
        modified_eval, baseline=baseline, notes=notes,
        target_max_loss=target_max_loss, horizon_years=horizon_years,
        methodology="What-if edit of a completed run (deterministic; no LLM).",
        rationale="Edits: " + ("; ".join(notes) if notes else "none"),
    )

    json_path = os.path.join(out_dir, f"whatif_{label}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    lines: list[str] = []
    push = lines.append
    push(f"# What-If: {label}")
    push("")
    push("**Edits applied** (deterministic; no LLM, recomputed for both books):")
    for e in edits_repr:
        push(f"- `{e}`")
    if notes:
        push("")
        push("**Book changes:** " + "; ".join(notes))
    push("")
    _comparison_md(push, comparison, None, modified_eval)
    push("---")
    push("")
    push("### Modified book — full deterministic detail")
    push("")
    _push_loss_floor_section(push, result)
    _push_risk_section(push, result)
    _push_correlation_section(push, result)
    _push_pricing_section(push, result)

    md_path = os.path.join(out_dir, f"whatif_{label}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, md_path


def _render_standalone(eval_result: dict, *, notes: list[str], target_max_loss: float,
                       horizon_years: int | None, capital: float, label: str,
                       out_dir: str = ".") -> tuple[str, str]:
    """Write whatif_<label>.{json,md} for a STANDALONE portfolio (direct mode,
    no comparison).  The .json is standard harness schema (for the designer);
    the .md is a single-portfolio risk report."""
    alloc = eval_result["allocations"]
    result = _schema_result(
        eval_result, baseline=None, notes=notes, target_max_loss=target_max_loss,
        horizon_years=horizon_years,
        methodology="User-specified portfolio (deterministic risk analysis; no LLM).",
        rationale=f"Standalone analysis of a {len(alloc)}-holding portfolio.",
    )
    json_path = os.path.join(out_dir, f"whatif_{label}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    lines: list[str] = []
    push = lines.append
    push(f"# Portfolio Risk Analysis: {label}")
    push("")
    push(f"User-specified portfolio — deterministic analysis, no LLM. Assumptions: "
         f"loss cap ≤{target_max_loss:.0%}, horizon {horizon_years}y, "
         f"capital ${capital:,.0f}.")
    if notes:
        push("")
        push("**Note:** " + "; ".join(notes))
    push("")
    push("**Holdings:**")
    for t, w in sorted(alloc.items(), key=lambda kv: -kv[1]):
        push(f"- `{t}` — {w:.1%}")
    push("")

    lf = eval_result["blocks"]["loss_floor"]
    if lf.get("performed") and lf.get("worst_gross_annual_loss") is not None:
        within = "within" if lf.get("organic_pass") else "**BREACHES**"
        push(f"> **Loss cap:** worst stress-year loss "
             f"{_fmt_pct(lf['worst_gross_annual_loss'])} ({lf.get('worst_year')}) — "
             f"{within} the ≤{target_max_loss:.0%} cap.")
        push("")

    con = eval_result["concentration"]
    singles = f" ({', '.join(con['single_names'])})" if con["single_names"] else ""
    push(f"**Concentration:** effective-N {con['effective_n']}; largest holding "
         f"`{con['max_name']}` {_fmt_pct(con['max_name_weight'], 0)}; single-name equity "
         f"{_fmt_pct(con['single_name_weight'], 0)}{singles}.")
    tb = eval_result["trailing"]
    if tb and tb.get("ok"):
        push("")
        push(f"**Trailing {tb['years']}y:** return {_fmt_pct(tb['total_return'])}, "
             f"vol {_fmt_pct(tb['annualised_vol'], 0)}, max drawdown {_fmt_pct(tb['max_drawdown'])}.")
    push("")
    push("---")
    push("")
    push("### Full deterministic detail")
    push("")
    _push_loss_floor_section(push, result)
    _push_risk_section(push, result)
    _push_correlation_section(push, result)
    _push_pricing_section(push, result)

    md_path = os.path.join(out_dir, f"whatif_{label}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, md_path


# ===========================================================================
# CLI
# ===========================================================================
class _EditAction(argparse.Action):
    """Append (kind, raw) to a shared ``edits`` list, preserving CLI order."""
    def __call__(self, parser, namespace, values, option_string=None):
        edits = getattr(namespace, "edits", None) or []
        edits.append((option_string.lstrip("-"), values))
        namespace.edits = edits


def _auto_label(ops: list[tuple]) -> str:
    if not ops:
        return "edit"
    op = ops[0]
    if op[0] == "substitute":
        base = f"{op[1]}-{op[2]}"
    elif op[0] == "set":
        base = f"set-{op[1]}"
    else:
        base = f"rm-{op[1]}"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in base)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="whatif.py",
        description="Deterministic portfolio risk analysis — no LLM, no API key. "
                    "EDIT a completed harness_output.json "
                    "(--substitute/--set/--remove → before/after comparison), OR "
                    "ANALYZE a portfolio you supply (--portfolio / --portfolio-file → "
                    "standalone risk report).",
    )
    p.add_argument("baseline", nargs="?", default=None,
                   help="path to a completed harness_output.json (EDIT mode; "
                        "omit when using --portfolio / --portfolio-file)")
    # --- direct (standalone) portfolio input ---
    p.add_argument("--portfolio", metavar='"T=w,T=w,…"',
                   help='analyze a portfolio you specify inline, fractions, e.g. '
                        '"SCHD=0.24,VTV=0.20,SGOV=0.56"')
    p.add_argument("--portfolio-file", metavar="PATH",
                   help="analyze a portfolio from a JSON file: a bare {ticker: weight} "
                        "map, OR {\"allocations\": {…}, \"max_loss\":…, "
                        "\"horizon_years\":…, \"capital\":…}")
    # --- edits (EDIT mode) ---
    p.add_argument("--substitute", dest="edits", action=_EditAction, metavar="A=B",
                   help="move A's weight to B (e.g. SPY=NVDA). Repeatable.")
    p.add_argument("--set", dest="edits", action=_EditAction, metavar="TICKER=WEIGHT",
                   help="pin TICKER at WEIGHT (fraction, e.g. GLD=0.10); others fill the rest.")
    p.add_argument("--remove", dest="edits", action=_EditAction, metavar="TICKER",
                   help="drop TICKER; the rest renormalize. Repeatable.")
    # --- assumptions: explicit flag > baseline/file > default (0.05 / 5y / 100K) ---
    p.add_argument("--max-loss", type=float, default=None, metavar="FRACTION",
                   help="annual loss cap (direct-mode default 0.05; else from baseline/file)")
    p.add_argument("--horizon-years", type=int, default=None,
                   help="investment horizon, years (direct-mode default 5; else from baseline/file)")
    p.add_argument("--capital", type=float, default=None,
                   help="capital for the lot-size check (direct-mode default 100000; else from baseline/file)")
    # --- output ---
    p.add_argument("-o", "--label", default=None,
                   help="output label → whatif_<label>.{json,md} (auto if omitted)")
    p.add_argument("--trailing-years", type=int, default=5,
                   help="trailing backtest window in years (default 5)")
    p.add_argument("--out-dir", default=".", help="output directory (default cwd)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    direct = bool(args.portfolio or args.portfolio_file)
    edits = getattr(args, "edits", None) or []

    if direct:
        if args.baseline or edits:
            print("ERROR: --portfolio / --portfolio-file is STANDALONE — don't combine it "
                  "with a harness_output.json baseline or --substitute/--set/--remove.",
                  file=sys.stderr)
            return 2
        if args.portfolio and args.portfolio_file:
            print("ERROR: give either --portfolio or --portfolio-file, not both.",
                  file=sys.stderr)
            return 2
        return _run_direct(args)

    # --- EDIT mode ---
    if not args.baseline:
        print("ERROR: give a completed harness_output.json (+ edits), or use "
              "--portfolio / --portfolio-file for a standalone book.", file=sys.stderr)
        return 2
    if not edits:
        print("ERROR: no edits given (use --substitute / --set / --remove), or use "
              "--portfolio / --portfolio-file for a standalone analysis.", file=sys.stderr)
        return 2
    return _run_edit(args, edits)


def _run_edit(args, raw_edits: list) -> int:
    """EDIT mode: harness_output.json + edits → before/after comparison."""
    try:
        ops = _parse_edits(raw_edits)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    baseline = load_baseline(args.baseline)
    baseline["_source_path"] = os.path.abspath(args.baseline)
    base_fp = baseline.get("final_proposal") or {}
    baseline_alloc = {t: float(w) for t, w in (base_fp.get("allocations") or {}).items()}
    if not baseline_alloc:
        print("ERROR: baseline has no final_proposal.allocations to edit.", file=sys.stderr)
        return 1

    # Assumptions: explicit flag > baseline value > default.
    target_max_loss = (args.max_loss if args.max_loss is not None
                       else float(baseline.get("target_max_loss") or 0.05))
    horizon_years = (args.horizon_years if args.horizon_years is not None
                     else baseline.get("horizon_years"))
    capital = (args.capital if args.capital is not None
               else float((baseline.get("pricing") or {}).get("capital") or DEFAULT_CAPITAL))
    descriptions = base_fp.get("descriptions") or {}

    try:
        modified_alloc, notes = apply_edits(baseline_alloc, ops)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    label = args.label or _auto_label(ops)
    edits_repr = [(" ".join(str(x) for x in op)) for op in ops]
    print(f"\nWHAT-IF: {label}")
    print("Edits:", "; ".join(edits_repr))
    for n in notes:
        print(f"  · {n}")

    qtypes = _security_types(set(baseline_alloc) | set(modified_alloc))
    print("Evaluating baseline (reusing computed blocks where present) …")
    base_eval = evaluate(baseline_alloc, descriptions=descriptions,
                         target_max_loss=target_max_loss, horizon_years=horizon_years,
                         capital=capital, qtypes=qtypes,
                         trailing_years=args.trailing_years, precomputed=baseline)
    print("Evaluating modified book …")
    mod_eval = evaluate(modified_alloc, descriptions=descriptions,
                        target_max_loss=target_max_loss, horizon_years=horizon_years,
                        capital=capital, qtypes=qtypes,
                        trailing_years=args.trailing_years, precomputed=None)

    comparison = compare(base_eval, mod_eval,
                         target_max_loss=target_max_loss, horizon_years=horizon_years)
    json_path, md_path = render(baseline, mod_eval, comparison,
                                edits_repr=edits_repr, notes=notes,
                                target_max_loss=target_max_loss,
                                horizon_years=horizon_years, label=label,
                                out_dir=args.out_dir)

    print(f"\n{comparison['loss_cap_verdict']}")
    if comparison["loss_cap"]["flipped_to_breach"]:
        print("⚠️  This edit FLIPS the loss cap to BREACHED.")
    print(f"\nWrote {json_path}\n      {md_path}")
    return 0


def _run_direct(args) -> int:
    """DIRECT mode: a user-supplied portfolio → standalone risk analysis."""
    try:
        if args.portfolio_file:
            raw, file_assump = load_portfolio_file(args.portfolio_file)
        else:
            raw, file_assump = parse_portfolio_string(args.portfolio), {}
        alloc, notes = _normalize_weights(raw)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Assumptions: explicit flag > file value > default (5% / 5y / 100K).
    max_loss = (args.max_loss if args.max_loss is not None
                else float(file_assump.get("max_loss", 0.05)))
    horizon = (args.horizon_years if args.horizon_years is not None
               else int(file_assump.get("horizon_years", 5)))
    capital = (args.capital if args.capital is not None
               else float(file_assump.get("capital", 100000.0)))

    label = args.label or "portfolio"
    print(f"\nPORTFOLIO ANALYSIS: {label}")
    print("Holdings:", "; ".join(f"{t} {w:.1%}"
                                 for t, w in sorted(alloc.items(), key=lambda kv: -kv[1])))
    print(f"Assumptions: cap ≤{max_loss:.0%}, horizon {horizon}y, capital ${capital:,.0f}")
    for n in notes:
        print(f"  · {n}")

    qtypes = _security_types(set(alloc))
    print("Evaluating portfolio …")
    eval_ = evaluate(alloc, descriptions={}, target_max_loss=max_loss,
                     horizon_years=horizon, capital=capital, qtypes=qtypes,
                     trailing_years=args.trailing_years, precomputed=None)

    json_path, md_path = _render_standalone(eval_, notes=notes, target_max_loss=max_loss,
                                            horizon_years=horizon, capital=capital,
                                            label=label, out_dir=args.out_dir)

    lf = eval_["blocks"]["loss_floor"]
    if lf.get("performed") and lf.get("worst_gross_annual_loss") is not None:
        verdict = "within" if lf.get("organic_pass") else "BREACHES"
        print(f"\nLoss cap ≤{max_loss:.0%}: worst stress-year loss "
              f"{lf['worst_gross_annual_loss']:+.1%} ({lf.get('worst_year')}) — {verdict}.")
    print(f"\nWrote {json_path}\n      {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
