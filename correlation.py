"""
Pairwise-correlation snapshot for the Portfolio Optimization Harness.

This is the third self-contained, no-LLM post-processing sibling alongside
``pricing.py`` and ``risk.py``: it takes ``allocations`` directly, never
raises, and degrades gracefully (any failure populates the ``error`` field
so the report can render a helpful note instead of a stack trace).

It replaces the old Advisor agent, which *recalled* correlations from model
memory — unreliably.  Empirically the Advisor systematically OVERSTATED
correlations among the cash / short-duration sleeve (e.g. it reported
SHY↔SGOV ≈ 0.9 when the real daily figure is ≈ 0.1, because near-zero-
duration T-bills barely co-move with anything) and UNDERSTATED the genuine
intermediate-duration bond cluster (AGG↔IEI↔SCHP ≈ 0.85-0.95).  Correlation
is a computation, not a judgement, so we compute it from real history.

Method (deliberately simple and honest):

  1. Over a trailing window (default 5y), download daily total-return
     history for the holdings via yfinance, using the REAL tickers (no
     asset-class proxying — unlike ``risk.py``, which proxies young ETFs to
     reach the 2008 tail; here we want each holding's own behaviour).
  2. Drop holdings with no priceable history (model-invented option legs
     like "SPY_PUT_SPREAD") — they have no sensible return correlation.
  3. Compute the pairwise Pearson correlation of daily returns (pandas does
     pairwise-complete deletion; pairs with under ``CORR_MIN_PERIODS``
     overlapping observations are skipped as unreliable).
  4. Surface every pair with |ρ| >= ``CORR_REPORT_THRESHOLD`` (0.5), flagging
     the subset at |ρ| >= ``CORR_HIGH_THRESHOLD`` (0.85) as highly redundant.

Honesty notes (surfaced in ``CORRELATION_DISCLAIMER`` and the result): these
are DAILY-return correlations over the stated window; correlation is regime-
and frequency-dependent (it generally rises in stress and differs at
monthly/annual frequencies), so it is one lens, not a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Mapping

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
CORR_WINDOW_YEARS = 5          # trailing window for the daily-return sample
CORR_MIN_PERIODS = 120         # ~6 months of overlap required for a pair's ρ
CORR_REPORT_THRESHOLD = 0.5    # surface pairs at |ρ| >= this (matches old advisor)
CORR_HIGH_THRESHOLD = 0.85     # flag pairs at |ρ| >= this as highly redundant

CORRELATION_DISCLAIMER = (
    "Pairwise Pearson correlations of DAILY total returns over the trailing "
    "window shown below, computed from Yahoo Finance history (not recalled "
    "from a model). Correlation is regime- and frequency-dependent: it "
    "generally RISES in market stress and differs at monthly / annual "
    "frequencies, so treat these as one lens rather than a constant. "
    "Holdings with no priceable history (e.g., option overlays) are omitted, "
    "and pairs without enough overlapping history are skipped. Pairs at "
    "|ρ| >= 0.85 are flagged as highly redundant. This is NOT investment "
    "advice."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CorrelationPair:
    """One pairwise correlation entry (|ρ| >= CORR_REPORT_THRESHOLD)."""
    a: str
    b: str
    rho: float
    high: bool          # |ρ| >= CORR_HIGH_THRESHOLD — highly redundant


@dataclass
class CorrelationResult:
    """Aggregated pairwise-correlation snapshot."""
    window_years: int = 0
    sample_start: str = ""
    sample_end: str = ""
    sample_days: int = 0
    frequency: str = "daily"
    report_threshold: float = CORR_REPORT_THRESHOLD
    high_threshold: float = CORR_HIGH_THRESHOLD
    coverage_weight: float = 0.0          # share of the book actually modeled
    modeled_tickers: list[str] = field(default_factory=list)
    dropped_tickers: list[str] = field(default_factory=list)   # non-priceable
    pairs: list[CorrelationPair] = field(default_factory=list)  # sorted |ρ| desc
    high_pairs_count: int = 0
    disclaimer: str = ""
    source: str = "yfinance"
    computed_at: str = ""
    error: str | None = None              # set when the snapshot could not run


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_correlation(
    allocations: Mapping[str, float],
    *,
    window_years: int = CORR_WINDOW_YEARS,
) -> CorrelationResult:
    """
    Compute a pairwise daily-return correlation snapshot for ``allocations``
    (a mapping of ticker -> weight; pass ``final_proposal.allocations``
    directly).

    Never raises — on any failure the returned result's ``error`` field is
    populated so the report can render a helpful note instead of a stack
    trace.
    """
    print("\n" + "=" * 60)
    print("CORRELATION — computing pairwise return correlations (yfinance) …")
    print("=" * 60)

    result = CorrelationResult(
        window_years=window_years,
        disclaimer=CORRELATION_DISCLAIMER,
        computed_at=datetime.now().isoformat(timespec="seconds"),
    )

    # --- Lazy imports so a missing scientific stack degrades gracefully ---
    try:
        import logging
        import pandas as pd
        import yfinance as yf
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    except ImportError as exc:
        result.error = (
            f"required library missing ({exc}). Run `uv sync` from the "
            f"testbench root to install pandas / yfinance."
        )
        print(f"  ⚠️  {result.error}")
        return result

    # --- Collect positive-weight holdings ---
    weights: dict[str, float] = {}
    for ticker, weight in allocations.items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if w > 0:
            weights[ticker] = weights.get(ticker, 0.0) + w
    total_weight = sum(weights.values())
    if not weights:
        result.error = "no positive-weight holdings to correlate"
        print(f"  ⚠️  {result.error}")
        return result

    tickers = list(weights.keys())

    # --- Download daily history over the trailing window ---
    start = (
        pd.Timestamp.today().normalize() - pd.DateOffset(years=window_years)
    ).strftime("%Y-%m-%d")
    try:
        px = yf.download(
            tickers,
            start=start,
            end=None,
            auto_adjust=True,
            progress=False,
        )["Close"]
    except Exception as exc:
        result.error = f"yfinance download failed ({type(exc).__name__}: {exc})"
        print(f"  ⚠️  {result.error}")
        return result

    # yfinance returns a Series for a single ticker — normalize to DataFrame.
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    # --- Drop holdings with too little priceable history (synthetic legs,
    #     unknown tickers, brand-new funds) ---
    present = [
        t for t in tickers
        if t in px.columns and int(px[t].notna().sum()) >= CORR_MIN_PERIODS
    ]
    result.dropped_tickers = [t for t in tickers if t not in present]
    for t in result.dropped_tickers:
        print(f"  ✗ dropped (no / too little priceable history): {t}")

    kept_weight = sum(weights[t] for t in present)
    result.coverage_weight = round(
        kept_weight / total_weight if total_weight > 0 else 0.0, 6
    )
    result.modeled_tickers = present

    if not present:
        result.error = "none of the holdings could be priced via yfinance"
        print(f"  ⚠️  {result.error}")
        return result

    # --- Daily returns + sample window bookkeeping ---
    rets = px[present].pct_change().dropna(how="all")
    if rets.empty:
        result.error = "no overlapping return history for the holdings"
        print(f"  ⚠️  {result.error}")
        return result
    result.sample_start = str(rets.index[0].date())
    result.sample_end = str(rets.index[-1].date())
    result.sample_days = int(len(rets))

    print(
        f"  Sample: {result.sample_start} → {result.sample_end} "
        f"(~{window_years}y daily); coverage {result.coverage_weight:.0%} "
        f"of the book ({len(present)} of {len(tickers)} holdings)"
    )

    # --- Pairwise correlation (pandas does pairwise-complete deletion) ---
    if len(present) < 2:
        print("  (only one priceable holding — no pairs to report)")
        return result

    corr = rets[present].corr(min_periods=CORR_MIN_PERIODS)
    pairs: list[CorrelationPair] = []
    for a, b in combinations(present, 2):
        try:
            rho = float(corr.at[a, b])
        except (KeyError, TypeError, ValueError):
            continue
        if rho != rho:                      # NaN — insufficient overlap
            continue
        if abs(rho) >= CORR_REPORT_THRESHOLD:
            pairs.append(CorrelationPair(
                a=a, b=b,
                rho=round(rho, 4),
                high=abs(rho) >= CORR_HIGH_THRESHOLD,
            ))

    pairs.sort(key=lambda p: -abs(p.rho))
    result.pairs = pairs
    result.high_pairs_count = sum(1 for p in pairs if p.high)

    print(
        f"  {len(pairs)} pair(s) at |ρ| >= {CORR_REPORT_THRESHOLD:.2f}; "
        f"{result.high_pairs_count} at |ρ| >= {CORR_HIGH_THRESHOLD:.2f} (high)"
    )
    for p in pairs:
        flag = "  ⚠ HIGH" if p.high else ""
        print(f"    {p.a:8s} ↔ {p.b:8s}  ρ={p.rho:+.2f}{flag}")

    return result
