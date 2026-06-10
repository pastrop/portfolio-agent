"""
Return-distribution (Monte-Carlo) risk profile for the Portfolio
Optimization Harness.

Where ``pricing.py`` answers "can I actually buy this with my capital?",
this module answers "if I hold this, where might I end up?".  It replaces
the single point estimate ``expected_max_drawdown`` with a distribution:
for several holding horizons it reports the median outcome, the chance of
ending underwater, and the unlucky tails.

Method (deliberately simple and honest):

  1. Map each holding to a long-history asset-class PROXY where the actual
     ETF is too young (so the sample can reach back through the 2008 GFC).
     Holdings that can't be priced at all (model-invented option legs like
     "SPY_PUT_SPREAD") are dropped and the remaining weights renormalized.
  2. Pull daily total-return history for the proxy set from yfinance and
     build a fixed-weight (daily-rebalanced) portfolio return series.
  3. Block-bootstrap that daily series into many multi-year paths (6-month
     blocks preserve volatility clustering and strings of bad quarters,
     rather than assuming each year is independent), and read the terminal
     compounded return off each path.

Like ``pricing.py`` this module is self-contained — no dependency on
``harness.py`` — and degrades gracefully: any failure (missing yfinance,
no priceable tickers, too little history) returns a result whose ``error``
field is populated so the report can render a helpful note instead of a
stack trace.

IMPORTANT honesty notes, surfaced in ``RISK_DISCLAIMER`` and the result:
proxies approximate an asset CLASS, not the exact fund; any non-priceable
hedge sleeve is excluded, so the modeled downside is typically MORE
conservative than the real (hedged) portfolio; and the sample contains
only a handful of true crises, so deep-tail numbers are rough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
TRADING_DAYS = 252
RISK_HISTORY_START = "2006-01-01"   # pull from here; the GFC is the prize
RISK_HORIZONS = (1, 3, 5, 10)       # holding periods, in years
RISK_BLOCK_DAYS = 126               # 6-month bootstrap blocks (regime-preserving)
RISK_N_SIMS = 20_000                # Monte-Carlo paths per horizon
RISK_SEED = 7                       # fixed → reproducible reports

# Best-effort long-history asset-class proxies, so the bootstrap sample can
# reach back through the 2008 crisis even when the actual holding is a young
# ETF.  Only the proxy's ASSET CLASS needs to match — exact tracking is not
# the point; we want a realistic return / volatility / correlation profile
# and a real crisis inside the window.  Tickers NOT listed here are used
# as-is.  Every substitution is reported transparently in the result.
RISK_PROXY_MAP: dict[str, str] = {
    # cash / T-bills
    "SGOV": "SHV", "BIL": "SHV", "GBIL": "SHV", "SHV": "SHV",
    # TIPS / inflation-linked
    "SCHP": "TIP", "VTIP": "TIP", "STIP": "TIP", "LTPZ": "TIP",
    # investment-grade corporate credit
    "VCIT": "LQD", "VCSH": "LQD", "IGIB": "LQD", "IGSB": "LQD",
    # US aggregate bonds
    "BND": "AGG", "SCHZ": "AGG", "IUSB": "AGG",
    # intermediate treasuries
    "GOVT": "IEF", "VGIT": "IEF", "SCHR": "IEF",
    # long treasuries
    "VGLT": "TLT", "EDV": "TLT", "SPTL": "TLT",
    # high-yield credit
    "JNK": "HYG", "USHY": "HYG", "SHYG": "HYG",
    # municipal bonds (Vanguard intermediate tax-exempt fund — history to 1977)
    "MUB": "VWITX", "VTEB": "VWITX", "TFI": "VWITX", "SUB": "VWITX",
    # US dividend tilts
    "DGRO": "VIG", "SCHD": "VIG", "VYM": "VIG", "HDV": "VIG",
    "NOBL": "VIG", "DVY": "VIG",
    # international developed equity
    "VEA": "EFA", "IEFA": "EFA", "SCHF": "EFA", "IDEV": "EFA", "VXUS": "EFA",
    # emerging-market equity
    "IEMG": "EEM", "SCHE": "EEM", "SPEM": "EEM",
    # gold
    "IAU": "GLD", "GLDM": "GLD", "SGOL": "GLD", "BAR": "GLD",
    # broad commodities
    "PDBC": "DBC", "GSG": "DBC", "BCI": "DBC", "COMT": "DBC",
    # US REITs
    "SCHH": "VNQ", "IYR": "VNQ", "XLRE": "VNQ", "RWR": "VNQ",
}

RISK_DISCLAIMER = (
    "Monte-Carlo estimates from a block-bootstrap of historical daily "
    "returns. Long-history asset-class PROXIES are substituted for young "
    "ETFs so the sample can include the 2008 crisis (substitutions listed "
    "below); a proxy approximates asset-class behaviour, not the exact "
    "fund. Any non-priceable sleeve (e.g., options overlays) is EXCLUDED "
    "and weights renormalized, so the modeled downside may be MORE "
    "conservative than the actual hedged portfolio. The sample contains "
    "only a handful of real crises, so deep-tail figures are rough. "
    "Results assume daily rebalancing, no costs / taxes / slippage, and "
    "that the future resembles resampled history. This is NOT a forecast "
    "or investment advice."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class HorizonStats:
    """Terminal (start→finish) outcome distribution for one holding period."""
    horizon_years: int
    median: float           # 50th-percentile total compounded return
    mean: float             # average total compounded return
    prob_end_down: float    # P(total return < 0) — chance of ending underwater
    bad_5th: float          # 5th-percentile total return (1-in-20 unlucky)
    severe_1st: float       # 1st-percentile total return (1-in-100)


@dataclass
class RiskProfileResult:
    """Aggregated return-distribution report."""
    sample_start: str = ""
    sample_end: str = ""
    sample_days: int = 0
    sample_years: float = 0.0
    includes_2008: bool = False
    limiting_ticker: str | None = None    # the holding that constrained the window
    annualized_return: float = 0.0        # realized in the sample (geometric)
    annualized_vol: float = 0.0           # realized in the sample
    coverage_weight: float = 0.0          # share of the book actually modeled
    modeled_weights: dict[str, float] = field(default_factory=dict)  # proxy→weight
    proxy_substitutions: list[dict] = field(default_factory=list)    # {original, proxy}
    dropped_tickers: list[str] = field(default_factory=list)         # non-priceable
    horizons: list[HorizonStats] = field(default_factory=list)
    n_sims: int = 0
    block_days: int = 0
    seed: int = 0
    disclaimer: str = ""
    source: str = "yfinance"
    computed_at: str = ""
    error: str | None = None              # set when the profile could not run


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_proxies(
    allocations: Mapping[str, float],
) -> tuple[dict[str, float], list[dict], float]:
    """
    Map holdings to their proxy tickers and aggregate weights onto each
    proxy.  Returns ``(proxy_weights_raw, substitutions, total_weight)``
    where ``proxy_weights_raw`` is keyed by proxy ticker and NOT yet
    renormalized (caller renormalizes after dropping unpriceable names).
    """
    proxy_weights: dict[str, float] = {}
    substitutions: list[dict] = []
    total = 0.0
    for ticker, weight in allocations.items():
        w = float(weight)
        if w <= 0:
            continue
        proxy = RISK_PROXY_MAP.get(ticker, ticker)
        if proxy != ticker:
            substitutions.append({"original": ticker, "proxy": proxy})
        proxy_weights[proxy] = proxy_weights.get(proxy, 0.0) + w
        total += w
    return proxy_weights, substitutions, total


def bracket_horizons(horizon_years: int | None) -> tuple[int, ...]:
    """
    Choose the Monte-Carlo reporting horizons so the table BRACKETS the run's
    investment horizon — the reader sees outcomes shorter than, at, and longer
    than the target rather than the fixed 1/3/5/10y set regardless of horizon.

    Contract (deliberately conservative about the acceptance gate):

      * ``horizon_years is None`` or ``horizon_years == 10`` -> the historical
        default ``RISK_HORIZONS`` (``(1, 3, 5, 10)``) is returned UNCHANGED, so
        a no-flag / default run reproduces today's exact table byte-for-byte.
      * otherwise -> a deterministic, sorted, de-duplicated tuple that always
        CONTAINS ``horizon_years`` itself and places a shorter and a longer
        outcome on either side of it (a "near" point at roughly ⅓ and a "far"
        point at roughly 2× the target), anchored on a short reference point so
        the early-life picture never disappears.

    Purely arithmetic — no randomness — so the chosen grid is itself
    reproducible (the bootstrap seeding lives in ``RISK_SEED``).
    """
    # Default / explicit-10 -> preserve today's table exactly (acceptance gate).
    if horizon_years is None or horizon_years == 10:
        return RISK_HORIZONS

    h = int(horizon_years)
    if h < 1:                       # defensive: callers validate >= 1 upstream
        h = 1

    # Bracket points around the target: a short anchor, a "near" point below
    # the target, the target itself, and a "far" point above it.  max()/int()
    # keep everything a positive whole number of years and collapse degenerate
    # cases (e.g. very short horizons) onto the target without duplication.
    near = max(1, h // 3)           # ~⅓ of the way in — the shorter outcome
    far = max(h + 1, h * 2)         # ~2× the target — the longer outcome
    candidates = {1, near, h, far}  # 1y anchor keeps the early-life picture
    return tuple(sorted(candidates))


def _bootstrap_terminal(port, horizon_years, n_sims, block, rng):
    """Block-bootstrap ``n_sims`` terminal compounded returns over the horizon."""
    import numpy as np

    T = horizon_years * TRADING_DAYS
    n = len(port)
    nb = int(np.ceil(T / block))
    out = np.empty(n_sims, dtype=float)
    chunk = 4000
    for off in range(0, n_sims, chunk):
        m = min(chunk, n_sims - off)
        starts = rng.integers(0, n - block, size=(m, nb))
        idx = starts[:, :, None] + np.arange(block)[None, None, :]
        paths = port[idx].reshape(m, nb * block)[:, :T]
        out[off:off + m] = np.prod(1.0 + paths, axis=1) - 1.0
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_risk_profile(
    allocations: Mapping[str, float],
    *,
    horizon_years: int | None = None,
    horizons: tuple[int, ...] | None = None,
    n_sims: int = RISK_N_SIMS,
    block_days: int = RISK_BLOCK_DAYS,
    seed: int = RISK_SEED,
) -> RiskProfileResult:
    """
    Build a return-distribution risk profile for ``allocations`` (a mapping
    of ticker -> weight; pass ``final_proposal.allocations`` directly).

    Reporting horizons are chosen so the table BRACKETS the run's investment
    horizon:

      * pass ``horizon_years`` (the run's ``--horizon-years`` value) and the
        grid is derived via :func:`bracket_horizons` to surround that target
        with a shorter and a longer outcome.  ``horizon_years`` of ``None`` or
        ``10`` (the default) preserves today's fixed ``(1, 3, 5, 10)`` table —
        the acceptance gate for a no-flag run.
      * pass ``horizons`` to override the grid explicitly (wins over
        ``horizon_years``); when neither is supplied the historical default
        ``RISK_HORIZONS`` is used.

    Never raises — on any failure the returned result's ``error`` field is
    populated so the report can render a helpful note.
    """
    # Resolve the reporting grid: an explicit ``horizons`` override wins; else
    # bracket the run's ``horizon_years`` (which falls back to the default
    # 1/3/5/10y set when None or 10).
    if horizons is None:
        horizons = bracket_horizons(horizon_years)
    print("\n" + "=" * 60)
    print("RISK PROFILE — bootstrapping the return distribution …")
    print("=" * 60)

    result = RiskProfileResult(
        n_sims=n_sims,
        block_days=block_days,
        seed=seed,
        disclaimer=RISK_DISCLAIMER,
        computed_at=datetime.now().isoformat(timespec="seconds"),
    )

    # --- Lazy imports so a missing scientific stack degrades gracefully ---
    try:
        import logging
        import numpy as np
        import pandas as pd
        import yfinance as yf
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    except ImportError as exc:
        result.error = (
            f"required library missing ({exc}). Run `uv sync` from the "
            f"testbench root to install numpy / pandas / yfinance."
        )
        print(f"  ⚠️  {result.error}")
        return result

    # --- Resolve proxies + aggregate weights ---
    proxy_weights_raw, substitutions, _ = _resolve_proxies(allocations)
    result.proxy_substitutions = substitutions
    if not proxy_weights_raw:
        result.error = "no positive-weight holdings to model"
        print(f"  ⚠️  {result.error}")
        return result

    proxy_tickers = list(proxy_weights_raw.keys())
    if substitutions:
        for s in substitutions:
            print(f"  ↪ proxy: {s['original']:18s} → {s['proxy']}")

    # --- Download daily history ---
    try:
        px = yf.download(
            proxy_tickers,
            start=RISK_HISTORY_START,
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
        px = px.to_frame(name=proxy_tickers[0])

    # --- Drop tickers with no usable data (e.g., synthetic option legs) ---
    present = [t for t in proxy_tickers if t in px.columns and px[t].notna().any()]
    dropped_proxies = [t for t in proxy_tickers if t not in present]
    # Map dropped proxies back to original ticker names for the report.
    if dropped_proxies:
        inv = {s["proxy"]: s["original"] for s in substitutions}
        result.dropped_tickers = sorted(
            {inv.get(t, t) for t in dropped_proxies}
        )
        for t in result.dropped_tickers:
            print(f"  ✗ dropped (no priceable history): {t}")

    if not present:
        result.error = "none of the holdings could be priced via yfinance"
        print(f"  ⚠️  {result.error}")
        return result

    # Per-ticker inception BEFORE the common dropna, so we can report which
    # holding limited how far back the window reaches.
    raw_first = {t: px[t].first_valid_index() for t in present}
    px = px[present].dropna()
    if len(px) < 2 * block_days:
        result.error = (
            f"insufficient overlapping history ({len(px)} days) for the "
            f"common holding set — need at least {2 * block_days}"
        )
        print(f"  ⚠️  {result.error}")
        return result

    # --- Renormalize the modeled weights over what survived ---
    kept_weight = sum(proxy_weights_raw[t] for t in present)
    weights = pd.Series(
        {t: proxy_weights_raw[t] / kept_weight for t in present}
    )
    result.coverage_weight = round(kept_weight, 6)
    result.modeled_weights = {t: round(float(w), 6) for t, w in weights.items()}

    # --- Identify the holding that limited the start of the window ---
    limiting = max(raw_first, key=lambda t: raw_first[t]) if raw_first else None
    inv = {s["proxy"]: s["original"] for s in substitutions}
    result.limiting_ticker = inv.get(limiting, limiting)

    # --- Build the daily portfolio return series ---
    rets = px[present].pct_change().dropna()
    port = (rets @ weights).to_numpy()
    n = len(port)
    result.sample_start = str(rets.index[0].date())
    result.sample_end = str(rets.index[-1].date())
    result.sample_days = n
    result.sample_years = round(n / TRADING_DAYS, 1)
    result.includes_2008 = rets.index[0] <= pd.Timestamp("2008-01-01")
    result.annualized_return = round(
        float((1.0 + port).prod() ** (TRADING_DAYS / n) - 1.0), 6
    )
    result.annualized_vol = round(float(port.std() * np.sqrt(TRADING_DAYS)), 6)

    print(
        f"  Sample: {result.sample_start} → {result.sample_end} "
        f"(~{result.sample_years:g}y, "
        f"{'incl. 2008' if result.includes_2008 else 'NO 2008'}); "
        f"coverage {result.coverage_weight:.0%}; "
        f"ann.ret {result.annualized_return:.1%}, vol {result.annualized_vol:.1%}"
    )

    # --- Monte-Carlo each horizon ---
    rng = np.random.default_rng(seed)
    for h in horizons:
        term = _bootstrap_terminal(port, h, n_sims, block_days, rng)
        stats = HorizonStats(
            horizon_years=h,
            median=round(float(np.median(term)), 6),
            mean=round(float(term.mean()), 6),
            prob_end_down=round(float(np.mean(term < 0.0)), 6),
            bad_5th=round(float(np.percentile(term, 5)), 6),
            severe_1st=round(float(np.percentile(term, 1)), 6),
        )
        result.horizons.append(stats)
        print(
            f"  {h:2d}y: median {stats.median:+6.0%}  "
            f"P(end down) {stats.prob_end_down:4.0%}  "
            f"bad(5th) {stats.bad_5th:+6.0%}  severe(1st) {stats.severe_1st:+6.0%}"
        )

    return result
