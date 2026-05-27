"""
Pricing & lot-size feasibility step for the Portfolio Optimization Harness.

This module is deliberately self-contained: it has no dependency on
``harness.py`` (and therefore no risk of circular imports).  ``run_pricing``
takes the allocations dict directly rather than a ``PortfolioProposal``
dataclass, which keeps the API surface small and lets pricing be exercised
independently for tests or one-off scripts.

For each ticker in the allocations:

    target_$  = capital * weight
    shares    = floor(target_$ / price)
    actual_$  = shares * price
    actual_w  = actual_$ / capital
    drift     = actual_w - weight

Per-ticker failures (unknown ticker, network blip, model-invented
pseudo-ticker like "SPX_PUT_SPREAD") are recorded gracefully and never
abort the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
DEFAULT_CAPITAL = 100_000.0  # USD assumed for whole-share lot-size check
PRICING_DISCLAIMER = (
    "Prices are fetched from Yahoo Finance via the `yfinance` library and "
    "reflect the most recent available quote (last close or recent "
    "intraday). They may differ from real-time market data, and Yahoo's "
    "unofficial API can occasionally return stale or missing values. "
    "Do NOT rely on these numbers for trading decisions."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TickerPricing:
    """One row of the pricing / lot-size feasibility table."""
    ticker: str
    weight: float                       # target weight from the final proposal
    status: str = "ok"                  # "ok" | "error"
    price: float | None = None          # last available quote in USD
    target_dollars: float = 0.0         # capital * weight
    shares: int = 0                     # floor(target_dollars / price)
    actual_dollars: float = 0.0         # shares * price
    actual_weight: float = 0.0          # actual_dollars / capital
    weight_drift: float = 0.0           # actual_weight - target weight (signed)
    error: str | None = None


@dataclass
class PricingResult:
    """Aggregated pricing + lot-size feasibility report."""
    capital: float = 0.0
    total_invested: float = 0.0
    leftover_cash: float = 0.0
    max_abs_drift: float = 0.0          # largest |weight_drift| across OK rows
    rows: list[TickerPricing] = field(default_factory=list)
    failed_tickers: list[str] = field(default_factory=list)
    disclaimer: str = ""
    source: str = "yfinance"
    fetched_at: str = ""                # ISO timestamp of the fetch
    error: str | None = None            # set when pricing as a whole could not run


# ---------------------------------------------------------------------------
# Internal helper: single-ticker price fetch
# ---------------------------------------------------------------------------
def _fetch_one_price(ticker: str) -> tuple[float | None, str | None]:
    """
    Fetch the latest available price for one ticker via yfinance.
    Returns (price, error_message).  On success: (float, None).
    On any failure (unknown ticker, network blip, NaN result):
    (None, "human-readable reason").  Never raises.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        return None, f"yfinance not installed ({exc})"
    try:
        t = yf.Ticker(ticker)
        price: float | None = None
        # fast_info is the cheap path; supports dict and attribute access
        # across yfinance versions.
        try:
            fi = t.fast_info
            try:
                price = fi["last_price"]
            except (KeyError, TypeError):
                price = getattr(fi, "last_price", None)
        except Exception:
            price = None
        # NaN guard (yfinance sometimes returns NaN for unknown tickers).
        if price is not None and isinstance(price, float) and price != price:
            price = None
        # Fall back to a 1-day history pull if fast_info gave us nothing.
        if price is None:
            try:
                hist = t.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            except Exception:
                price = None
        if price is None or price <= 0:
            return None, "no price returned (ticker may be unrecognised)"
        return float(price), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_pricing(
    allocations: Mapping[str, float],
    capital: float,
) -> PricingResult:
    """
    Fetch the latest price for each ticker in ``allocations`` and compute
    a whole-share lot-size feasibility check against ``capital``.

    ``allocations`` is a mapping of ticker -> target weight (e.g.
    ``{"VOO": 0.4, "TLT": 0.2, ...}``).  Pass ``final_proposal.allocations``
    directly from the harness.

    Per-ticker failures (unknown ticker, network blip, model-invented
    pseudo-ticker) are recorded in the result as rows with status="error"
    and an explanatory message; the pipeline is never aborted by them.

    If yfinance itself is missing, returns a result whose ``error`` field
    is populated so the markdown report can render a helpful note.
    """
    print("\n" + "=" * 60)
    print(
        f"PRICING — fetching latest prices from yfinance "
        f"(capital=${capital:,.0f}) …"
    )
    print("=" * 60)

    # Up-front import check — if yfinance is missing, return a marker
    # result so the report can render a helpful note instead of a stack trace.
    try:
        import yfinance as _yf  # noqa: F401
        import logging
        # yfinance is chatty about failed tickers — quiet it down so the
        # harness log stays readable.  We surface per-ticker errors ourselves.
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    except ImportError as exc:
        msg = (
            f"yfinance is not installed ({exc}). Run `uv sync` from the "
            f"testbench root to install it."
        )
        print(f"  ⚠️  {msg}")
        return PricingResult(
            capital=capital,
            disclaimer=PRICING_DISCLAIMER,
            error=msg,
            fetched_at=datetime.now().isoformat(timespec="seconds"),
        )

    rows: list[TickerPricing] = []
    failed: list[str] = []

    for ticker, weight in allocations.items():
        w = float(weight)
        target_dollars = capital * w
        price, err = _fetch_one_price(ticker)
        if price is None:
            print(f"  ✗ {ticker:25s}  {err}")
            rows.append(TickerPricing(
                ticker=ticker,
                weight=w,
                status="error",
                target_dollars=round(target_dollars, 2),
                error=err,
            ))
            failed.append(ticker)
            continue
        shares = int(target_dollars // price)
        actual_dollars = shares * price
        print(
            f"  ✓ {ticker:25s}  ${price:>10,.2f}   "
            f"{shares:>6d} sh   ${actual_dollars:>12,.2f}"
        )
        rows.append(TickerPricing(
            ticker=ticker,
            weight=w,
            status="ok",
            price=round(price, 4),
            target_dollars=round(target_dollars, 2),
            shares=shares,
            actual_dollars=round(actual_dollars, 2),
        ))

    total_invested = sum(r.actual_dollars for r in rows)
    leftover_cash = capital - total_invested
    # Fill in actual_weight / weight_drift now that the totals are known.
    for r in rows:
        if r.status == "ok" and capital > 0:
            r.actual_weight = round(r.actual_dollars / capital, 6)
            r.weight_drift = round(r.actual_weight - r.weight, 6)
    max_abs_drift = max(
        (abs(r.weight_drift) for r in rows if r.status == "ok"),
        default=0.0,
    )

    print(f"\n  Total invested  : ${total_invested:>12,.2f}")
    print(f"  Leftover cash   : ${leftover_cash:>12,.2f}")
    print(f"  Max |Δ weight|  : {max_abs_drift:.2%}")
    if failed:
        print(f"  Failed tickers  : {', '.join(failed)}")

    return PricingResult(
        capital=capital,
        total_invested=round(total_invested, 2),
        leftover_cash=round(leftover_cash, 2),
        max_abs_drift=round(max_abs_drift, 6),
        rows=rows,
        failed_tickers=failed,
        disclaimer=PRICING_DISCLAIMER,
        source="yfinance",
        fetched_at=datetime.now().isoformat(timespec="seconds"),
    )
