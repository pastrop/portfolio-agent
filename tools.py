"""
Tool implementations + schemas for the Portfolio Optimization Harness.

This module is the source of truth for:

  * ``compute_backtest(weights, start_date, end_date)`` — actual historical
    performance of a portfolio over a date range, backed by yfinance.
    Used by the Evaluator (to score CONSTRAINT_COMPLIANCE on real
    2008 / 2020 / 2022 losses instead of estimating from training memory)
    and the Refiner (to verify its revisions stay within the loss cap).

  * The JSON schemas Claude needs to call those tools:
      - ``COMPUTE_BACKTEST_TOOL`` (client-side, our handler runs)
      - ``WEB_SEARCH_TOOL``       (server-side, Anthropic runs it)

The handlers are kept small and HONEST about data limitations: when an
ETF wasn't around in the requested date range (e.g., DBMF didn't exist
in 2008), the result surfaces the missing-data list and the
``coverage_weight`` so the model can reason about partial coverage
rather than silently renormalising and producing a misleading number.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Tool schemas (declared up here so agents.py just imports them)
# ---------------------------------------------------------------------------

# Client-side tool: we provide the handler (compute_backtest below).
COMPUTE_BACKTEST_TOOL: dict[str, Any] = {
    "name": "compute_backtest",
    "description": (
        "Compute the ACTUAL historical performance of a portfolio over a "
        "date range, using daily total-return data from Yahoo Finance. "
        "Returns total return, max drawdown, volatility, best/worst day, "
        "and (importantly) which tickers had no data in the range "
        "(some ETFs didn't exist in 2008 — e.g., DBMF launched 2019). "
        "Use this to score constraint compliance against the spec's loss "
        "cap with real numbers instead of estimating losses from memory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "weights": {
                "type": "object",
                "additionalProperties": {"type": "number"},
                "description": (
                    "Ticker -> weight (e.g., {\"VOO\": 0.6, \"BND\": 0.4}). "
                    "Weights should sum to ~1.0; the tool does not "
                    "renormalise. Use real US-listed ETF tickers."
                ),
            },
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD (e.g., '2008-01-01').",
            },
            "end_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD (e.g., '2008-12-31').",
            },
        },
        "required": ["weights", "start_date", "end_date"],
    },
}


# Server-side tool: declared via {"type": ...}, no handler needed.
# Anthropic's infrastructure executes the searches and embeds the results
# in the same response.  ``max_uses`` keeps cost / latency bounded.
WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}


# ---------------------------------------------------------------------------
# compute_backtest handler
# ---------------------------------------------------------------------------
def compute_backtest(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Real backtest of ``weights`` from ``start_date`` to ``end_date`` using
    yfinance daily total-return data.

    Returns a dict the model can read directly:

        {
          "start_date":       "2008-01-01",
          "end_date":         "2008-12-31",
          "trading_days":     252,
          "tickers_with_data":    ["VOO", "BND", ...],
          "tickers_missing_data": ["DBMF"],  # didn't exist in this range
          "coverage_weight":      0.95,      # sum of weights with data
          "total_return":         -0.0451,   # cumulative, dividends reinvested
          "max_drawdown":         -0.0871,   # peak-to-trough on cumulative
          "annualised_volatility": 0.18,     # daily-stdev * sqrt(252)
          "best_day":             0.034,
          "worst_day":            -0.052,
          "note": "..."  # human-readable caveat when coverage_weight < 1.0
        }

    On any failure (yfinance missing, bad dates, all tickers unknown, etc.)
    raises with a clear message — call_claude catches this and feeds the
    error back to the model via an is_error=True tool_result, so the model
    can fix its input and retry.
    """
    # ---- Validate inputs ----
    weights = inputs.get("weights")
    start_date = inputs.get("start_date")
    end_date = inputs.get("end_date")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(
            "weights must be a non-empty dict of ticker -> number"
        )
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        raise ValueError("start_date and end_date must be YYYY-MM-DD strings")
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d")
        ed = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"date must be YYYY-MM-DD: {exc}"
        ) from exc
    if sd >= ed:
        raise ValueError(
            f"start_date ({start_date}) must be before end_date ({end_date})"
        )

    # ---- Lazy import yfinance ----
    # Lazy so test mocks don't need yfinance and so import-time of the
    # module is cheap.  Also quiet yfinance's chatty logger.
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            f"compute_backtest needs yfinance + pandas ({exc}). "
            f"Run `uv sync` from the testbench root."
        ) from exc
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    # ---- Download adjusted-close prices (total return, dividends reinvested) ----
    tickers = list(weights.keys())
    try:
        df = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,    # adjusted close = total return
            group_by="column",
            threads=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"yfinance download failed: {type(exc).__name__}: {exc}"
        ) from exc

    # yfinance returns a multi-level column dataframe for multiple tickers,
    # a flat one for a single ticker.  Normalise to a {ticker: Series} dict.
    if df is None or df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {tickers} in {start_date}..{end_date}"
        )

    closes: dict[str, Any] = {}
    if len(tickers) == 1:
        ticker = tickers[0]
        if "Close" in df.columns:
            closes[ticker] = df["Close"].dropna()
    else:
        # Multi-ticker: columns are a MultiIndex (Price, Ticker).
        # Extract the "Close" slice.
        try:
            close_df = df["Close"]
        except KeyError:
            # Some yfinance versions flip the multi-index levels.
            close_df = df.xs("Close", axis=1, level=0)
        for ticker in tickers:
            if ticker in close_df.columns:
                series = close_df[ticker].dropna()
                if not series.empty:
                    closes[ticker] = series

    tickers_with_data = [t for t in tickers if t in closes and len(closes[t]) > 1]
    tickers_missing_data = [t for t in tickers if t not in tickers_with_data]
    coverage_weight = round(
        sum(float(weights[t]) for t in tickers_with_data), 4
    )

    if not tickers_with_data:
        raise RuntimeError(
            f"none of {tickers} had usable data in {start_date}..{end_date} "
            f"(some ETFs didn't exist in this range; consider a later "
            f"date or different tickers)"
        )

    # ---- Compute aligned daily returns ----
    # Concatenate to a DataFrame, drop rows where ANY ticker has NaN so
    # the per-day weighted sum stays consistent.
    aligned = pd.concat(
        {t: closes[t] for t in tickers_with_data},
        axis=1,
        join="inner",
    )
    daily_returns = aligned.pct_change().dropna()
    if daily_returns.empty:
        raise RuntimeError(
            f"only {len(aligned)} aligned trading day(s) in "
            f"{start_date}..{end_date} — too few to compute returns"
        )

    # ---- Portfolio daily return = sum(weight_i * return_i) over covered tickers ----
    weights_series = pd.Series(
        {t: float(weights[t]) for t in tickers_with_data}
    )
    portfolio_daily = daily_returns @ weights_series

    cumulative = (1.0 + portfolio_daily).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0)

    # Max drawdown: peak-to-trough on the cumulative return path.
    running_peak = cumulative.cummax()
    drawdown_series = (cumulative / running_peak) - 1.0
    max_drawdown = float(drawdown_series.min())

    daily_std = float(portfolio_daily.std())
    annualised_vol = daily_std * math.sqrt(252) if not math.isnan(daily_std) else 0.0
    best_day = float(portfolio_daily.max())
    worst_day = float(portfolio_daily.min())

    note = ""
    if tickers_missing_data:
        missing_str = ", ".join(tickers_missing_data)
        note = (
            f"Partial coverage: {coverage_weight:.0%} of the portfolio was "
            f"backtested. Missing tickers ({missing_str}) had no data in "
            f"this range — they were excluded from the calculation, so "
            f"the reported numbers describe only the covered subset. "
            f"Reason about whether their absence materially changes the "
            f"scenario's outcome."
        )

    return {
        "start_date": start_date,
        "end_date": end_date,
        "trading_days": int(len(portfolio_daily)),
        "tickers_with_data": tickers_with_data,
        "tickers_missing_data": tickers_missing_data,
        "coverage_weight": coverage_weight,
        "total_return": round(total_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "annualised_volatility": round(annualised_vol, 6),
        "best_day": round(best_day, 6),
        "worst_day": round(worst_day, 6),
        "note": note,
    }
