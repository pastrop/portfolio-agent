"""
Markdown report writer for the Portfolio Optimization Harness.

This module is deliberately self-contained: it takes the result dict
produced by ``harness.run_harness`` and writes a human-readable
``.md`` file.  It imports nothing from ``harness`` — every piece of
run config (model, iteration count, pass threshold, target max loss)
is read from the result dict, with sensible fallbacks for older
trace files that pre-date those keys.

Keeping the writer dependency-free means an old ``harness_output.json``
can be re-rendered without rerunning the pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Fallbacks used only when a result dict pre-dates the relevant key.
# Real runs always populate these via ``run_harness``.
# ---------------------------------------------------------------------------
_FALLBACK_MODEL = "(unknown model)"
_FALLBACK_MAX_ITERATIONS = "?"
_FALLBACK_PASS_THRESHOLD = 7
_FALLBACK_TARGET_MAX_LOSS = 0.05


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def _format_value(v: Any) -> str:
    """Render a JSON-ish value as a readable markdown fragment."""
    if v is None:
        return "_(none)_"
    if isinstance(v, str):
        return v.strip() or "_(empty)_"
    if isinstance(v, (list, dict)):
        return "```json\n" + json.dumps(v, indent=2) + "\n```"
    return str(v)


def _format_weight(w: Any) -> str:
    try:
        return f"{float(w):.2%}"
    except (TypeError, ValueError):
        return str(w)


# ---------------------------------------------------------------------------
# Pricing & risk sections — shared by the optimized and preservation reports.
# Both are no-LLM steps, so they are populated in BOTH regimes; factoring
# them out lets the preservation report reuse them verbatim.
# ---------------------------------------------------------------------------
def _push_pricing_section(push: Any, result: dict[str, Any]) -> None:
    """Render the 'Latest Prices & Lot-Size Feasibility' section."""
    pricing = result.get("pricing") or {}
    if pricing.get("performed"):
        push("## Latest Prices & Lot-Size Feasibility")
        push("")
        if pricing.get("disclaimer"):
            push(f"> ⚠️ **Data source disclaimer.** {pricing['disclaimer']}")
            push("")

        cap = float(pricing.get("capital", 0) or 0)
        total_inv = float(pricing.get("total_invested", 0) or 0)
        leftover = float(pricing.get("leftover_cash", 0) or 0)
        max_drift = float(pricing.get("max_abs_drift", 0) or 0)
        fetched = pricing.get("fetched_at", "")
        failed_tickers = pricing.get("failed_tickers") or []

        push(f"- **Assumed capital:** ${cap:,.2f}")
        push(f"- **Total invested (whole shares):** ${total_inv:,.2f}")
        push(f"- **Leftover cash:** ${leftover:,.2f}")
        push(f"- **Max |weight drift|:** {max_drift:.2%}")
        if fetched:
            push(f"- **Fetched at:** {fetched}")
        if failed_tickers:
            ft = ", ".join(f"`{t}`" for t in failed_tickers)
            push(f"- **Unpriced tickers:** {ft}")
        push("")

        rows_p = pricing.get("rows") or []
        if rows_p:
            push("| Ticker | Price | Target W. | Target $ | Shares | "
                 "Actual $ | Actual W. | Δ Weight | Status |")
            push("|--------|------:|----------:|---------:|------:|"
                 "---------:|----------:|---------:|--------|")
            for r in rows_p:
                ticker = r.get("ticker", "")
                status = r.get("status", "ok")
                weight = float(r.get("weight", 0) or 0)
                target_d = float(r.get("target_dollars", 0) or 0)
                if status == "ok":
                    price_v = float(r.get("price", 0) or 0)
                    shares_v = int(r.get("shares", 0) or 0)
                    actual_d = float(r.get("actual_dollars", 0) or 0)
                    actual_w = float(r.get("actual_weight", 0) or 0)
                    drift = float(r.get("weight_drift", 0) or 0)
                    sign = "+" if drift > 0 else ""
                    push(
                        f"| `{ticker}` | ${price_v:,.2f} | "
                        f"{weight:.2%} | ${target_d:,.2f} | "
                        f"{shares_v:,d} | ${actual_d:,.2f} | "
                        f"{actual_w:.2%} | {sign}{drift:.2%} | ok |"
                    )
                else:
                    err = r.get("error") or "unpriced"
                    push(
                        f"| `{ticker}` | — | {weight:.2%} | "
                        f"${target_d:,.2f} | — | — | — | — | {err} |"
                    )
            push("")
    elif pricing.get("skipped_reason"):
        push("## Latest Prices & Lot-Size Feasibility")
        push("")
        push(f"_Pricing pass skipped — {pricing['skipped_reason']}._")
        push("")
    elif pricing.get("error"):
        push("## Latest Prices & Lot-Size Feasibility")
        push("")
        push(f"_Pricing pass could not run — {pricing['error']}_")
        push("")


def _push_risk_section(push: Any, result: dict[str, Any]) -> None:
    """Render the 'Return Distribution (Monte-Carlo)' section."""
    risk = result.get("risk_profile") or {}
    if risk.get("performed"):
        push("## Return Distribution (Monte-Carlo)")
        push("")
        if risk.get("disclaimer"):
            push(f"> ⚠️ **How to read this.** {risk['disclaimer']}")
            push("")

        start = risk.get("sample_start", "")
        end = risk.get("sample_end", "")
        years = risk.get("sample_years", 0) or 0
        incl08 = risk.get("includes_2008")
        cov = float(risk.get("coverage_weight", 0) or 0)
        ann_r = float(risk.get("annualized_return", 0) or 0)
        ann_v = float(risk.get("annualized_vol", 0) or 0)
        limiting = risk.get("limiting_ticker")

        push(
            f"- **Sample window:** {start} → {end} (~{years:g}y), "
            f"{'**includes** the 2008 crisis' if incl08 else '⚠️ **does NOT include** 2008'}"
        )
        if not incl08 and limiting:
            push(
                f"  - window limited by `{limiting}` (no long-history proxy); "
                f"tail estimates are correspondingly benign"
            )
        push(f"- **Coverage:** {cov:.0%} of the book modeled "
             f"(realized sample: {ann_r:.1%} return, {ann_v:.1%} volatility)")

        subs = risk.get("proxy_substitutions") or []
        if subs:
            sub_str = ", ".join(
                f"`{s.get('original')}`→`{s.get('proxy')}`" for s in subs
            )
            push(f"- **Long-history proxies used:** {sub_str}")
        dropped = risk.get("dropped_tickers") or []
        if dropped:
            dr = ", ".join(f"`{t}`" for t in dropped)
            push(
                f"- **Excluded (not priceable, e.g. option overlays):** {dr} "
                f"— their weight was renormalized away, so the modeled "
                f"downside is *more conservative* than the hedged portfolio"
            )
        push("")

        horizons = risk.get("horizons") or []
        if horizons:
            push("Invest a lump sum today, look again after each horizon — "
                 "where your money lands:")
            push("")
            push("| Horizon | Median outcome | Chance of ending down | "
                 "Bad case (1-in-20) | Severe case (1-in-100) |")
            push("|---------|---------------:|----------------------:|"
                 "-------------------:|-----------------------:|")
            for hz in horizons:
                yrs = hz.get("horizon_years", 0)
                med = float(hz.get("median", 0) or 0)
                pdn = float(hz.get("prob_end_down", 0) or 0)
                bad = float(hz.get("bad_5th", 0) or 0)
                sev = float(hz.get("severe_1st", 0) or 0)
                push(
                    f"| {yrs} year{'s' if yrs != 1 else ''} | {med:+.0%} | "
                    f"{pdn:.0%} | {bad:+.0%} | {sev:+.0%} |"
                )
            push("")
            push(
                "_**Median outcome** = the typical (50/50) total return. "
                "**Chance of ending down** = probability you finish with less "
                "than you started. **Bad / Severe case** = the 1-in-20 and "
                "1-in-100 unlucky finishes._"
            )
            push("")
    elif risk.get("skipped_reason"):
        push("## Return Distribution (Monte-Carlo)")
        push("")
        push(f"_Risk-profile pass skipped — {risk['skipped_reason']}._")
        push("")
    elif risk.get("error"):
        push("## Return Distribution (Monte-Carlo)")
        push("")
        push(f"_Risk-profile pass could not run — {risk['error']}_")
        push("")


def _push_correlation_section(push: Any, result: dict[str, Any]) -> None:
    """
    Render the 'Correlation Snapshot (computed)' section.

    This replaces the old Advisor agent's recalled-from-memory correlation
    table: the numbers here are computed from real yfinance history (see
    ``correlation.py``), so they are trustworthy where the Advisor's were
    not (it systematically misjudged the cash / short-duration sleeve).
    """
    corr = result.get("correlation") or {}
    if corr.get("performed"):
        push("## Correlation Snapshot (computed)")
        push("")
        if corr.get("disclaimer"):
            push(f"> ⚠️ **How to read this.** {corr['disclaimer']}")
            push("")

        start = corr.get("sample_start", "")
        end = corr.get("sample_end", "")
        win = corr.get("window_years", 0) or 0
        cov = float(corr.get("coverage_weight", 0) or 0)
        freq = corr.get("frequency", "daily")
        high_thr = float(corr.get("high_threshold", 0.85) or 0.85)
        rep_thr = float(corr.get("report_threshold", 0.5) or 0.5)

        push(f"- **Sample window:** {start} → {end} (~{win:g}y {freq} returns)")
        push(f"- **Coverage:** {cov:.0%} of the book priced")
        dropped = corr.get("dropped_tickers") or []
        if dropped:
            dr = ", ".join(f"`{t}`" for t in dropped)
            push(
                f"- **Excluded (no priceable history, e.g. option overlays):** "
                f"{dr}"
            )
        push("")

        pairs = corr.get("pairs") or []
        high_n = int(corr.get("high_pairs_count", 0) or 0)
        if pairs:
            push(
                f"Pairs with |ρ| ≥ {rep_thr:.2f} shown, strongest first. "
                f"**{high_n}** pair(s) at |ρ| ≥ {high_thr:.2f} are flagged as "
                f"highly redundant (⚠)."
            )
            push("")
            push("| Ticker A | Ticker B | ρ | |")
            push("|----------|----------|---:|:--|")
            for p in pairs:
                try:
                    rho_s = f"{float(p.get('rho', 0)):+.2f}"
                except (TypeError, ValueError):
                    rho_s = str(p.get("rho", ""))
                flag = " ⚠ high" if p.get("high") else ""
                push(
                    f"| `{p.get('a', '')}` | `{p.get('b', '')}` | "
                    f"{rho_s} |{flag} |"
                )
            push("")
        else:
            push(
                f"_No pairs at |ρ| ≥ {rep_thr:.2f} — the holdings are well "
                f"diversified by this measure._"
            )
            push("")
    elif corr.get("skipped_reason"):
        push("## Correlation Snapshot (computed)")
        push("")
        push(f"_Correlation pass skipped — {corr['skipped_reason']}._")
        push("")
    elif corr.get("error"):
        push("## Correlation Snapshot (computed)")
        push("")
        push(f"_Correlation pass could not run — {corr['error']}_")
        push("")


def _push_loss_floor_section(push: Any, result: dict[str, Any]) -> None:
    """
    Render the 'Loss-Floor Compliance (delivered book)' section.

    A deterministic backtest of the HELD allocations vs the annual loss cap
    (see ``loss_floor.py``).  It exists to make a specific failure visible:
    the pipeline can certify ``≤cap`` by crediting an enforcement mechanism
    (e.g. an options overlay) that lives only in the spec, while the delivered
    holdings are long-only and unhedged.  This section reports the GROSS
    calendar-year losses of the actual holdings and flags when the ``≤cap``
    claim depends on a hedge the portfolio does not hold.
    """
    lf = result.get("loss_floor") or {}
    title = "## Loss-Floor Compliance (delivered book)"
    if lf.get("performed"):
        push(title)
        push("")
        if lf.get("disclaimer"):
            push(f"> ⚠️ **How to read this.** {lf['disclaimer']}")
            push("")

        cap = float(lf.get("max_loss", 0.05) or 0.05)
        worst = lf.get("worst_gross_annual_loss")
        wy = lf.get("worst_year")
        legs = lf.get("hedge_legs") or []

        if lf.get("relies_on_unheld_mechanism"):
            push(
                f"> ❌ **NOT COMPLIANT AS DELIVERED.** The listed holdings lost "
                f"**{worst:.1%} gross in {wy}** (cap **−{cap:.0%}**) and hold "
                f"**no hedge leg** — the ≤{cap:.0%} claim depends on a mechanism "
                f"(e.g. an options overlay) that is **not in this portfolio**. "
                f"A reader who buys these weights is unhedged."
            )
        elif lf.get("organic_pass"):
            push(
                f"> ✅ **Organically compliant.** Worst gross calendar-year loss "
                f"was **{worst:.1%}** ({wy}), within the **−{cap:.0%}** cap "
                f"without relying on any hedge."
            )
        elif worst is not None:
            leg_s = ", ".join(f"`{t}`" for t in legs) or "—"
            push(
                f"> ⚠️ **Hedged book.** Gross loss breaches the cap "
                f"(**{worst:.1%}** in {wy}), but the portfolio HOLDS a hedge "
                f"leg ({leg_s}); compliance depends on that held overlay "
                f"performing as designed."
            )
        else:
            push(f"_Could not judge: {lf.get('skipped_reason') or 'no usable data'}._")
        push("")

        per_year = lf.get("per_year") or []
        if per_year:
            push(
                "Gross calendar-year backtest of the DELIVERED holdings "
                "(proxy-substituted for pre-inception coverage):"
            )
            push("")
            push("| Year | Gross annual return | Max drawdown | Coverage | vs cap |")
            push("|------|--------------------:|-------------:|---------:|:-------|")
            for p in per_year:
                tr = p.get("gross_annual_return")
                if tr is None:
                    push(f"| {p.get('year', '')} | _no data_ | — | — | — |")
                    continue
                dd = p.get("gross_max_drawdown")
                cov = float(p.get("coverage_weight", 0) or 0)
                breach = "❌ breach" if p.get("breaches_cap") else "✅ ok"
                dd_s = f"{dd:.1%}" if isinstance(dd, (int, float)) else "—"
                push(
                    f"| {p.get('year', '')} | {tr:+.1%} | {dd_s} | "
                    f"{cov:.0%} | {breach} |"
                )
            push("")
        subs = lf.get("proxy_substitutions") or []
        if subs:
            sub_s = ", ".join(f"`{s['original']}`→`{s['proxy']}`" for s in subs)
            push(f"- **Proxy substitutions (for coverage):** {sub_s}")
            push("")
    elif lf.get("skipped_reason"):
        push(title)
        push("")
        push(f"_Loss-floor check skipped — {lf['skipped_reason']}._")
        push("")
    elif lf.get("error"):
        push(title)
        push("")
        push(f"_Loss-floor check could not run — {lf['error']}_")
        push("")


# ---------------------------------------------------------------------------
# Horizon / posture header line (shared by both regimes)
# ---------------------------------------------------------------------------
def _push_horizon_header(push: Any, horizon_years: Any, horizon_posture: Any) -> None:
    """
    Emit the one-line horizon/posture header bullet, if the result dict
    carries the Phase 2 keys.  Pre-Phase-2 traces lack both fields, in
    which case we stay silent rather than print a misleading default —
    the rest of the report renders exactly as it did before.
    """
    if horizon_years is None and horizon_posture is None:
        return
    parts: list[str] = []
    if horizon_years is not None:
        try:
            yrs = int(horizon_years)
            parts.append(f"{yrs} year{'s' if yrs != 1 else ''}")
        except (TypeError, ValueError):
            parts.append(str(horizon_years))
    if horizon_posture:
        parts.append(f"posture: {horizon_posture}")
    push(f"- **Investment horizon:** {' — '.join(parts)}")


# ---------------------------------------------------------------------------
# Preservation short-circuit report (<3y, no LLM agents ran)
# ---------------------------------------------------------------------------
def _write_preservation_report(result: dict[str, Any], path: str) -> None:
    """
    Render the capital-preservation story for a ``mode == "preservation"``
    run.  Under the 3-year floor the optimizer is intentionally bypassed:
    no Planner/Generator/Evaluator/Refiner ran, so the usual iteration /
    refinement sections would be empty shells.  We skip them entirely and
    instead show:

      * a banner explaining the short-circuit,
      * the redirect message (why the optimizer was not run),
      * the deterministic template allocation table, and
      * the (no-LLM) pricing, risk-profile, and correlation sections.

    ``final_proposal.expected_annual_return`` / ``expected_max_drawdown``
    are ``None`` in this regime, so every numeric format here guards
    against missing/None values.
    """
    final_p = result.get("final_proposal") or {}
    model = result.get("model", _FALLBACK_MODEL)
    horizon_years = result.get("horizon_years")
    horizon_posture = result.get("horizon_posture")
    band = result.get("preservation_band")
    redirect = result.get("redirect_message")

    out: list[str] = []
    push = out.append

    # ---- Header ----
    push("# Portfolio Optimization Harness — Report")
    push("")
    push(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    push(f"- **Model:** `{model}` _(no LLM agents were invoked for this run)_")
    _push_horizon_header(push, horizon_years, horizon_posture)
    push(f"- **Mode:** capital preservation (short-circuit)")
    if band:
        push(f"- **Preservation band:** {band}")
    push("")

    # ---- Preservation banner ----
    push("## ⚓ Capital-Preservation Short-Circuit")
    push("")
    push(
        "This run fell **below the 3-year horizon floor**, so the "
        "optimization pipeline was intentionally **not run**.  At horizons "
        "this short, risk *capacity* — not risk *tolerance* — is the "
        "binding constraint: there is too little time to recover from a "
        "drawdown, so the harness returns a fixed, deterministic "
        "capital-preservation template instead of an optimized portfolio."
    )
    push("")
    if redirect:
        push("> " + redirect.strip().replace("\n", "\n> "))
        push("")

    # ---- Template allocation ----
    push("## Capital-Preservation Allocation (template)")
    push("")
    allocs = final_p.get("allocations") or {}
    descs = final_p.get("descriptions") or {}
    if allocs:
        if descs:
            push("| Ticker | Weight | Description |")
            push("|--------|-------:|-------------|")
            for ticker, weight in allocs.items():
                desc = descs.get(ticker, "")
                push(f"| `{ticker}` | {_format_weight(weight)} | {desc} |")
        else:
            push("| Ticker | Weight |")
            push("|--------|-------:|")
            for ticker, weight in allocs.items():
                push(f"| `{ticker}` | {_format_weight(weight)} |")
        push("")
    else:
        push("_(No allocation present in result.)_")
        push("")

    # expected_* are None in preservation mode — surface them honestly
    # rather than coercing to 0.00% (which would imply a modeled figure).
    er = final_p.get("expected_annual_return")
    dd = final_p.get("expected_max_drawdown")
    er_s = _format_weight(er) if er is not None else "_(not modeled)_"
    dd_s = _format_weight(dd) if dd is not None else "_(not modeled)_"
    push(f"- **Expected annual return:** {er_s}")
    push(f"- **Expected max drawdown:** {dd_s}")
    push("")
    if final_p.get("methodology"):
        push("### Methodology")
        push("")
        push(_format_value(final_p["methodology"]))
        push("")
    if final_p.get("rationale"):
        push("### Rationale")
        push("")
        push(_format_value(final_p["rationale"]))
        push("")

    # ---- Pricing & lot-size feasibility (no-LLM step, populated) ----
    _push_pricing_section(push, result)

    # ---- Return distribution (Monte-Carlo risk profile, no-LLM step) ----
    _push_risk_section(push, result)

    # ---- Correlation snapshot (no-LLM step, populated) ----
    _push_correlation_section(push, result)

    # ---- Loss-floor compliance (no-LLM step) ----
    _push_loss_floor_section(push, result)

    with open(path, "w") as f:
        f.write("\n".join(out))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def write_markdown_report(result: dict[str, Any], path: str) -> None:
    """
    Write a human-readable Markdown report mirroring the JSON trace.
    Preserves every structured field; raw model text is tucked into
    collapsible <details> blocks so the file stays scannable.
    """
    spec = result.get("spec") or {}
    final_p = result.get("final_proposal") or {}
    final_e = result.get("final_evaluation") or {}
    history = result.get("iteration_history") or []
    sel = result.get("selected_iteration")
    target = result.get("target_max_loss", _FALLBACK_TARGET_MAX_LOSS)
    model = result.get("model", _FALLBACK_MODEL)
    max_iters = result.get("max_iterations", _FALLBACK_MAX_ITERATIONS)
    pass_threshold = result.get("pass_threshold", _FALLBACK_PASS_THRESHOLD)
    refinement = result.get("refinement") or {}
    refined_promoted = bool(refinement.get("promoted"))

    # Phase 2 horizon-awareness. A result dict that pre-dates Phase 2 has
    # no `mode` key; per the locked spec a missing mode is treated as
    # "optimized" for back-compat. `horizon_years` / `horizon_posture` are
    # self-describing echoes — absent on old traces, so guard with None.
    mode = result.get("mode", "optimized")
    horizon_years = result.get("horizon_years")
    horizon_posture = result.get("horizon_posture")

    # ---- Preservation short-circuit (<3y) ----
    # In preservation mode NO LLM agents ran: iteration_history and refinement
    # are empty/skipped stubs, so we render a dedicated story (banner +
    # redirect + template table + pricing + risk + correlation) and return —
    # we deliberately do NOT fall through to the empty optimization shells.
    if mode == "preservation":
        _write_preservation_report(result, path)
        return

    out: list[str] = []
    push = out.append

    # ---- Header ----
    push("# Portfolio Optimization Harness — Report")
    push("")
    push(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    push(f"- **Model:** `{model}`")
    _push_horizon_header(push, horizon_years, horizon_posture)
    push(f"- **Iterations run:** {len(history)} of {max_iters}")
    if sel is not None:
        push(f"- **Selected iteration:** {sel} — best passing portfolio, closest to {target:.1%} target")
    else:
        # No iteration passed QA. The orchestrator's policy is to keep
        # the FIRST iteration in that case; derive its index from the
        # `selected: True` flag in the history list so the report
        # stays accurate even if the policy changes later.
        fallback_idx = next(
            (h.get("iteration") for h in history if h.get("selected")),
            1,
        )
        push(
            f"- **Selected iteration:** {fallback_idx} (no iteration "
            f"passed QA — kept first attempt as fallback; later "
            f"iterations tend to over-correct toward conservative returns)"
        )
    if refinement.get("performed"):
        if refined_promoted:
            push("- **Refinement:** performed — refined portfolio PROMOTED to final")
        else:
            push("- **Refinement:** performed — refined version NOT promoted (kept selected as final)")
    elif refinement.get("skipped_reason"):
        push(f"- **Refinement:** skipped ({refinement['skipped_reason']})")
    correlation_hdr = result.get("correlation") or {}
    if correlation_hdr.get("performed"):
        n_pairs = len(correlation_hdr.get("pairs") or [])
        n_high = int(correlation_hdr.get("high_pairs_count", 0) or 0)
        push(
            f"- **Correlation:** computed — {n_pairs} pair(s) at |ρ| ≥ 0.5, "
            f"{n_high} highly redundant (|ρ| ≥ 0.85)"
        )
    elif correlation_hdr.get("skipped_reason"):
        push(f"- **Correlation:** skipped ({correlation_hdr['skipped_reason']})")
    pricing_hdr = result.get("pricing") or {}
    if pricing_hdr.get("performed"):
        n_failed = len(pricing_hdr.get("failed_tickers") or [])
        cap_h = pricing_hdr.get("capital", 0.0) or 0.0
        suffix = f", {n_failed} ticker(s) unpriced" if n_failed else ""
        push(
            f"- **Pricing:** performed — yfinance quotes for "
            f"${cap_h:,.0f} capital lot-size check{suffix}"
        )
    elif pricing_hdr.get("skipped_reason"):
        push(f"- **Pricing:** skipped ({pricing_hdr['skipped_reason']})")
    loss_floor_hdr = result.get("loss_floor") or {}
    if loss_floor_hdr.get("performed"):
        wl = loss_floor_hdr.get("worst_gross_annual_loss")
        wy = loss_floor_hdr.get("worst_year")
        if loss_floor_hdr.get("relies_on_unheld_mechanism"):
            push(
                f"- **⚠️ Loss-floor:** NOT compliant as delivered — lost "
                f"{wl:.1%} gross in {wy} with no hedge held; the ≤{target:.0%} "
                f"claim relies on a mechanism NOT in the portfolio"
            )
        elif loss_floor_hdr.get("organic_pass"):
            push(
                f"- **Loss-floor:** organically within {target:.0%} "
                f"(worst gross {wl:.1%} in {wy})"
            )
        elif wl is not None:
            push(
                f"- **Loss-floor:** breaches {target:.0%} gross "
                f"({wl:.1%} in {wy}) but holds a hedge leg"
            )
    elif loss_floor_hdr.get("skipped_reason"):
        push(f"- **Loss-floor:** skipped ({loss_floor_hdr['skipped_reason']})")
    push(f"- **Target max loss:** {target:.1%}")
    push(f"- **Pass threshold:** average score ≥ {pass_threshold} with no single score ≤ 4 AND evaluator says passed")
    push("")

    # ---- Final portfolio ----
    if refined_promoted:
        push("## Final Portfolio (refined)")
    else:
        push("## Final Portfolio (selected)")
    push("")
    allocs = final_p.get("allocations") or {}
    descs = final_p.get("descriptions") or {}
    if allocs:
        if descs:
            push("| Ticker | Weight | Description |")
            push("|--------|-------:|-------------|")
            for ticker, weight in allocs.items():
                desc = descs.get(ticker, "")
                push(f"| `{ticker}` | {_format_weight(weight)} | {desc} |")
        else:
            push("| Ticker | Weight |")
            push("|--------|-------:|")
            for ticker, weight in allocs.items():
                push(f"| `{ticker}` | {_format_weight(weight)} |")
        push("")
    try:
        er = float(final_p.get("expected_annual_return", 0))
        dd = float(final_p.get("expected_max_drawdown", 0))
        push(f"- **Expected annual return:** {er:.2%}")
        push(f"- **Expected max drawdown:** {dd:.2%}  (target ≤ {target:.1%})")
    except (TypeError, ValueError):
        push(f"- **Expected annual return:** {final_p.get('expected_annual_return')}")
        push(f"- **Expected max drawdown:** {final_p.get('expected_max_drawdown')}")
    push(f"- **Passed QA:** {final_e.get('passed', False)}")
    push(f"- **Average QA score:** {final_e.get('average_score', 0)}")
    push("")

    # ---- Iteration summary ----
    push("## Iteration Summary")
    push("")
    push("| # | Avg Score | Exp. Return | Max Loss | Δ to target | Passed | Selected |")
    push("|--:|----------:|------------:|---------:|------------:|:------:|:--------:|")
    for h in history:
        try:
            dd_i = float(h.get("expected_max_drawdown", 0))
            er_i = float(h.get("expected_return", 0))
        except (TypeError, ValueError):
            dd_i, er_i = 0.0, 0.0
        dist = abs(dd_i - target)
        passed = "YES" if h.get("passed") else "NO"
        sel_mark = "★" if h.get("selected") else ""
        push(
            f"| {h.get('iteration')} | {h.get('average_score', 0):.2f} | "
            f"{er_i:.2%} | {dd_i:.2%} | {dist:.2%} | {passed} | {sel_mark} |"
        )
    push("")

    # ---- Investment spec ----
    push("## Investment Spec (from Planner)")
    push("")
    for key in ("objective", "constraints", "asset_universe", "risk_budget", "evaluation_criteria"):
        if key in spec and spec[key] not in (None, ""):
            label = key.replace("_", " ").title()
            push(f"### {label}")
            push("")
            push(_format_value(spec[key]))
            push("")
    if spec.get("raw_text"):
        push("<details><summary>Planner raw response</summary>")
        push("")
        push("```")
        push(spec["raw_text"].rstrip())
        push("```")
        push("")
        push("</details>")
        push("")

    # ---- Selected portfolio methodology / rationale ----
    push("## Selected Portfolio — Methodology & Rationale")
    push("")
    if final_p.get("methodology"):
        push("### Methodology")
        push("")
        push(_format_value(final_p["methodology"]))
        push("")
    if final_p.get("rationale"):
        push("### Rationale")
        push("")
        push(_format_value(final_p["rationale"]))
        push("")
    if final_p.get("raw_text"):
        push("<details><summary>Generator raw response</summary>")
        push("")
        push("```")
        push(final_p["raw_text"].rstrip())
        push("```")
        push("")
        push("</details>")
        push("")

    # ---- Selected portfolio evaluation ----
    push("## Selected Portfolio — Evaluator Report")
    push("")
    scores = final_e.get("scores") or {}
    if scores:
        push("### Scores")
        push("")
        push("| Criterion | Score |")
        push("|-----------|------:|")
        for k, v in scores.items():
            push(f"| {k.replace('_', ' ')} | {v} |")
        push(f"| **Average** | **{final_e.get('average_score', 0)}** |")
        push("")
    if final_e.get("critique"):
        push("### Critique")
        push("")
        push(_format_value(final_e["critique"]))
        push("")
    if final_e.get("raw_text"):
        push("<details><summary>Evaluator raw response</summary>")
        push("")
        push("```")
        push(final_e["raw_text"].rstrip())
        push("```")
        push("")
        push("</details>")
        push("")

    # ---- Post-selection refinement ----
    if refinement.get("performed"):
        push("## Post-Selection Refinement")
        push("")
        if refined_promoted:
            push(
                "The Refiner addressed every critique point above and the "
                "Evaluator re-scored the result. **The refined portfolio "
                "passed QA and was promoted to `Final Portfolio` (top of "
                "this report).** Section below shows the before/after."
            )
        else:
            push(
                "The Refiner addressed every critique point above, but the "
                "refined portfolio **did not pass QA on re-evaluation** (or "
                f"exceeded the {target:.1%} drawdown target), so the originally "
                "selected portfolio was kept as the final answer. Section "
                "below shows what the Refiner produced for comparison."
            )
        push("")

        imp = refinement.get("improvements") or {}
        ref_p = refinement.get("refined_proposal") or {}
        ref_e = refinement.get("refined_evaluation") or {}

        # Score deltas table
        score_deltas = imp.get("score_deltas") or {}
        if score_deltas:
            push("### Score deltas (Selected → Refined)")
            push("")
            push("| Criterion | Selected | Refined | Δ |")
            push("|-----------|---------:|--------:|--:|")
            for k, d in score_deltas.items():
                delta = d.get("delta", 0)
                sign = "+" if delta > 0 else ""
                push(
                    f"| {k.replace('_', ' ')} | {d.get('before', '?')} | "
                    f"{d.get('after', '?')} | {sign}{delta} |"
                )
            avg_b = imp.get("average_score_before", 0)
            avg_a = imp.get("average_score_after", 0)
            avg_d = imp.get("average_score_delta", 0)
            sign = "+" if avg_d > 0 else ""
            push(f"| **Average** | **{avg_b}** | **{avg_a}** | **{sign}{avg_d}** |")
            push("")

        # Headline metric deltas
        push("### Portfolio metric deltas")
        push("")
        try:
            er_b = float(imp.get("expected_return_before", 0))
            er_a = float(imp.get("expected_return_after", 0))
            dd_b = float(imp.get("expected_max_drawdown_before", 0))
            dd_a = float(imp.get("expected_max_drawdown_after", 0))
            push("| Metric | Selected | Refined | Δ |")
            push("|--------|---------:|--------:|--:|")
            push(
                f"| Expected annual return | {er_b:.2%} | {er_a:.2%} | "
                f"{'+' if er_a >= er_b else ''}{(er_a - er_b):.2%} |"
            )
            push(
                f"| Expected max drawdown | {dd_b:.2%} | {dd_a:.2%} | "
                f"{'+' if dd_a >= dd_b else ''}{(dd_a - dd_b):.2%} |"
            )
            push(
                f"| Distance to {target:.0%} target | "
                f"{abs(dd_b - target):.2%} | {abs(dd_a - target):.2%} | "
                f"{'+' if abs(dd_a - target) >= abs(dd_b - target) else ''}"
                f"{(abs(dd_a - target) - abs(dd_b - target)):.2%} |"
            )
            push(
                f"| Passed QA | {imp.get('passed_before')} | "
                f"{imp.get('passed_after')} | — |"
            )
            push("")
        except (TypeError, ValueError):
            push("_(Could not compute metric deltas — see raw values below.)_")
            push("")

        # Allocation changes
        changes = imp.get("allocation_changes") or []
        if changes:
            push("### Allocation changes")
            push("")
            push("| Ticker | Selected | Refined | Δ Weight | Kind |")
            push("|--------|---------:|--------:|---------:|------|")
            for c in changes:
                before = float(c.get("before", 0))
                after = float(c.get("after", 0))
                delta = float(c.get("delta", 0))
                sign = "+" if delta > 0 else ""
                before_s = f"{before:.2%}" if before > 0 else "—"
                after_s = f"{after:.2%}" if after > 0 else "—"
                push(
                    f"| `{c.get('ticker')}` | {before_s} | {after_s} | "
                    f"{sign}{delta:.2%} | {c.get('kind')} |"
                )
            push("")

        # Refined methodology / rationale
        if ref_p.get("methodology"):
            push("### Refined methodology")
            push("")
            push(_format_value(ref_p["methodology"]))
            push("")
        if ref_p.get("rationale"):
            push("### Refined rationale (point-by-point response to critique)")
            push("")
            push(_format_value(ref_p["rationale"]))
            push("")

        # Refined evaluator report
        if ref_e.get("scores"):
            push("### Re-evaluator scores")
            push("")
            push("| Criterion | Score |")
            push("|-----------|------:|")
            for k, v in ref_e["scores"].items():
                push(f"| {k.replace('_', ' ')} | {v} |")
            push(f"| **Average** | **{ref_e.get('average_score', 0)}** |")
            push(f"| **Passed** | **{ref_e.get('passed', False)}** |")
            push("")
        if ref_e.get("critique"):
            push("### Re-evaluator critique")
            push("")
            push(_format_value(ref_e["critique"]))
            push("")

        if ref_p.get("raw_text") or ref_e.get("raw_text"):
            push("<details><summary>Refiner & re-evaluator raw responses</summary>")
            push("")
            if ref_p.get("raw_text"):
                push("```")
                push("--- Refiner raw response ---")
                push(ref_p["raw_text"].rstrip())
                push("```")
                push("")
            if ref_e.get("raw_text"):
                push("```")
                push("--- Re-evaluator raw response ---")
                push(ref_e["raw_text"].rstrip())
                push("```")
                push("")
            push("</details>")
            push("")

    # ---- Pricing & lot-size feasibility ----
    _push_pricing_section(push, result)

    # ---- Return distribution (Monte-Carlo risk profile) ----
    _push_risk_section(push, result)

    # ---- Correlation snapshot (computed, no LLM) ----
    _push_correlation_section(push, result)

    # ---- Loss-floor compliance (no-LLM step) ----
    _push_loss_floor_section(push, result)

    # ---- Per-iteration detail ----
    push("## Iteration History (detailed)")
    push("")
    for h in history:
        sel_mark = " — ★ selected" if h.get("selected") else ""
        push(f"### Iteration {h.get('iteration')}{sel_mark}")
        push("")
        try:
            dd_i = float(h.get("expected_max_drawdown", 0))
            er_i = float(h.get("expected_return", 0))
            push(f"- **Passed:** {h.get('passed')}")
            push(f"- **Average score:** {h.get('average_score', 0):.2f}")
            push(f"- **Expected return:** {er_i:.2%}")
            push(f"- **Expected max drawdown:** {dd_i:.2%}")
        except (TypeError, ValueError):
            push(f"- **Passed:** {h.get('passed')}")
            push(f"- **Average score:** {h.get('average_score')}")
            push(f"- **Expected return:** {h.get('expected_return')}")
            push(f"- **Expected max drawdown:** {h.get('expected_max_drawdown')}")
        push("")
        scores_i = h.get("scores") or {}
        if scores_i:
            push("**Scores:**")
            push("")
            push("| Criterion | Score |")
            push("|-----------|------:|")
            for k, v in scores_i.items():
                push(f"| {k.replace('_', ' ')} | {v} |")
            push("")
        allocs_i = h.get("allocations") or {}
        descs_i = h.get("descriptions") or {}
        if allocs_i:
            push("**Allocations:**")
            push("")
            if descs_i:
                push("| Ticker | Weight | Description |")
                push("|--------|-------:|-------------|")
                for ticker, weight in allocs_i.items():
                    desc = descs_i.get(ticker, "")
                    push(f"| `{ticker}` | {_format_weight(weight)} | {desc} |")
            else:
                push("| Ticker | Weight |")
                push("|--------|-------:|")
                for ticker, weight in allocs_i.items():
                    push(f"| `{ticker}` | {_format_weight(weight)} |")
            push("")
        snip = h.get("critique_snippet")
        if snip:
            push("**Critique snippet (first 300 chars):**")
            push("")
            for line in snip.splitlines():
                push(f"> {line}")
            push("")

    with open(path, "w") as f:
        f.write("\n".join(out))
