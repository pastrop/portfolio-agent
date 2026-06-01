"""
Portfolio Optimization Harness — orchestrator + CLI
====================================================
A multi-agent harness inspired by Anthropic's "Harness design for
long-running application development" blog post, applied to financial
portfolio optimisation.

Composition (Planner → (Generator ↔ Evaluator + intra-Advisor) × N →
Selector → Refiner → Re-evaluate → Final Advisor → Pricing) lives in
this file.  The five LLM agents themselves are in ``agents.py``; the
Anthropic client wrapper and JSON parser are in ``api.py``; the
dataclasses passed between agents are in ``models.py``; the markdown
report writer is in ``report.py``; the yfinance-backed lot-size step
is in ``pricing.py``.

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
    run_advisor,
    run_evaluator,
    run_generator,
    run_planner,
    run_refiner,
)
from models import (
    AdvisorOutput,
    EvaluationResult,
    InvestmentSpec,
    PortfolioProposal,
)
from pricing import DEFAULT_CAPITAL, PRICING_DISCLAIMER, run_pricing
from report import write_markdown_report

# ---------------------------------------------------------------------------
# Orchestrator-only policy constants
# ---------------------------------------------------------------------------
# (Model / token / retry config lives in api.py; per-agent model selection
# is api.MODEL / api.PLANNER_MODEL / api.ADVISOR_MODEL.  The CLI block at
# the bottom of this file patches those module-level globals when --test
# or --model X is provided.)
MAX_ITERATIONS = 3          # generator ↔ evaluator rounds
PASS_THRESHOLD = 7          # minimum average score (out of 10) to pass QA
TARGET_MAX_LOSS = 0.05      # default loss budget we want expected_max_drawdown to land on (overridable via --max-loss)
# The under-utilisation band auto-derives from TARGET_MAX_LOSS: a drawdown
# below this fraction of the budget is "wasting risk capacity".  Whenever
# TARGET_MAX_LOSS is patched (CLI --max-loss, server runner), recompute
# UNDER_UTILISATION_BAND = UNDER_UTILISATION_RATIO * TARGET_MAX_LOSS.
UNDER_UTILISATION_RATIO = 0.8
UNDER_UTILISATION_BAND = UNDER_UTILISATION_RATIO * TARGET_MAX_LOSS
# Threshold for which Advisor-flagged correlation pairs count as
# actionable feedback for the next Generator iteration.  The Advisor
# surfaces pairs at |ρ| ≥ 0.5 for the report, but for prompt feedback
# we only want the egregious overlaps (≥ 0.7) so the next prompt
# isn't cluttered with marginal pairs.
ADVISOR_FEEDBACK_RHO_THRESHOLD = 0.7


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
def _format_advisor_feedback(advisor_output: AdvisorOutput | None) -> str:
    """
    Format the Advisor's correlation findings as a concrete, actionable
    block for the next Generator iteration's prompt.  Only pairs whose
    absolute correlation is ≥ ``ADVISOR_FEEDBACK_RHO_THRESHOLD`` (default
    0.7) are surfaced — below that we don't want to clutter the prompt
    with marginal overlaps.

    Returns "" when the Advisor is missing, found nothing actionable,
    or wasn't run.
    """
    if advisor_output is None:
        return ""
    pairs = []
    for p in (advisor_output.correlation_pairs or []):
        try:
            rho = float(p.get("rho", 0) or 0)
        except (TypeError, ValueError):
            continue
        if abs(rho) >= ADVISOR_FEEDBACK_RHO_THRESHOLD:
            pairs.append((p.get("a", ""), p.get("b", ""), rho))
    suggestions = advisor_output.suggestions or []
    if not pairs and not suggestions:
        return ""

    lines: list[str] = [
        "ADVISOR FINDINGS — OVERLAPPING HOLDINGS IN YOUR PREVIOUS PORTFOLIO:",
        "",
        (
            f"The previous portfolio contains highly correlated holdings "
            f"(|ρ| ≥ {ADVISOR_FEEDBACK_RHO_THRESHOLD:.2f}). Each pair "
            f"represents redundant exposure — collapse each pair into "
            f"ONE ticker in your next iteration."
        ),
        "",
    ]
    if pairs:
        lines.append("Correlated pairs to consolidate:")
        for a, b, rho in sorted(pairs, key=lambda x: -abs(x[2])):
            lines.append(f"  • {a} ↔ {b}  (ρ ≈ {rho:+.2f})")
        lines.append("")
    if suggestions:
        lines.append("Specific consolidation suggestions:")
        for s in suggestions:
            merge_from = " + ".join(s.get("merge_from") or []) or "(unspecified)"
            merge_into = s.get("merge_into") or "(unspecified)"
            tradeoff = s.get("tradeoff") or ""
            line = f"  • Replace {merge_from} → {merge_into}"
            if tradeoff:
                line += f"  (tradeoff: {tradeoff})"
            lines.append(line)
        lines.append("")
    return "\n".join(lines)


def _build_feedback(
    evaluation: EvaluationResult,
    proposal: PortfolioProposal,
    advisor_output: AdvisorOutput | None = None,
) -> str:
    """
    Produce the next round's feedback string based on the current evaluation.

    Three regimes (selects the BASE feedback text):
      • Failed QA              → pass the critique through unchanged (today's behaviour).
      • Passed but over target → tell the generator to bring drawdown down to
                                 ~TARGET_MAX_LOSS without sacrificing scores.
      • Passed but well
        under target           → tell the generator the risk budget is being
                                 wasted and to push drawdown closer to (but
                                 under) TARGET_MAX_LOSS.
      • Passed and near target → still ask for a refinement attempt — the
                                 selector will keep the best across iterations.

    If ``advisor_output`` is provided, its concrete correlation-pair
    findings (≥ ``ADVISOR_FEEDBACK_RHO_THRESHOLD``) are prepended to the
    base feedback.  This converts the Advisor from a one-shot report
    decoration into a real signal in the generator ↔ evaluator loop —
    the Generator gets actual overlapping pairs to fix in the next
    iteration, instead of a vague "avoid correlation" rule it cannot
    apply (LLMs can't reliably estimate ρ from memory).
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

    advisor_section = _format_advisor_feedback(advisor_output)
    if advisor_section:
        return advisor_section + "\n" + base
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


def run_harness(
    user_goal: str,
    *,
    refine: bool = True,
    advise: bool = True,
    price: bool = True,
    capital: float = DEFAULT_CAPITAL,
) -> dict[str, Any]:
    """
    Main entry point.  Runs the full Planner → Generator ↔ Evaluator loop,
    then (by default) a post-selection REFINER pass, an ADVISOR pass, and
    a PRICING / lot-size feasibility pass.

    Flow:
      1. Planner expands the goal into a full investment spec.
      2. MAX_ITERATIONS rounds of Generator ↔ Evaluator.  Always runs
         all rounds; never breaks early.
      3. Select the best PASSING iteration whose expected_max_drawdown
         is closest to TARGET_MAX_LOSS.  Falls back to the last iteration
         if nothing passed.
      4. If `refine=True`, run a single Refiner pass on the selected
         portfolio: Refiner addresses every critique point, then the
         Evaluator re-scores.  If the refined version passes QA, it is
         PROMOTED to `final_proposal`; otherwise the selected one stays
         as the final answer.
      5. If `advise=True`, run a single ADVISOR pass on the final
         portfolio.  The Advisor is read-only — it never modifies the
         portfolio.  It returns a pairwise correlation snapshot and a
         list of structured "merge X+Y into Z" suggestions with explicit
         trade-offs, so the human reader can decide whether to apply any.
      6. If `price=True`, fetch the latest price for each ticker in the
         final portfolio via yfinance and compute a whole-share lot-size
         feasibility check against `capital` USD.  Per-ticker failures
         (unknown ticker, network blip, model-invented pseudo-ticker) are
         recorded gracefully and never abort the pipeline.
    """
    # --- Step 1: Plan ---
    spec = run_planner(user_goal)

    history: list[dict] = []
    proposals: list[PortfolioProposal] = []
    evaluations: list[EvaluationResult] = []
    feedback: str | None = None

    for i in range(1, MAX_ITERATIONS + 1):
        # --- Step 2: Generate ---
        proposal = run_generator(
            spec, feedback=feedback, iteration=i, max_loss=TARGET_MAX_LOSS,
        )

        # --- Step 3: Evaluate ---
        evaluation = run_evaluator(
            spec, proposal,
            pass_threshold=PASS_THRESHOLD, max_loss=TARGET_MAX_LOSS,
        )

        proposals.append(proposal)
        evaluations.append(evaluation)

        # --- Step 3a: Intra-iteration Advisor (feedback signal) ---
        # Run the Advisor on this iteration's portfolio so the next
        # round's Generator prompt gets CONCRETE correlation pairs to
        # collapse, rather than a vague "avoid overlap" rule that the
        # LLM cannot apply.  Skip on the last iteration (no next round),
        # when advise is disabled, or when the proposal produced no
        # allocations.  The final Advisor pass (Step 6) still runs for
        # the report and may see a different portfolio.
        # None ⇒ didn't run; int (including 0) ⇒ ran and fed N pairs forward
        intra_advisor: AdvisorOutput | None = None
        intra_pairs_count: int | None = None
        if (
            advise
            and i < MAX_ITERATIONS
            and proposal.allocations
        ):
            intra_advisor = run_advisor(proposal)
            intra_pairs_count = sum(
                1 for p in (intra_advisor.correlation_pairs or [])
                if abs(float(p.get("rho", 0) or 0))
                   >= ADVISOR_FEEDBACK_RHO_THRESHOLD
            )

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
            "intra_advisor_pairs_count": intra_pairs_count,
            "selected": False,
        })

        print(f"\n--- Iteration {i} result ---")
        print(f"  Scores              : {evaluation.scores}")
        print(f"  Average             : {evaluation.average_score}")
        print(f"  Passed              : {evaluation.passed}")
        print(f"  Expected max loss   : {proposal.expected_max_drawdown:.2%}")
        print(f"  Target max loss     : {TARGET_MAX_LOSS:.0%}")
        if intra_advisor is not None:
            print(
                f"  Advisor (pre-feedback) flagged {intra_pairs_count} "
                f"pair(s) at |ρ| ≥ {ADVISOR_FEEDBACK_RHO_THRESHOLD:.2f} "
                f"— feeding into iteration {i + 1}"
            )

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

        feedback = _build_feedback(
            evaluation, proposal, advisor_output=intra_advisor,
        )

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

    # --- Step 6: Advisor (read-only, advisory only) ---
    advisor_block: dict[str, Any] = {
        "performed": False,
        "skipped_reason": None,
        "suggestions": [],
        "correlation_pairs": [],
        "notes": "",
        "raw_text": "",
    }
    if not advise:
        advisor_block["skipped_reason"] = "disabled via --no-advisor / --test"
        print("\n(Advisor step skipped.)\n")
    elif not final_proposal.allocations:
        advisor_block["skipped_reason"] = "no allocations to advise on"
        print("\n(Advisor step skipped — no allocations.)\n")
    else:
        advisor_output = run_advisor(final_proposal)
        advisor_block.update({
            "performed": True,
            "suggestions": advisor_output.suggestions,
            "correlation_pairs": advisor_output.correlation_pairs,
            "notes": advisor_output.notes,
            "raw_text": advisor_output.raw_text,
        })
        n_pairs = len(advisor_output.correlation_pairs)
        n_sugg = len(advisor_output.suggestions)
        print(
            f"\n💡  Advisor returned {n_pairs} correlated pair(s) and "
            f"{n_sugg} consolidation suggestion(s).  The portfolio was "
            f"NOT modified — see the report for details.\n"
        )

    # --- Step 7: Pricing & lot-size feasibility (yfinance) ---
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

    return {
        "model": api.MODEL,
        "max_iterations": MAX_ITERATIONS,
        "pass_threshold": PASS_THRESHOLD,
        "target_max_loss": TARGET_MAX_LOSS,
        "spec": asdict(spec),
        "final_proposal": asdict(final_proposal),
        "final_evaluation": asdict(final_evaluation),
        "selected_iteration": selected_idx,
        "selected_proposal": asdict(selected_proposal),
        "selected_evaluation": asdict(selected_evaluation),
        "iteration_history": history,
        "refinement": refinement_block,
        "advisor": advisor_block,
        "pricing": pricing_block,
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
              uv run python harness.py --no-refine
              uv run python harness.py --no-advisor
              uv run python harness.py --no-prices
              uv run python harness.py --capital 250000
              uv run python harness.py --max-loss 0.10
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
            "Override the model. Accepts short aliases (haiku, sonnet, opus) or a "
            "full Anthropic model ID. Ignored when --test is also set."
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
        "--no-advisor",
        action="store_true",
        help=(
            "Skip the post-selection Advisor pass. By default, after the "
            "final portfolio is determined, an Advisor agent produces a "
            "pairwise correlation snapshot and concrete consolidation "
            "suggestions (advisory only — never modifies the portfolio). "
            "--test implies --no-advisor."
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

    # Apply --test first (it takes precedence), then individual overrides.
    refine = not args.no_refine
    advise = not args.no_advisor
    price = not args.no_prices
    capital = args.capital
    if args.test:
        # --test forces ALL agents to haiku (and disables everything else).
        # Per-agent model globals live in api.py — patch them there so
        # call_claude picks them up on the next call.
        api.MODEL = _MODEL_ALIASES["haiku"]
        api.PLANNER_MODEL = _MODEL_ALIASES["haiku"]
        api.ADVISOR_MODEL = _MODEL_ALIASES["haiku"]
        MAX_ITERATIONS = 1
        refine = False   # --test always implies --no-refine
        advise = False   # --test always implies --no-advisor
        price = False    # --test always implies --no-prices
        print(
            f"[TEST MODE] model={api.MODEL}  iterations={MAX_ITERATIONS}  "
            f"refine={refine}  advise={advise}  price={price}"
        )
    else:
        if args.model:
            # --model X overrides ALL agents to X (escape hatch — runs
            # the entire pipeline on one tier, useful for cost / quality
            # comparison or when one tier is overloaded).
            resolved = _MODEL_ALIASES.get(args.model, args.model)
            api.MODEL = resolved
            api.PLANNER_MODEL = resolved
            api.ADVISOR_MODEL = resolved
            print(f"[model override — all agents] {api.MODEL}")
        else:
            # No override — show the per-agent assignments so the user
            # knows the pipeline isn't uniform.
            print(
                f"[per-agent models] generator/evaluator/refiner={api.MODEL}  "
                f"planner={api.PLANNER_MODEL}  advisor={api.ADVISOR_MODEL}"
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
        if not advise:
            print("[advisor disabled]")
        if not price:
            print("[pricing disabled]")
        else:
            print(f"[pricing enabled  capital=${capital:,.0f}]")

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

    result = run_harness(
        goal,
        refine=refine,
        advise=advise,
        price=price,
        capital=capital,
    )

    # Dump full trace to a JSON file for inspection
    with open("harness_output.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Also write a human-readable Markdown report
    write_markdown_report(result, "harness_output.md")

    print("\n" + "=" * 60)
    print("FINAL PORTFOLIO")
    print("=" * 60)
    sel = result.get("selected_iteration")
    if sel is not None:
        print(f"  Selected iteration : {sel} of {MAX_ITERATIONS} (best passing)")
    else:
        print(f"  Selected iteration : last of {MAX_ITERATIONS} (no iteration passed)")
    print(f"  Target max loss    : {result['target_max_loss']:.1%}\n")

    if result["final_proposal"]:
        for ticker, weight in result["final_proposal"]["allocations"].items():
            print(f"  {ticker:20s}  {weight:6.1%}")
        print(f"\n  Expected return    : {result['final_proposal']['expected_annual_return']:.1%}")
        print(f"  Expected max loss  : {result['final_proposal']['expected_max_drawdown']:.1%}")
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
