"""
The five LLM agents for the Portfolio Optimization Harness.

Each agent is a (SYSTEM prompt + ``run_*`` function) pair.  The
orchestrator in ``harness.py`` composes them in the
Planner → (Generator ↔ Evaluator) × N → Refiner → Re-evaluate →
Advisor sequence; the Advisor is also run between iterations to feed
its correlation findings back into the next round's Generator prompt.

Dependencies:
* ``models`` for the dataclass return types (InvestmentSpec,
  PortfolioProposal, EvaluationResult, AdvisorOutput)
* ``api`` for the Anthropic client wrapper (``call_claude``) and the
  fail-soft JSON parser (``_parse_json_response``).  Per-agent model
  selection (``api.PLANNER_MODEL``, ``api.ADVISOR_MODEL``) and the
  Refiner's larger token budget (``api.REFINER_MAX_TOKENS``) are read
  at call time via ``import api``, so CLI patches in ``harness.py``
  take effect without any extra wiring.

Orchestrator policy constants (``PASS_THRESHOLD``, ``MAX_ITERATIONS``)
that affect agent behavior are accepted as keyword arguments rather
than imported, so this module has no dependency on ``harness``.
"""

from __future__ import annotations

import textwrap

import api
from api import call_claude, _parse_json_response
from models import (
    AdvisorOutput,
    EvaluationResult,
    InvestmentSpec,
    PortfolioProposal,
)
from tools import (
    COMPUTE_BACKTEST_TOOL,
    WEB_SEARCH_TOOL,
    compute_backtest,
)


# Tool registries — agents.py knows which agent gets which tool.
# Server-side tools (web_search) have no handler — Anthropic runs them.
# Client-side tools (compute_backtest) need a handler in the dispatch map.
_PLANNER_TOOLS = [WEB_SEARCH_TOOL]
_BACKTEST_TOOLS = [COMPUTE_BACKTEST_TOOL]
_BACKTEST_HANDLERS = {"compute_backtest": compute_backtest}


# ---------------------------------------------------------------------------
# AGENT 1 — PLANNER
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = textwrap.dedent("""\
    You are an expert investment planner.  Your job is to take a brief,
    high-level investment goal and expand it into a detailed, actionable
    investment specification that a portfolio construction engine can execute.

    Be ambitious but realistic.  Think about:
    • Which asset classes and specific instruments are available to a US retail
      investor (stocks, ETFs, bonds, REITs, commodities, options overlays, etc.)
    • Concrete risk constraints (max drawdown, volatility targets, correlation
      limits, concentration limits)
    • Benchmark and evaluation criteria (how will we know the portfolio is good?)
    • Time horizon assumptions and rebalancing cadence
    • Tail-risk scenarios the portfolio must survive

    AVAILABLE TOOL — web_search:
    Your training data has a knowledge cutoff.  Before writing the spec,
    use `web_search` (server-side, executed by Anthropic) to check
    CURRENT market conditions and bake them into the spec.  You have
    up to 5 searches — be specific about what you ask.  Suggested
    queries (run only the ones you actually need):
      • "current US federal funds rate"
      • "current US 10-year Treasury yield"
      • "S&P 500 current level"
      • "current VIX level"
      • Any recent (last 30 days) macro news that should inform asset
        universe selection or the risk budget (e.g., Fed policy shift,
        major credit event, geopolitical regime change).
    Cite the numbers you find in the spec's `objective` or
    `risk_budget` field so the downstream construction agent knows
    what regime you assumed.

    Your FINAL response must be ONLY a raw JSON object with these keys:
      objective, constraints, asset_universe, risk_budget, evaluation_criteria

    Strict format rules — your response will be machine-parsed:
      • The response MUST begin with `{` and end with `}`.  Nothing
        before, nothing after.
      • NO markdown fences (no ```json, no ```).
      • NO prose preamble such as "Here is the spec:", "Based on my
        searches:", "Now I will provide:", etc.
      • If you used web_search, bake the findings DIRECTLY into the
        relevant field values (objective, risk_budget, etc.).  Do NOT
        narrate the search process or summarise what you found in
        prose — just incorporate the numbers into the fields.
""")


def _investor_profile_block(horizon_years: int, posture: dict) -> str:
    """
    Render the horizon-aware INVESTOR PROFILE hard-constraint block injected
    into the Planner's USER message (optimized regime, horizon >= 3y).

    The harness derives the glide-path posture deterministically (label +
    growth ceiling) and passes it in; this block translates it into an
    explicit, machine-honoured constraint for the downstream construction
    agents: cap GROWTH-asset weight (equity + REIT + commodity + high-yield)
    at the ceiling, and fill the remainder with high-grade bonds / TIPS /
    cash.  The Evaluator enforces the ceiling mechanically (see
    ``run_evaluator``'s ``growth_ceiling`` check), so this is not a polite
    suggestion — the spec must encode it as a binding constraint.
    """
    ceiling = posture["growth_ceiling"]
    return textwrap.dedent(f"""\
        INVESTOR PROFILE — HARD CONSTRAINTS (horizon-derived, binding):
        • Investment horizon: {horizon_years} year(s).
        • Risk posture: {posture['label']} (glide-path band {posture['band']}).
        • GROWTH-ASSET CEILING: the combined weight of growth assets —
          equity (US / international / EM / factor / dividend / small-cap),
          REITs, broad commodities, gold, managed futures, and HIGH-YIELD
          credit — MUST NOT exceed {ceiling:.0%} of the portfolio.
        • The REMAINDER (at least {1 - ceiling:.0%}) MUST be high-grade
          defensive assets: investment-grade / Treasury / aggregate bonds,
          TIPS / inflation-linked, and cash / T-bills.
        • The max annual loss budget still applies INDEPENDENTLY: keep the
          portfolio within its stated max-loss constraint AND under this
          growth ceiling — whichever binds tighter wins.
        Bake these limits into the spec's `constraints` and `risk_budget`
        fields as explicit, numeric rules so the construction engine can
        honour them.
    """)


def run_planner(
    user_goal: str,
    *,
    horizon_years: int = 10,
    posture: dict | None = None,
) -> InvestmentSpec:
    """
    Expand the user goal into a full investment spec.

    When ``posture`` is provided (the harness's deterministic glide-path
    result for ``horizon_years``), an INVESTOR PROFILE hard-constraint block
    is prepended to the Planner's USER message so the spec encodes the
    horizon-derived growth-asset ceiling.  When ``posture`` is None the
    behaviour is byte-identical to the pre-horizon Planner (back-compat for
    any caller that does not pass a posture).
    """
    print("\n" + "=" * 60)
    print("PLANNER — expanding user goal into investment spec …")
    print("=" * 60)

    user_msg = user_goal
    if posture is not None:
        user_msg = (
            _investor_profile_block(horizon_years, posture)
            + "\n"
            + user_goal
        )

    raw = call_claude(
        PLANNER_SYSTEM, user_msg,
        model=api.PLANNER_MODEL,
        max_tokens=api.PLANNER_MAX_TOKENS,   # spec + web_search needs >4096
        tools=_PLANNER_TOOLS,                # web_search (server-side)
    )
    print(raw[:500], "…\n" if len(raw) > 500 else "\n")

    data = _parse_json_response(raw, agent="planner")

    return InvestmentSpec(
        objective=data.get("objective", ""),
        constraints=data.get("constraints", ""),
        asset_universe=data.get("asset_universe", ""),
        risk_budget=data.get("risk_budget", ""),
        evaluation_criteria=data.get("evaluation_criteria", ""),
        raw_text=raw,
    )


# ---------------------------------------------------------------------------
# AGENT 2 — GENERATOR
# ---------------------------------------------------------------------------
_GENERATOR_SYSTEM_TEMPLATE = textwrap.dedent("""\
    You are an expert quantitative portfolio constructor.
    You receive a detailed investment specification and (optionally) feedback
    from a previous QA round.  Your job is to produce a concrete portfolio
    allocation that meets ALL constraints.

    Think step-by-step:
    1. Reason about which instruments best satisfy the objective under the
       constraints.
    2. Use a mental model of mean-variance optimisation with a drawdown
       overlay.  Reference realistic historical statistics (you may
       approximate from memory — this is an exploration, not live trading).
    3. Stress-test your own proposal against the tail-risk scenarios in the
       spec BEFORE submitting.
    4. If you received evaluator feedback, address every point raised.

    RISK BUDGET DISCIPLINE — IMPORTANT:
    Treat the {MAX_LOSS} max annual loss as a budget to be USED, not avoided.
    A portfolio with a 2% expected_max_drawdown is wasting risk capacity
    and almost certainly leaving return on the table.  Aim your
    expected_max_drawdown as close to {MAX_LOSS} as you honestly can WITHOUT
    crossing it.  If prior feedback indicated you were over {MAX_LOSS}, your top
    priority this round is bringing the drawdown down — even at some cost
    to expected return.  If prior feedback indicated you were well under
    {MAX_LOSS}, raise the return profile by deploying more risk-bearing exposure
    until your drawdown is near (but under) {MAX_LOSS}.

    DIVERSIFICATION — IMPORTANT:
    The portfolio must be diversified across genuinely independent risk
    factors, not just across many tickers.  LLMs cannot reliably estimate
    pairwise correlations from memory, so DO NOT try.  Instead, use this
    explicit rule:

        From each overlap group below, choose AT MOST ONE ticker.

        US broad equity:           VOO, VTI, SPY, IVV, SPLG, ITOT, SCHB
        US large-cap factor tilts: QUAL, MTUM, VLUE, USMV, SPHQ, SPLV
        US dividend tilts:         SCHD, DGRO, VYM, HDV, NOBL, DVY
        US small-cap:              IWM, VB, IJR, SCHA, VTWO
        Int'l developed equity:    VEA, IEFA, VXUS, SCHF, IDEV
        Emerging-market equity:    VWO, IEMG, EEM, SCHE, SPEM
        Intermediate Treasuries:   IEF, GOVT, VGIT, SCHR
        Long Treasuries:           TLT, EDV, VGLT, SPTL
        Short Treasuries / cash:   SHV, SHY, BIL, SGOV, GBIL
        Investment-grade credit:   LQD, VCIT, VCSH, IGIB, IGSB
        High-yield credit:         HYG, JNK, USHY, SHYG
        US aggregate bonds:        AGG, BND, SCHZ, IUSB
        TIPS / inflation-linked:   TIP, SCHP, VTIP, STIP, LTPZ
        Municipal bonds:           MUB, VTEB, TFI, SUB
        Gold:                      GLD, IAU, GLDM, SGOL, BAR
        Broad commodities:         DBC, PDBC, GSG, BCI, COMT
        US REITs:                  VNQ, IYR, SCHH, XLRE, RWR
        Managed futures:           DBMF, KMLM, CTA, WTMF

    Picking VOO + QUAL + SCHD is NOT diversification — all three are
    ~0.85-0.95 correlated US large-cap equity.  Pick one.  Same for
    IEF + GOVT, TLT + EDV, LQD + VCIT, GLD + IAU, etc.

    Across groups, also be deliberate: do not combine instruments whose
    main risk factor is the same in different wrappers (e.g., HYG +
    high-equity-beta credit is mostly equity risk; LTPZ + EDV is mostly
    long-duration rate risk).  Spread across genuinely independent
    factors — equity, duration, credit, inflation-linked, gold,
    commodities, managed futures.

    If you must pick a ticker not on these lists, do so — but state
    explicitly in the rationale why it does not overlap with anything
    already in your allocation.


    Respond ONLY with a JSON object:
    {
      "allocations": {"TICKER_OR_ASSET": weight, ...},
      "descriptions": {"TICKER_OR_ASSET": "one-line plain-English description", ...},
      "expected_annual_return": float,
      "expected_max_drawdown": float,
      "methodology": "string",
      "rationale": "string"
    }

    The "descriptions" map must contain one entry for EVERY ticker in
    "allocations".  Each description should be a single concise sentence
    (≤ 15 words) that tells a non-expert what the instrument is, e.g.:
      "VOO": "Vanguard S&P 500 ETF — tracks the 500 largest US companies",
      "TLT": "iShares 20+ Year Treasury ETF — long-duration US government bonds",
      "SPX_PUT_SPREAD": "Protective put spread on the S&P 500 index — tail-risk hedge"
    Weights must sum to 1.0.  No markdown fences — raw JSON only.
""")


def generator_system(max_loss: float) -> str:
    """Render the Generator system prompt for a given max-loss budget."""
    return _GENERATOR_SYSTEM_TEMPLATE.replace("{MAX_LOSS}", f"{max_loss:.0%}")


def run_generator(
    spec: InvestmentSpec,
    feedback: str | None = None,
    iteration: int = 1,
    *,
    max_loss: float = 0.05,
) -> PortfolioProposal:
    print("\n" + "=" * 60)
    print(f"GENERATOR — building portfolio (iteration {iteration}) …")
    print("=" * 60)

    user_msg = f"INVESTMENT SPEC:\n{spec.raw_text}\n"
    if feedback:
        user_msg += f"\nEVALUATOR FEEDBACK FROM PREVIOUS ROUND:\n{feedback}\n"
        user_msg += "\nAddress every issue raised.  Revise the portfolio accordingly."

    raw = call_claude(generator_system(max_loss), user_msg)
    print(raw[:600], "…\n" if len(raw) > 600 else "\n")

    data = _parse_json_response(raw, agent="generator")

    return PortfolioProposal(
        allocations=data.get("allocations", {}),
        descriptions=data.get("descriptions", {}),
        expected_annual_return=data.get("expected_annual_return", 0),
        expected_max_drawdown=data.get("expected_max_drawdown", 0),
        methodology=data.get("methodology", ""),
        rationale=data.get("rationale", ""),
        raw_text=raw,
    )


# ---------------------------------------------------------------------------
# AGENT 3 — EVALUATOR
# ---------------------------------------------------------------------------
_EVALUATOR_SYSTEM_TEMPLATE = textwrap.dedent("""\
    You are a skeptical, rigorous portfolio risk analyst — the QA agent.
    Your job is to independently evaluate a proposed portfolio against an
    investment specification.  You must be TOUGH.  Do NOT give the benefit
    of the doubt.

    Grade each criterion on a 1-10 scale.  Be specific about failures.

    AVAILABLE TOOL — compute_backtest:
    You have a `compute_backtest(weights, start_date, end_date)` tool
    that returns the ACTUAL historical performance of any portfolio
    over any date range (total return, max drawdown, volatility,
    plus a list of any tickers that had no data in the range).
    USE IT.  Do not estimate 2008 / 2020 / 2022 losses from training
    memory — invoke the tool and use the returned numbers to score
    CONSTRAINT_COMPLIANCE.

    Suggested invocations for the standard stress windows:
      • 2008 GFC:        start='2008-01-01', end='2008-12-31'
      • 2020 COVID:      start='2020-02-01', end='2020-12-31'
      • 2022 rate-hike:  start='2022-01-01', end='2022-12-31'

    Pay attention to `tickers_missing_data` and `coverage_weight` in
    the result — if a meaningful weight of the portfolio wasn't around
    in the date range (e.g., DBMF didn't exist in 2008), reason about
    whether a comparable substitute would have changed the outcome,
    and call out the partial coverage in your critique.

    The tool returns the PRE-MECHANISM gross loss.  Apply the spec's
    enforcement mechanism on top of that (see CRITERION 1 below) to
    get the post-mechanism loss you judge against the {MAX_LOSS} cap.

    CRITERIA (graded 1-10):
    1. CONSTRAINT COMPLIANCE — Does the portfolio stay within the ≤{MAX_LOSS}
       max annual loss constraint under realistic historical scenarios?
       Check: 2008 GFC, 2020 COVID crash, 2022 rate-hike drawdown.

       IMPORTANT — RESPECT THE SPEC'S ENFORCEMENT MECHANISM:
       If the spec's `constraints` define an `enforcement_mechanism` for
       the loss cap (e.g., a dynamic de-risking trigger, an options
       hedge overlay, a rebalancing rule), MODEL that mechanism when
       computing each scenario's loss.  Apply the trigger logic (or
       hedge payoff) to the gross drawdown, then judge the
       POST-mechanism net annual loss against the {MAX_LOSS} cap.

       A pre-mechanism gross loss above {MAX_LOSS} is acceptable IF the
       mechanism plausibly contains the post-mechanism annual loss to
       ≤ {MAX_LOSS}.  The spec defined the mechanism; honour it.

       Score ≤ 4 only when either:
         • The post-mechanism net annual loss STILL breaches {MAX_LOSS} in a
           realistic scenario — i.e., even when the mechanism works
           as designed, the portfolio would have lost more than {MAX_LOSS} in
           2008 / 2020 / 2022, OR
         • The mechanism is implausible for the scenario (e.g., a
           slow drawdown-trigger cannot catch a gap-down event large
           enough to blow through it in a single session; an options
           overlay is sized too small to offset the expected loss;
           the trigger requires liquidity that wouldn't exist in the
           scenario being modelled).

       ALSO score ≤ 6 if the expected_max_drawdown is materially BELOW
       {MAX_LOSS} (e.g., well under it) without a specific constraint forcing that
       level of conservatism — under-utilising the {MAX_LOSS} risk budget is a
       flaw, not a virtue, because it sacrifices return for no good
       reason.  A portfolio that lands near (but under) {MAX_LOSS} should score
       highest on this criterion; one that lands far below it should
       be marked down for wasting risk capacity.

    2. RETURN POTENTIAL — Is the expected return realistic and competitive
       given the constraints?  Penalise both overestimation and unnecessary
       conservatism.

    3. DIVERSIFICATION — Are risks well-spread across uncorrelated sources?
       Penalise concentration and hidden correlations (e.g., all equity-like).
       Penalize the agent for proposing multiple highly correlated assets.

    4. IMPLEMENTABILITY — Can a US retail investor actually buy these
       instruments easily and cheaply?  Penalise illiquids, high-fee
       products, or instruments requiring institutional access. Penalize
       proposing strategies with a total number of tickers > 15.

    5. METHODOLOGY RIGOUR — Is the construction approach sound, or does it
       hand-wave?  Are the return / risk estimates grounded in data?

    Respond ONLY with a JSON object:
    {
      "scores": {
        "constraint_compliance": int,
        "return_potential": int,
        "diversification": int,
        "implementability": int,
        "methodology_rigour": int
      },
      "critique": "Detailed critique addressing each criterion …",
      "passed": true/false
    }

    THE "passed" FIELD IS YOUR INDEPENDENT JUDGEMENT (not a derived rule).
    Return passed=true ONLY if you would unconditionally recommend this
    portfolio to a real client TODAY.  Return passed=false — even if the
    average score is ≥ 7 — when you found ANY of:
      • A binding hard-constraint violation (FX hedging, ticker cap,
        sector cap, leverage, instrument restrictions, etc.).
      • A realistic historical scenario where the POST-mechanism annual
        loss still breaches the {MAX_LOSS} cap — i.e., even crediting the
        spec's enforcement mechanism as designed, the portfolio would
        have lost more than {MAX_LOSS} in 2008 / 2020 / 2022.  A pre-mechanism
        gross loss above {MAX_LOSS} is NOT a fail on its own when the mechanism
        plausibly contains the post-mechanism loss.
      • Material methodology gaps that make the loss estimates
        unreliable (e.g., key scenarios not stress-tested at all,
        return / risk numbers invented out of thin air, instruments
        outside the spec's `asset_universe`).
    Your "passed" value will be combined with the numeric pass rule
    (average ≥ 7 AND no single score ≤ 4) — all three must hold to pass.

    CRITIQUE FLAGS RISKS EVEN WHEN PASSING — IMPORTANT:
    The pass/fail judgement and the critique serve different purposes.
    Pass on the merits of the spec as written; flag the residual risks
    of the mechanism the spec relies on SEPARATELY in your critique so
    the human reader can see them.  Even when you return passed=true,
    your critique MUST surface the failure modes of the enforcement
    mechanism, including (when applicable):
      • Trigger slippage in fast-moving markets
      • Whipsaw risk — trigger de-risks then misses the rebound
      • Gap-down risk — a single session that overshoots the trigger
        threshold before any de-risking can execute
      • Basis risk in hedge overlays (hedge moves differently than
        the asset it's meant to protect)
      • Liquidity / execution risk during a stress event
      • Reliance on a single mechanism as the sole tail backstop
    Mention which scenario each risk would matter in.  Do NOT lower the
    pass judgement on the basis of these risks alone if the mechanism
    plausibly works as designed — flag them and pass.

    No markdown fences — raw JSON only.
""")


def evaluator_system(max_loss: float) -> str:
    """Render the Evaluator system prompt for a given max-loss budget."""
    return _EVALUATOR_SYSTEM_TEMPLATE.replace("{MAX_LOSS}", f"{max_loss:.0%}")


def run_evaluator(
    spec: InvestmentSpec,
    proposal: PortfolioProposal,
    *,
    pass_threshold: int = 7,
    max_loss: float = 0.05,
    growth_ceiling: float | None = None,
) -> EvaluationResult:
    """
    Score the proposal on five criteria and produce a critique.  The
    composite ``passed`` flag is True only when (a) the average score is
    >= ``pass_threshold``, (b) no single score is <= 4, AND (c) the
    Evaluator's own ``passed`` judgement in the response is True — all
    three must hold.  ``pass_threshold`` and ``max_loss`` are the
    orchestrator's policy knobs (live in ``harness.py``) and are passed
    in so this module stays orchestrator-agnostic.

    When ``growth_ceiling`` is not None (the horizon glide-path ceiling),
    an additional DETERMINISTIC, MECHANICAL gate runs on top of the LLM
    judgement: the portfolio's growth-asset weight is measured via the
    shared classifier (``harness.growth_asset_weight``) and, if it EXCEEDS
    the ceiling, ``passed`` is forced False and a specific critique line is
    appended.  This gate can only turn a pass into a fail, never the
    reverse — it cannot rescue an LLM-failed proposal.
    """
    print("\n" + "=" * 60)
    print("EVALUATOR — stress-testing the portfolio …")
    print("=" * 60)

    user_msg = (
        f"INVESTMENT SPEC:\n{spec.raw_text}\n\n"
        f"PROPOSED PORTFOLIO:\n{proposal.raw_text}\n"
    )

    raw = call_claude(
        evaluator_system(max_loss), user_msg,
        tools=_BACKTEST_TOOLS,
        tool_handlers=_BACKTEST_HANDLERS,
        # Long stress narration + JSON scores/critique (now incl. the growth
        # ceiling) truncate at the 4096 default; give it Refiner-level room.
        max_tokens=api.EVALUATOR_MAX_TOKENS,
        # 3 standard stress windows + headroom for sub-scenarios + final
        # text emission; well clear of the default 8 but explicit here so
        # the cap is documented at the call site.
        max_tool_rounds=10,
    )
    print(raw[:600], "…\n" if len(raw) > 600 else "\n")

    data = _parse_json_response(raw, agent="evaluator")

    scores = data.get("scores", {})
    avg = sum(scores.values()) / len(scores) if scores else 0
    any_critical_fail = any(v <= 4 for v in scores.values())
    # Honor the evaluator's own pass/fail judgement.  The model often spots
    # binding-constraint violations (e.g. FX hedging, ticker count) that the
    # numeric average + min-score rule alone would let through.  Missing
    # field defaults to False — we want an explicit "yes" from the model.
    model_said_passed = bool(data.get("passed", False))
    passed = (
        avg >= pass_threshold
        and not any_critical_fail
        and model_said_passed
    )

    critique = data.get("critique", "")

    # --- Deterministic growth-ceiling gate (horizon glide path) ---------
    # A hard, mechanical check ON TOP of the LLM judgement.  Lazy import of
    # the shared classifier from harness avoids a circular import at module
    # load (harness imports this module at its top).  This can only turn a
    # pass into a fail — never the reverse.
    if growth_ceiling is not None:
        from harness import growth_asset_weight  # lazy — avoids import cycle
        measured = growth_asset_weight(proposal)
        if measured > growth_ceiling + 1e-9:
            passed = False
            ceiling_note = (
                f"GROWTH-CEILING VIOLATION (deterministic check): measured "
                f"growth-asset weight {measured:.1%} EXCEEDS the horizon "
                f"ceiling of {growth_ceiling:.0%}. Growth assets (equity, "
                f"REITs, commodities, gold, managed futures, high-yield "
                f"credit) must be reduced to at or below {growth_ceiling:.0%}, "
                f"with the remainder in high-grade bonds / TIPS / cash. "
                f"Forced QA FAIL regardless of scores."
            )
            critique = (critique + "\n\n" + ceiling_note) if critique else ceiling_note

    return EvaluationResult(
        passed=passed,
        scores=scores,
        average_score=round(avg, 2),
        critique=critique,
        raw_text=raw,
    )


# ---------------------------------------------------------------------------
# AGENT 4 — REFINER  (post-selection fine-tuner)
# ---------------------------------------------------------------------------
_REFINER_SYSTEM_TEMPLATE = textwrap.dedent("""\
    You are a senior portfolio fine-tuner.  Your input is:
      • An investment specification.
      • A portfolio that has already been selected as the best of several
        candidates.
      • A detailed critique from the QA evaluator pointing out specific
        issues with that portfolio.

    Your job is to produce a REVISED portfolio that addresses every issue
    raised in the critique, while preserving everything the critique did
    NOT flag.  This is a SURGICAL EDIT, not a rewrite.

    AVAILABLE TOOL — compute_backtest:
    You have a `compute_backtest(weights, start_date, end_date)` tool
    that returns actual historical performance.  Use it to VERIFY your
    revised portfolio before emitting it — at minimum on 2008
    (start='2008-01-01', end='2008-12-31'), 2020 (start='2020-02-01',
    end='2020-12-31'), and 2022 (start='2022-01-01', end='2022-12-31').
    If a backtest shows the revision still breaches the {MAX_LOSS} cap (after
    accounting for the spec's enforcement_mechanism), iterate further
    before responding — don't ship weights you haven't verified.

    Hard rules:
    1. Maintain the ≤{MAX_LOSS} max annual loss constraint under realistic
       historical scenarios (2008, 2020, 2022).  Verify with the tool.
    2. Aim expected_max_drawdown close to (but under) {MAX_LOSS} — do NOT waste
       the risk budget by becoming overly conservative.
    3. Address EVERY distinct issue in the critique.  If the critique
       lists three separate problems, your revision must visibly address
       all three.
    4. Preserve unflagged characteristics of the original portfolio:
       keep instruments and weights that the critique did not call out,
       unless removing them is necessary to fix something that WAS
       flagged.

    Use the rationale field to explain, point by point, how each
    critique item was addressed.  Be concrete: "Issue X — fixed by Y".

    Respond ONLY with a JSON object using the same schema as the Generator:
    {
      "allocations": {"TICKER_OR_ASSET": weight, ...},
      "descriptions": {"TICKER_OR_ASSET": "one-line plain-English description", ...},
      "expected_annual_return": float,
      "expected_max_drawdown": float,
      "methodology": "string",
      "rationale": "string — must address each critique point explicitly"
    }

    The "descriptions" map must contain one entry per ticker.
    Weights must sum to 1.0.  No markdown fences — raw JSON only.
""")


def refiner_system(max_loss: float) -> str:
    """Render the Refiner system prompt for a given max-loss budget."""
    return _REFINER_SYSTEM_TEMPLATE.replace("{MAX_LOSS}", f"{max_loss:.0%}")


def run_refiner(
    spec: InvestmentSpec,
    selected_proposal: PortfolioProposal,
    selected_evaluation: EvaluationResult,
    *,
    max_iterations: int = 3,
    max_loss: float = 0.05,
) -> PortfolioProposal:
    """
    Take the harness-selected portfolio plus the evaluator's critique and
    produce a single revised proposal that addresses every critique item.

    ``max_iterations`` is included in the Refiner's user prompt as
    flavor text ("currently best of N candidates"); it and ``max_loss``
    are orchestrator policy knobs (live in ``harness.py``) and are passed
    in so this module stays orchestrator-agnostic.
    """
    print("\n" + "=" * 60)
    print("REFINER — fine-tuning the selected portfolio against critique …")
    print("=" * 60)

    user_msg = (
        f"INVESTMENT SPEC:\n{spec.raw_text}\n\n"
        f"SELECTED PORTFOLIO (currently best of {max_iterations}):\n"
        f"{selected_proposal.raw_text}\n\n"
        f"EVALUATOR SCORES: {selected_evaluation.scores} "
        f"(average {selected_evaluation.average_score})\n"
        f"EVALUATOR PASS JUDGEMENT: {selected_evaluation.passed}\n\n"
        f"DETAILED CRITIQUE TO ADDRESS:\n{selected_evaluation.critique}\n\n"
        f"Produce a revised portfolio that fixes every issue above while "
        f"preserving the rest.  Spell out in `rationale` how each critique "
        f"point was addressed."
    )

    raw = call_claude(
        refiner_system(max_loss), user_msg,
        max_tokens=api.REFINER_MAX_TOKENS,
        tools=_BACKTEST_TOOLS,
        tool_handlers=_BACKTEST_HANDLERS,
        # Refiner may iterate: backtest → see breach → adjust weights →
        # backtest again.  Give it room (3 windows × 2-3 attempts +
        # final emission).
        max_tool_rounds=12,
    )
    print(raw[:600], "…\n" if len(raw) > 600 else "\n")

    data = _parse_json_response(raw, agent="refiner")

    return PortfolioProposal(
        allocations=data.get("allocations", {}),
        descriptions=data.get("descriptions", {}),
        expected_annual_return=data.get("expected_annual_return", 0),
        expected_max_drawdown=data.get("expected_max_drawdown", 0),
        methodology=data.get("methodology", ""),
        rationale=data.get("rationale", ""),
        raw_text=raw,
    )


# ---------------------------------------------------------------------------
# AGENT 5 — ADVISOR  (advisory only, never modifies the portfolio)
# ---------------------------------------------------------------------------
ADVISOR_SYSTEM = textwrap.dedent("""\
    You are a portfolio diversification advisor.  You are NOT allowed to
    change the portfolio.  Your single job is to look at the FINAL
    portfolio (which has already passed QA or been chosen as the best
    effort) and surface two things for the human reader:

    1. A pairwise CORRELATION SNAPSHOT for tickers that move similarly.
       For every PAIR of holdings whose long-run correlation is |ρ| ≥ 0.5,
       output an entry.  Use realistic historical correlations (you may
       approximate from memory; this is a snapshot, not a backtest).
       Be honest about uncertainty — these are model-recalled, not
       computed from data.

    2. Concrete SIMPLIFICATION SUGGESTIONS.  For each cluster of highly
       correlated holdings (typically ρ ≥ 0.75) that look redundant,
       propose a specific consolidation.

    HARD RULES for suggestions:
    • Suggest a REAL replacement ticker (e.g., GOVT, AGG, VXUS, BNDX)
      that a US retail investor can buy easily.  Do not invent tickers.
    • Be EXPLICIT about what the user gives up — every suggestion MUST
      include a "tradeoff" string.  Examples of legitimate tradeoffs:
        "Loses the explicit short/intermediate Treasury barbell."
        "Loses tax-exempt muni income exposure."
        "Combines investment-grade credit with Treasuries — credit risk
         becomes implicit rather than sized separately."
    • Do NOT suggest consolidations across genuinely different risk
      factors (e.g., merging LQD into IEF collapses credit and rate
      exposure — flag it as a NOT-recommended merge if you mention it).
    • If the portfolio is already well-consolidated, return an empty
      "suggestions" list rather than inventing weak ideas.

    Respond ONLY with a JSON object:
    {
      "correlation_pairs": [
        {"a": "TICKER_A", "b": "TICKER_B", "rho": 0.85,
         "note": "optional short context"},
        ...
      ],
      "suggestions": [
        {
          "merge_from": ["TICKER_X", "TICKER_Y"],
          "merge_into": "REPLACEMENT_TICKER",
          "rationale": "one-sentence why they overlap",
          "tradeoff": "explicit description of what is lost"
        },
        ...
      ],
      "notes": "optional caveats about correlation estimates / regime sensitivity"
    }

    No markdown fences — raw JSON only.
""")


def run_advisor(final_proposal: PortfolioProposal) -> AdvisorOutput:
    """
    Inspect a portfolio and produce advisory consolidation suggestions
    plus a pairwise correlation snapshot.  Never modifies the portfolio
    itself.  Used in two places by the orchestrator — between iterations
    (its findings feed the next Generator round's prompt) and after the
    final portfolio is determined (read-only, for the report).
    """
    print("\n" + "=" * 60)
    print("ADVISOR — scanning final portfolio for correlation / simplification …")
    print("=" * 60)

    # Build a compact view of the holdings (ticker, weight, description)
    rows = []
    descs = final_proposal.descriptions or {}
    for ticker, weight in final_proposal.allocations.items():
        desc = descs.get(ticker, "")
        rows.append(f"  {ticker:30s}  {float(weight):.2%}   {desc}")
    holdings_block = "\n".join(rows) if rows else "  (empty)"

    user_msg = (
        f"FINAL PORTFOLIO (do NOT modify — advise only):\n{holdings_block}\n\n"
        f"Produce the correlation snapshot and simplification suggestions "
        f"per the system prompt schema.  Focus on pairs that move together "
        f"in normal regimes; flag (in notes) any pairs whose correlation "
        f"changes materially in stress."
    )

    raw = call_claude(ADVISOR_SYSTEM, user_msg, model=api.ADVISOR_MODEL)
    print(raw[:500], "…\n" if len(raw) > 500 else "\n")

    data = _parse_json_response(raw, agent="advisor")

    return AdvisorOutput(
        suggestions=data.get("suggestions", []) or [],
        correlation_pairs=data.get("correlation_pairs", []) or [],
        notes=data.get("notes", "") or "",
        raw_text=raw,
    )
