# Portfolio Optimization Harness

A multi-agent implementation inspired by Anthropic's [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps), applied to financial portfolio optimisation.

The harness started as the original three-agent pattern (Planner → Generator → Evaluator). It has since grown to five agents with explicit selection, refinement, and advisory steps to address the gap between "passes QA" and "ready to actually use."

## Architecture

```
                            ┌──────────┐
                            │ PLANNER  │   (Sonnet)
                            └─────┬────┘
                                  │ Investment Spec
                                  ▼
        ┌──────────────────────────────────────────────────────┐
        │  for i in 1..MAX_ITERATIONS (default 3)              │
        │                                                      │
        │  ┌───────────┐                                       │
        │  │ GENERATOR │ ◀──── critique + advisor pairs        │
        │  │  (Opus)   │       (concrete tickers to consolidate)│
        │  └─────┬─────┘                                       │
        │        │ Portfolio                                   │
        │        ▼                                             │
        │  ┌───────────┐                                       │
        │  │ EVALUATOR │ ── scores + passed flag               │
        │  │  (Opus)   │                                       │
        │  └─────┬─────┘                                       │
        │        │                                             │
        │        ▼                                             │
        │  ┌───────────┐                                       │
        │  │  ADVISOR  │ ── correlation pairs ≥ 0.7            │
        │  │  (Haiku,  │    (skipped on last iter — no         │
        │  │  intra)   │     next round to feed)               │
        │  └─────┬─────┘                                       │
        │        │                                             │
        │        ▼                                             │
        │  history[i] = {alloc, scores, passed, …}             │
        └──────────────────────────────────────────────────────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │   SELECTOR   │   best passing iteration
                          │ (no LLM call)│   (closest |drawdown − 5%|);
                          │              │   falls back to iter 1 if
                          │              │   no iteration passed
                          └──────┬───────┘
                                 │ selected_proposal
                                 ▼
                          ┌──────────────┐
                          │   REFINER    │   surgically address every
                          │   (Opus)     │   critique point
                          └──────┬───────┘
                                 │ refined_proposal
                                 ▼
                          ┌──────────────┐
                          │  EVALUATOR   │   re-score the refined version
                          │  (Opus,      │
                          │   re-run)    │
                          └──────┬───────┘
                                 │
                                 ▼
                       passes QA AND
                       drawdown ≤ 5%?
                       ┌─────┴──────┐
                      yes           no
                       │             │
              promote refined   keep selected
                       └──────┬──────┘
                              │ final_proposal
                              ▼
                       ┌──────────────┐
                       │   ADVISOR    │   surface correlated holdings +
                       │  (Haiku,     │   structured merge suggestions
                       │  read-only)  │   (for the report)
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   PRICING    │   yfinance quotes + whole-share
                       │ (no LLM)     │   lot-size feasibility check
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ RISK PROFILE │   Monte-Carlo return distribution
                       │ (no LLM)     │   (block-bootstrap, 2008-inclusive
                       │              │   via long-history proxies)
                       └──────┬───────┘
                              │
                              ▼
                       harness_output.{json,md}
```

### Agent roles

| Agent | Role |
|---|---|
| **Planner** | Expands a brief user goal ("maximize return, ≤5% annual loss") into a detailed investment specification: asset universe, constraints, risk budget, evaluation criteria, tail-risk scenarios. |
| **Generator** | Constructs a concrete portfolio allocation that satisfies the spec. Now returns plain-English `descriptions` for each ticker alongside `allocations`. On later iterations, addresses every point from the evaluator's critique. |
| **Evaluator** | Independently stress-tests the portfolio against five 1–10 criteria (constraint compliance, return potential, diversification, implementability, methodology rigour) and writes a detailed critique. **Passes only if** `avg ≥ 7` AND `no single score ≤ 4` AND the evaluator's own `passed: true/false` judgement agrees — all three must hold. |
| **Selector** (no LLM call) | After all `MAX_ITERATIONS` rounds finish, picks the passing iteration whose `expected_max_drawdown` is closest to the loss target (`TARGET_MAX_LOSS`, default 5%). Tiebreak: smaller drawdown, then higher score. **Falls back to iteration 1** if no iteration passed — the critique-feedback loop tends to push later iterations toward increasingly conservative portfolios (lower returns at similar drawdown), so when nothing passes, the unbiased first attempt is usually the most balanced. |
| **Refiner** | Takes the selected portfolio + the evaluator's critique and produces a surgical revision that addresses every flagged issue while preserving what worked. Re-evaluated; **promoted to final only if it passes QA and stays within the loss target**, otherwise the selected version is kept. |
| **Advisor** (read-only) | Plays **two roles**, both read-only with respect to the portfolio. **(1) Per-iteration:** runs after each Generator/Evaluator pair (except the last) and emits correlation pairs at \|ρ\| ≥ 0.7; these pairs are fed into the next round's Generator feedback as concrete tickers to consolidate. **(2) Final pass:** runs once on the final portfolio to produce the report's pairwise correlation snapshot plus structured `{merge_from, merge_into, rationale, tradeoff}` consolidation suggestions — every suggestion has an explicit tradeoff so the human reader can decide whether to apply it. |
| **Pricing** (no LLM call) | Fetches the latest price for each ticker via [`yfinance`](https://github.com/ranaroussi/yfinance) and computes a whole-share lot-size feasibility check against `--capital` USD (default $100k). Per-ticker failures (unknown ticker, network blip, model-invented pseudo-ticker like `SPX_PUT_SPREAD`) degrade gracefully. Output includes a Yahoo Finance data-source disclaimer. |
| **Risk Profile** (no LLM call) | Replaces the single `expected_max_drawdown` point estimate with a **return distribution**. Block-bootstraps historical daily returns into many multi-year paths and reports, per holding horizon (1/3/5/10y), the **median outcome**, the **chance of ending down**, and the **1-in-20 / 1-in-100 unlucky tails**. Substitutes long-history asset-class proxies for young ETFs so the sample spans the 2008 crisis; drops non-priceable legs (option overlays) and renormalizes, so the modeled downside is conservative. yfinance-backed and fail-soft. |

### Key design decisions

1. **Separation of generation and evaluation.** A generator asked to grade its own work will praise it, so the Evaluator is a separate agent with a "skeptical, rigorous" prompt that scores five 1–10 criteria and independently judges pass/fail. The Evaluator honours the spec **as written** — when the Planner defines an `enforcement_mechanism` for the loss cap (e.g., a dynamic de-risking trigger), the Evaluator models that mechanism when stress-testing and judges the post-mechanism loss against the cap, rather than vetoing on the pre-mechanism gross drawdown. Pass/fail and the critique are orthogonal: portfolios that meet the spec's stated criteria pass even when residual mechanism risks (slippage, gap-down, single point of failure) exist, but those risks are still surfaced in the critique for the human reader.

2. **Always run all iterations, then select.** The original harness exited the loop as soon as one iteration passed. That gave away later iterations that could have landed closer to the risk-budget target (`TARGET_MAX_LOSS`). Now the loop always runs `MAX_ITERATIONS` rounds; after the loop, the Selector picks the best passing iteration by closeness to the target.

3. **Iteration feedback is regime-aware AND structurally informed.** Each round's feedback combines two layers: (a) regime-aware guidance from the Evaluator — when an iteration passes but its drawdown is over target (`TARGET_MAX_LOSS`) the Generator is told to push it down; when drawdown is well under target (≤ `UNDER_UTILISATION_BAND`) the Generator is told the risk budget is being wasted; failed iterations pass the critique through verbatim. (b) Concrete correlation pairs from the per-iteration Advisor, filtered to \|ρ\| ≥ `ADVISOR_FEEDBACK_RHO_THRESHOLD` (default 0.7), prepended to the feedback as a "ticker A ↔ ticker B (ρ ≈ 0.95) — collapse" block. Pass and failure paths both get the Advisor section.

4. **Refinement applies; advice influences but never overwrites.** The Refiner is the only agent that can replace the selected portfolio's holdings (and only if its output passes QA and stays within the loss target). The Advisor never edits the portfolio directly — but its per-iteration correlation findings DO feed the next Generator round, converting "avoid correlation" from an unactionable LLM rule into ticker-to-ticker guidance the model can actually follow. The final Advisor pass for the report stays purely read-only.

5. **Per-agent model tier.** Not every agent needs the most capable model. Generator / Evaluator / Refiner stay on Opus (the real reasoning work). Planner runs on Sonnet (recall + JSON structuring — doesn't need Opus). Advisor runs on Haiku (pattern-matching against training memory for known ticker correlations — Haiku is fine and the intra-loop placement makes its speed valuable). Saves ~17% of default-run cost with no observable quality impact on the agents that drive the outcome. `--model X` overrides all three to X — useful when one tier is overloaded or for direct cost/quality comparison.

6. **Evaluator honours the spec as written.** The Planner can define an `enforcement_mechanism` for the loss cap (e.g., a dynamic de-risking trigger, an options hedge overlay). When it does, the Evaluator models that mechanism when stress-testing and judges the **post-mechanism** annual loss against the cap — not the pre-mechanism gross drawdown. Pass/fail and the critique are orthogonal: portfolios that meet the spec's stated criteria pass, but the critique still surfaces residual mechanism risks (slippage, gap-down, single point of failure) so the human reader sees them. Without this, every aggressive portfolio gets vetoed for "breaching" a cap the spec's own mechanism was supposed to enforce.

7. **Resilience to transient API failures.** Each `call_claude` is wrapped in exponential-backoff retry on 429/5xx/529/connection/timeout errors (5 SDK retries + 6 outer attempts with 2/4/8/16/32s jitter, ≈62s max wall-clock). Auth/validation errors fast-fail. Refiner gets `REFINER_MAX_TOKENS = 8192` instead of the default 4096 (it emits both a full portfolio and a per-critique rationale; 4096 was truncating mid-string). All five `run_*` agents share a single fail-soft `_parse_json_response` helper — on unparseable model output, it returns `{}` and the pipeline degrades through dataclass defaults rather than crashing the whole run.

8. **The loss budget is one knob, not five.** The max annual loss (`TARGET_MAX_LOSS`, default 5%) is the single value that defines the risk cap, and it feeds three things at once: the Selector's target, the regime-aware feedback thresholds, and — crucially — the Generator / Evaluator / Refiner system prompts. The prompts are **templated** (a `{MAX_LOSS}` token rendered per run via `generator_system()` / `evaluator_system()` / `refiner_system()`) rather than carrying a hardcoded "5%". This matters because the prompts are the authoritative source of the constraint the models actually see: before templating, the "5%" lived as a literal string in three places the goal text never touched, so changing the goal alone did **not** change what the agents optimised for. Override the budget for a whole run with `--max-loss 0.10` (CLI) or the `max_loss` request field (server); both patch the global the same way `--iterations` patches `MAX_ITERATIONS`. The "wasting risk capacity" floor (`UNDER_UTILISATION_BAND`) **auto-derives** as 80% of the budget (`UNDER_UTILISATION_RATIO`), so it tracks the cap automatically. The agents stay orchestrator-agnostic — they receive `max_loss` as a parameter, mirroring how `pass_threshold` is already passed in.

9. **Distribution over point estimate.** `expected_max_drawdown` is a single number the *model* asserts; it answers neither "how likely is my upside?" nor "how bad is the downside, really?". The Risk Profile step (`risk.py`) replaces it with an empirical **return distribution**: it block-bootstraps real historical daily returns (6-month blocks, preserving volatility clustering and strings of bad quarters) into thousands of multi-year paths and reports, per horizon, the median outcome, the probability of ending underwater, and the unlucky tails. Three honesty choices drive the design: (a) **long-history asset-class proxies** (`SGOV→SHV`, `SCHP→TIP`, `MUB→VWITX`, …) substitute for young ETFs so the sample reaches back through 2008 — a benign post-2014 sample badly understates tail risk; (b) **non-priceable legs are dropped and renormalized** (option overlays can't be bootstrapped from a price series), which makes the modeled downside *conservative* relative to the hedged book — stated explicitly rather than hidden; (c) every substitution, the achieved sample window, and the coverage fraction are **reported in the output** so the reader can judge how much to trust the tail. Like Pricing, it's a no-LLM, yfinance-backed, fail-soft post-processing step; skip with `--no-risk`.

## Running

This project is a [uv](https://docs.astral.sh/uv/) workspace member of the parent `testbench` project. All dependencies are managed through uv — no manual `pip install`.

### First-time setup

From the repository root (`testbench/`):

```bash
uv sync
```

### Set the API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Default run

Per-agent models (Sonnet planner / Opus generator+evaluator+refiner / Haiku advisor), 3 iterations, refinement on, advisor on (both in-loop and final pass), pricing on:

```bash
uv run python harness.py
```

This makes ~12 API calls per run:

| Calls | Agent | Model |
|--:|---|---|
| 1 | Planner | Sonnet |
| 3 | Generator (one per iteration) | Opus |
| 3 | Evaluator (one per iteration) | Opus |
| 2 | Advisor (intra-loop, skipped on last iter) | Haiku |
| 1 | Refiner | Opus |
| 1 | Evaluator (re-run on refined) | Opus |
| 1 | Advisor (final, for the report) | Haiku |

Plus one yfinance batch (no API key needed; per-ticker failures degrade gracefully). Takes a few minutes against Opus.

### CLI flags

| Flag | Effect |
|---|---|
| `--test` | Smoke-test mode: Haiku 4.5 for ALL agents, 1 iteration, no refinement, no advisor, no pricing. ~3 API calls, cheapest end-to-end verification of the plumbing. Useful when Opus is overloaded or you just want to see the flow run. |
| `--model {haiku\|sonnet\|opus\|<full-id>}` | Override the model **for all agents** (planner / generator / evaluator / refiner / advisor). Aliases resolve to `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-7`. Any other string is passed through verbatim as a model ID. |
| `--iterations N` | Override `MAX_ITERATIONS` for this run (default 3). |
| `--max-loss FRACTION` | Override the max annual loss budget (`TARGET_MAX_LOSS`) as a fraction in `(0, 1)` — e.g. `0.10` for 10% (default 5%). Drives the Generator / Evaluator / Refiner prompts, the selection target, and the auto-derived under-utilisation band. **Unlike the other flags, this applies in `--test` mode too.** |
| `--no-refine` | Skip the post-selection Refiner pass. |
| `--no-advisor` | Skip the Advisor entirely — both the per-iteration feedback role and the final read-only pass for the report. |
| `--no-prices` | Skip the post-selection Pricing pass (yfinance lookups + lot-size feasibility). Pricing is on by default. |
| `--no-risk` | Skip the post-selection Risk-profile pass (Monte-Carlo return distribution). On by default; needs yfinance (extra historical-data pulls), so `--no-risk` is the way to run fully offline alongside `--no-prices`. |
| `--capital USD` | Capital assumed for the whole-share lot-size feasibility check (default $100,000). Only used when pricing is enabled. |

`--test` takes precedence — if combined with `--model` / `--iterations` / `--no-refine` / `--no-advisor` / `--no-prices` / `--no-risk` / `--capital`, the test-mode defaults win (test mode implies refine/advisor/pricing/risk all off). The one exception is `--max-loss`, which is orthogonal and applies even under `--test`.

### Examples

```bash
# Full default run (per-agent models, refinement on, advisor in-loop + final, pricing on)
uv run python harness.py

# Quick smoke test — Haiku for everything, 1 iteration, no refiner/advisor/pricing
uv run python harness.py --test

# Opus is overloaded? Full 3-iteration run on Sonnet (overrides per-agent split)
uv run python harness.py --model sonnet

# Run with refiner but skip the advisor entirely (also disables advisor-in-loop feedback)
uv run python harness.py --no-advisor

# Skip yfinance price-fetching (e.g., offline or rate-limited)
uv run python harness.py --no-prices

# Fully offline — skip both yfinance passes (pricing + risk distribution)
uv run python harness.py --no-prices --no-risk

# Lot-size feasibility for a $250k portfolio instead of the default $100k
uv run python harness.py --capital 250000

# Optimise for a 10% annual-loss budget instead of the default 5%
# (under-utilisation band auto-scales to 8%)
uv run python harness.py --max-loss 0.10

# Smallest meaningful real run — Haiku, 2 iterations, with refinement
uv run python harness.py --model haiku --iterations 2
```

## Output

Each run produces two files in the directory it was run from:

- **`harness_output.json`** — machine-readable trace: run config (`model`, `max_iterations`, `pass_threshold`, `target_max_loss`), spec, every iteration's allocations / scores / intra-advisor pair count, selected proposal, refinement block (before/after), advisor block (suggestions + correlations), pricing block (per-ticker prices + lot sizes + leftover cash), risk-profile block (per-horizon return distribution + sample window + proxy substitutions + coverage), and raw model responses. The trace is **self-describing** — an old `harness_output.json` can be re-rendered through `report.py` without rerunning the pipeline.
- **`harness_output.md`** — human-readable Markdown report. Renders cleanly in VS Code's built-in preview (`Cmd+Shift+V`) or any Markdown viewer. Contents:
  - Header summary (model, iterations, selected iteration, refinement / advisor / pricing status, target loss, pass rule)
  - **Final Portfolio** table with `Ticker | Weight | Description` columns
  - Iteration Summary table comparing all iterations on score, return, drawdown, and distance to target — selected iteration starred
  - Investment Spec from the Planner
  - Selected Portfolio Methodology + Rationale
  - Selected Portfolio Evaluator Scores + Critique (the critique surfaces residual mechanism risks even when the portfolio passed — slippage, gap-down, single point of failure, etc.)
  - **Post-Selection Refinement** section with Score deltas, Portfolio metric deltas, and Allocation changes tables, plus the Refiner's point-by-point rationale and the re-evaluator's report
  - **Simplification Suggestions** (final Advisor) — `{merge_from} → {merge_into}` items with explicit tradeoffs, plus a pairwise correlation table sorted by strongest |ρ|
  - **Latest Prices & Lot-Size Feasibility** — per-ticker prices, target $ vs. actual whole-share $, weight drift, leftover cash, plus a Yahoo Finance data-source disclaimer
  - **Return Distribution (Monte-Carlo)** — per-horizon (1/3/5/10y) table of median outcome, chance of ending down, and 1-in-20 / 1-in-100 unlucky tails; plus the sample window (whether it spans 2008), coverage, and the long-history proxy substitutions used
  - Per-iteration detail — including `Advisor pairs fed to iteration N` line showing the intra-loop feedback signal
  - Planner / Generator / Evaluator / Refiner / Advisor raw responses in collapsible `<details>` blocks

## Configuration

Edit the constants at the top of `harness.py`:

| Constant | Default | Meaning |
|---|---|---|
| `MODEL` | `claude-opus-4-7` | Default model for Generator / Evaluator / Refiner (override via `--model`, which patches all three per-agent globals) |
| `PLANNER_MODEL` | `claude-sonnet-4-6` | Model for the Planner (recall + JSON structuring) |
| `ADVISOR_MODEL` | `claude-haiku-4-5-20251001` | Model for the Advisor (pattern-matching against memory for correlations) |
| `MAX_TOKENS` | `4096` | Per-call token budget for most agents |
| `REFINER_MAX_TOKENS` | `8192` | Refiner-only token budget (emits full portfolio + per-critique rationale; 4096 was truncating mid-string) |
| `MAX_ITERATIONS` | `3` | Generator ↔ evaluator rounds (override via `--iterations`) |
| `PASS_THRESHOLD` | `7` | Minimum average score for a portfolio to pass |
| `TARGET_MAX_LOSS` | `0.05` | The max annual loss-budget target the selector aims for; also templated into the Generator / Evaluator / Refiner prompts. Override per run via `--max-loss` (or the server's `max_loss` field) |
| `UNDER_UTILISATION_RATIO` | `0.8` | Fraction of `TARGET_MAX_LOSS` below which drawdown counts as "wasting risk capacity" |
| `UNDER_UTILISATION_BAND` | `0.04` | Drawdowns below this are flagged as "wasting risk capacity" in feedback. **Auto-derives** as `UNDER_UTILISATION_RATIO × TARGET_MAX_LOSS` and is recomputed whenever `--max-loss` changes the budget |
| `ADVISOR_FEEDBACK_RHO_THRESHOLD` | `0.7` | Minimum \|ρ\| for an Advisor-flagged pair to be fed back into the next Generator iteration's prompt (Advisor still flags pairs ≥ 0.5 for the final report) |
| `DEFAULT_CAPITAL` | `100_000.0` | USD assumed for the whole-share lot-size feasibility check (override via `--capital`) — defined in `pricing.py` |
| `PRICING_DISCLAIMER` | … | Yahoo Finance data-source caveat shown in the markdown report's pricing section — defined in `pricing.py` |
| `RISK_HORIZONS` | `(1, 3, 5, 10)` | Holding periods (years) reported by the risk profile — defined in `risk.py` |
| `RISK_BLOCK_DAYS` | `126` | Bootstrap block length (~6 months) — preserves volatility clustering; `risk.py` |
| `RISK_N_SIMS` | `20_000` | Monte-Carlo paths simulated per horizon — `risk.py` |
| `RISK_SEED` | `7` | RNG seed for reproducible risk tables — `risk.py` |
| `RISK_PROXY_MAP` | … | Young-ETF → long-history asset-class proxy substitutions (so the sample spans 2008) — `risk.py` |
| `SDK_MAX_RETRIES` | `5` | SDK-level transparent retries on transient errors |
| `RETRY_MAX_ATTEMPTS` | `6` | Outer-wrapper attempts on top of SDK |
| `RETRY_INITIAL_BACKOFF_SECONDS` | `2.0` | First backoff before retry 2 |
| `RETRY_MAX_BACKOFF_SECONDS` | `32.0` | Cap on per-step backoff |
| `RETRYABLE_HTTP_STATUS` | `{429, 500, 502, 503, 504, 529}` | Status codes worth retrying |

## Extending This

Some natural next steps:

- **Structured critique.** Have the Evaluator return critique as a list of `{issue, severity, suggested_fix}` objects instead of one prose paragraph. The Refiner could then address each item explicitly and report which were resolved.
- **Deterministic constraint checker.** Add a Python pre-check that verifies hard rules (ticker count ≤ 15, sector caps, leverage, instruments restricted to the spec's `asset_universe`) before the LLM Evaluator sees the portfolio. Cheap, deterministic, no opinion drift. Would also catch the "Generator used a ticker outside the asset_universe" failure mode the Evaluator currently has to spot manually. (The same idea applies to correlation: a small static `|ρ| > 0.6` matrix in Python could enforce the diversification rule deterministically rather than relying on the LLM-as-judge.)
- **Real historical data for backtests.** `yfinance` is already in the pipeline for spot prices and lot-size feasibility — extend it to pull multi-year price history, then have the Evaluator run actual backtests on 2008 / 2020 / 2022 instead of estimating losses from memory. This would tighten the loss-cap judgement significantly.
- **Tool use.** Give the Generator and Evaluator Claude tool-use capabilities so they can call Python functions (mean-variance solver, Sharpe ratio, drawdown calc, the correlation lookup above) rather than reasoning from memory.
- **Sprint decomposition.** For a more complex version, break the work into sprints (asset selection → weight optimisation → tail-risk hedging → final review), each with its own generator ↔ evaluator loop.

## Disclaimer

This is a conceptual exploration of an AI engineering pattern. It is **NOT** financial advice. The portfolio allocations produced are illustrative and must not be used for actual investment decisions.
