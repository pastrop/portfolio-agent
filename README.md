# Portfolio Optimization Harness

A multi-agent implementation inspired by Anthropic's [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps), applied to financial portfolio optimisation.

The harness started as the original three-agent pattern (Planner → Generator → Evaluator). It has since grown to a four-agent pipeline (adding explicit selection and refinement), followed by three no-LLM post-processing steps — pricing, a Monte-Carlo risk profile, and a computed correlation snapshot — to address the gap between "passes QA" and "ready to actually use."

> **Note (correlation is computed, not recalled).** An earlier design had a fifth LLM "Advisor" agent estimate pairwise correlations from model memory. That proved unreliable — it systematically over-stated correlations in the cash / short-duration sleeve and under-stated the genuine bond-duration cluster — so it was replaced by a deterministic, yfinance-backed `correlation.py` step that computes the real numbers. See [`correlation.py`](correlation.py).

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
        │  │ GENERATOR │ ◀──── critique + drawdown guidance    │
        │  │  (Opus)   │                                       │
        │  └─────┬─────┘                                       │
        │        │ Portfolio                                   │
        │        ▼                                             │
        │  ┌───────────┐                                       │
        │  │ EVALUATOR │ ── scores + passed flag               │
        │  │  (Opus)   │                                       │
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
                       ┌──────────────┐
                       │ CORRELATION  │   computed pairwise daily-return
                       │ (no LLM)     │   correlations; flags |ρ| ≥ 0.85
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
| **Pricing** (no LLM call) | Fetches the latest price for each ticker via [`yfinance`](https://github.com/ranaroussi/yfinance) and computes a whole-share lot-size feasibility check against `--capital` USD (default $100k). Per-ticker failures (unknown ticker, network blip, model-invented pseudo-ticker like `SPX_PUT_SPREAD`) degrade gracefully. Output includes a Yahoo Finance data-source disclaimer. |
| **Risk Profile** (no LLM call) | Replaces the single `expected_max_drawdown` point estimate with a **return distribution**. Block-bootstraps historical daily returns into many multi-year paths and reports, per holding horizon (1/3/5/10y), the **median outcome**, the **chance of ending down**, and the **1-in-20 / 1-in-100 unlucky tails**. Substitutes long-history asset-class proxies for young ETFs so the sample spans the 2008 crisis; drops non-priceable legs (option overlays) and renormalizes, so the modeled downside is conservative. yfinance-backed and fail-soft. |
| **Correlation** (no LLM call) | Computes the **pairwise daily-return correlation** of the final holdings from real yfinance history over a trailing window (default 5y). Surfaces every pair at \|ρ\| ≥ 0.5 and flags the highly redundant ones at \|ρ\| ≥ 0.85. Non-priceable legs (option overlays) are dropped; pairs with too little overlapping history are skipped. **Replaces the old LLM "Advisor"** whose recalled-from-memory correlations were systematically wrong for the cash / short-duration sleeve. yfinance-backed and fail-soft. |

### Key design decisions

1. **Separation of generation and evaluation.** A generator asked to grade its own work will praise it, so the Evaluator is a separate agent with a "skeptical, rigorous" prompt that scores five 1–10 criteria and independently judges pass/fail. The Evaluator honours the spec **as written** — when the Planner defines an `enforcement_mechanism` for the loss cap (e.g., a dynamic de-risking trigger), the Evaluator models that mechanism when stress-testing and judges the post-mechanism loss against the cap, rather than vetoing on the pre-mechanism gross drawdown. Pass/fail and the critique are orthogonal: portfolios that meet the spec's stated criteria pass even when residual mechanism risks (slippage, gap-down, single point of failure) exist, but those risks are still surfaced in the critique for the human reader.

2. **Always run all iterations, then select.** The original harness exited the loop as soon as one iteration passed. That gave away later iterations that could have landed closer to the risk-budget target (`TARGET_MAX_LOSS`). Now the loop always runs `MAX_ITERATIONS` rounds; after the loop, the Selector picks the best passing iteration by closeness to the target.

3. **Iteration feedback is regime-aware.** Each round's feedback is shaped by the Evaluator's result: when an iteration passes but its drawdown is over target (`TARGET_MAX_LOSS`) the Generator is told to push it down; when drawdown is well under target (≤ `UNDER_UTILISATION_BAND`) the Generator is told the risk budget is being wasted; failed iterations pass the critique through verbatim. (An earlier design also prepended per-iteration Advisor correlation pairs here; that was removed when correlation moved to a deterministic post-processing step — see decision 4. The Generator's own prompt still carries an explicit overlap-group rule to discourage near-duplicate holdings.)

4. **Correlation is computed, not recalled.** Pairwise-correlation reporting used to be a fifth LLM "Advisor" agent that estimated ρ from training memory — exactly the task LLMs are worst at. Empirically it over-stated correlations across the cash / short-duration sleeve (e.g. SHY↔SGOV ≈ 0.9 when the real daily figure is ≈ 0.1 — near-zero-duration T-bills barely co-move with anything) and under-stated the genuine intermediate-bond cluster. Correlation is a computation, not a judgement, so it now lives in a deterministic, yfinance-backed `correlation.py` step (a sibling of `pricing.py` / `risk.py`): self-contained, fail-soft, no LLM. The Refiner remains the only agent that can replace the selected portfolio's holdings (and only if its output passes QA and stays within the loss target).

5. **Per-agent model tier.** Not every agent needs the most capable model. Generator / Evaluator / Refiner stay on Opus (the real reasoning work). Planner runs on Sonnet (recall + JSON structuring — doesn't need Opus), which trims default-run cost with no observable quality impact on the agents that drive the outcome. `--model X` overrides both to X — useful when one tier is overloaded or for direct cost/quality comparison.

6. **Evaluator honours the spec as written.** The Planner can define an `enforcement_mechanism` for the loss cap (e.g., a dynamic de-risking trigger, an options hedge overlay). When it does, the Evaluator models that mechanism when stress-testing and judges the **post-mechanism** annual loss against the cap — not the pre-mechanism gross drawdown. Pass/fail and the critique are orthogonal: portfolios that meet the spec's stated criteria pass, but the critique still surfaces residual mechanism risks (slippage, gap-down, single point of failure) so the human reader sees them. Without this, every aggressive portfolio gets vetoed for "breaching" a cap the spec's own mechanism was supposed to enforce.

7. **Resilience to transient API failures.** Each `call_claude` is wrapped in exponential-backoff retry on 429/5xx/529/connection/timeout errors (5 SDK retries + 6 outer attempts with 2/4/8/16/32s jitter, ≈62s max wall-clock). Auth/validation errors fast-fail. Refiner gets `REFINER_MAX_TOKENS = 8192` instead of the default 4096 (it emits both a full portfolio and a per-critique rationale; 4096 was truncating mid-string). All four `run_*` agents share a single fail-soft `_parse_json_response` helper — on unparseable model output, it returns `{}` and the pipeline degrades through dataclass defaults rather than crashing the whole run.

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

Per-agent models (Sonnet planner / Opus generator+evaluator+refiner), 3 iterations, refinement on, pricing + risk + correlation on:

```bash
uv run python harness.py
```

This makes ~9 API calls per run:

| Calls | Agent | Model |
|--:|---|---|
| 1 | Planner | Sonnet |
| 3 | Generator (one per iteration) | Opus |
| 3 | Evaluator (one per iteration) | Opus |
| 1 | Refiner | Opus |
| 1 | Evaluator (re-run on refined) | Opus |

Plus yfinance batches for pricing / risk / correlation (no API key needed; per-ticker failures degrade gracefully). Takes a few minutes against Opus.

### CLI flags

| Flag | Effect |
|---|---|
| `--test` | Smoke-test mode: Haiku 4.5 for ALL agents, 1 iteration, no refinement, no pricing, no risk, no correlation. ~3 API calls, cheapest end-to-end verification of the plumbing. Useful when Opus is overloaded or you just want to see the flow run. |
| `--model {haiku\|sonnet\|opus\|<full-id>}` | Override the model **for all agents** (planner / generator / evaluator / refiner). Aliases resolve to `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-7`. Any other string is passed through verbatim as a model ID. |
| `--iterations N` | Override `MAX_ITERATIONS` for this run (default 3). |
| `--max-loss FRACTION` | Override the max annual loss budget (`TARGET_MAX_LOSS`) as a fraction in `(0, 1)` — e.g. `0.10` for 10% (default 5%). Drives the Generator / Evaluator / Refiner prompts, the selection target, and the auto-derived under-utilisation band. **Unlike the other flags, this applies in `--test` mode too.** |
| `--no-refine` | Skip the post-selection Refiner pass. |
| `--no-prices` | Skip the post-selection Pricing pass (yfinance lookups + lot-size feasibility). Pricing is on by default. |
| `--no-risk` | Skip the post-selection Risk-profile pass (Monte-Carlo return distribution). On by default; needs yfinance (extra historical-data pulls). |
| `--no-correlation` | Skip the post-selection Correlation snapshot (computed pairwise daily-return correlations). On by default; needs yfinance, so combine with `--no-prices --no-risk` to run fully offline. |
| `--capital USD` | Capital assumed for the whole-share lot-size feasibility check (default $100,000). Only used when pricing is enabled. |

`--test` takes precedence — if combined with `--model` / `--iterations` / `--no-refine` / `--no-prices` / `--no-risk` / `--no-correlation` / `--capital`, the test-mode defaults win (test mode implies refine/pricing/risk/correlation all off). The one exception is `--max-loss`, which is orthogonal and applies even under `--test`.

### Examples

```bash
# Full default run (per-agent models, refinement on, pricing + risk + correlation on)
uv run python harness.py

# Quick smoke test — Haiku for everything, 1 iteration, no refiner/pricing/risk/correlation
uv run python harness.py --test

# Opus is overloaded? Full 3-iteration run on Sonnet (overrides per-agent split)
uv run python harness.py --model sonnet

# Skip the computed correlation snapshot
uv run python harness.py --no-correlation

# Skip yfinance price-fetching (e.g., offline or rate-limited)
uv run python harness.py --no-prices

# Fully offline — skip all three yfinance passes (pricing + risk + correlation)
uv run python harness.py --no-prices --no-risk --no-correlation

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

- **`harness_output.json`** — machine-readable trace: run config (`model`, `max_iterations`, `pass_threshold`, `target_max_loss`), spec, every iteration's allocations / scores, selected proposal, refinement block (before/after), pricing block (per-ticker prices + lot sizes + leftover cash), risk-profile block (per-horizon return distribution + sample window + proxy substitutions + coverage), correlation block (computed pairwise snapshot + flagged high pairs), and raw model responses. The trace is **self-describing** — an old `harness_output.json` can be re-rendered through `report.py` without rerunning the pipeline.
- **`harness_output.md`** — human-readable Markdown report. Renders cleanly in VS Code's built-in preview (`Cmd+Shift+V`) or any Markdown viewer. Contents:
  - Header summary (model, iterations, selected iteration, refinement / pricing / risk / correlation status, target loss, pass rule)
  - **Final Portfolio** table with `Ticker | Weight | Description` columns
  - Iteration Summary table comparing all iterations on score, return, drawdown, and distance to target — selected iteration starred
  - Investment Spec from the Planner
  - Selected Portfolio Methodology + Rationale
  - Selected Portfolio Evaluator Scores + Critique (the critique surfaces residual mechanism risks even when the portfolio passed — slippage, gap-down, single point of failure, etc.)
  - **Post-Selection Refinement** section with Score deltas, Portfolio metric deltas, and Allocation changes tables, plus the Refiner's point-by-point rationale and the re-evaluator's report
  - **Latest Prices & Lot-Size Feasibility** — per-ticker prices, target $ vs. actual whole-share $, weight drift, leftover cash, plus a Yahoo Finance data-source disclaimer
  - **Return Distribution (Monte-Carlo)** — per-horizon (1/3/5/10y) table of median outcome, chance of ending down, and 1-in-20 / 1-in-100 unlucky tails; plus the sample window (whether it spans 2008), coverage, and the long-history proxy substitutions used
  - **Correlation Snapshot (computed)** — pairwise daily-return correlations (yfinance) for the final holdings, sorted by strongest |ρ|, with the highly redundant pairs (|ρ| ≥ 0.85) flagged
  - Per-iteration detail
  - Planner / Generator / Evaluator / Refiner raw responses in collapsible `<details>` blocks

## Configuration

Edit the constants at the top of `harness.py`:

| Constant | Default | Meaning |
|---|---|---|
| `MODEL` | `claude-opus-4-7` | Default model for Generator / Evaluator / Refiner (override via `--model`, which patches all three per-agent globals) |
| `PLANNER_MODEL` | `claude-sonnet-4-6` | Model for the Planner (recall + JSON structuring) |
| `MAX_TOKENS` | `4096` | Per-call token budget for most agents |
| `REFINER_MAX_TOKENS` | `8192` | Refiner-only token budget (emits full portfolio + per-critique rationale; 4096 was truncating mid-string) |
| `MAX_ITERATIONS` | `3` | Generator ↔ evaluator rounds (override via `--iterations`) |
| `PASS_THRESHOLD` | `7` | Minimum average score for a portfolio to pass |
| `TARGET_MAX_LOSS` | `0.05` | The max annual loss-budget target the selector aims for; also templated into the Generator / Evaluator / Refiner prompts. Override per run via `--max-loss` (or the server's `max_loss` field) |
| `UNDER_UTILISATION_RATIO` | `0.8` | Fraction of `TARGET_MAX_LOSS` below which drawdown counts as "wasting risk capacity" |
| `UNDER_UTILISATION_BAND` | `0.04` | Drawdowns below this are flagged as "wasting risk capacity" in feedback. **Auto-derives** as `UNDER_UTILISATION_RATIO × TARGET_MAX_LOSS` and is recomputed whenever `--max-loss` changes the budget |
| `DEFAULT_CAPITAL` | `100_000.0` | USD assumed for the whole-share lot-size feasibility check (override via `--capital`) — defined in `pricing.py` |
| `PRICING_DISCLAIMER` | … | Yahoo Finance data-source caveat shown in the markdown report's pricing section — defined in `pricing.py` |
| `RISK_HORIZONS` | `(1, 3, 5, 10)` | Holding periods (years) reported by the risk profile — defined in `risk.py` |
| `RISK_BLOCK_DAYS` | `126` | Bootstrap block length (~6 months) — preserves volatility clustering; `risk.py` |
| `RISK_N_SIMS` | `20_000` | Monte-Carlo paths simulated per horizon — `risk.py` |
| `RISK_SEED` | `7` | RNG seed for reproducible risk tables — `risk.py` |
| `RISK_PROXY_MAP` | … | Young-ETF → long-history asset-class proxy substitutions (so the sample spans 2008) — `risk.py` |
| `CORR_WINDOW_YEARS` | `5` | Trailing window (years) for the computed correlation snapshot — `correlation.py` |
| `CORR_HIGH_THRESHOLD` | `0.85` | \|ρ\| at/above which a pair is flagged "highly redundant" — `correlation.py` |
| `CORR_REPORT_THRESHOLD` | `0.5` | Minimum \|ρ\| for a pair to appear in the snapshot — `correlation.py` |
| `SDK_MAX_RETRIES` | `5` | SDK-level transparent retries on transient errors |
| `RETRY_MAX_ATTEMPTS` | `6` | Outer-wrapper attempts on top of SDK |
| `RETRY_INITIAL_BACKOFF_SECONDS` | `2.0` | First backoff before retry 2 |
| `RETRY_MAX_BACKOFF_SECONDS` | `32.0` | Cap on per-step backoff |
| `RETRYABLE_HTTP_STATUS` | `{429, 500, 502, 503, 504, 529}` | Status codes worth retrying |

## Extending This

Some natural next steps:

- **Structured critique.** Have the Evaluator return critique as a list of `{issue, severity, suggested_fix}` objects instead of one prose paragraph. The Refiner could then address each item explicitly and report which were resolved.
- **Deterministic constraint checker.** Add a Python pre-check that verifies hard rules (ticker count ≤ 15, sector caps, leverage, instruments restricted to the spec's `asset_universe`) before the LLM Evaluator sees the portfolio. Cheap, deterministic, no opinion drift. Would also catch the "Generator used a ticker outside the asset_universe" failure mode the Evaluator currently has to spot manually. (Correlation is already computed deterministically for *reporting* in `correlation.py`; a natural next step is to promote it to an *enforced* gate — fail QA when an unjustified pair exceeds, say, `|ρ| > 0.85`.)
- **Real historical data for backtests.** `yfinance` is already in the pipeline for spot prices and lot-size feasibility — extend it to pull multi-year price history, then have the Evaluator run actual backtests on 2008 / 2020 / 2022 instead of estimating losses from memory. This would tighten the loss-cap judgement significantly.
- **Tool use.** Give the Generator and Evaluator more Claude tool-use capabilities so they can call Python functions (mean-variance solver, Sharpe ratio, the computed correlation matrix from `correlation.py`) during construction rather than reasoning from memory.
- **Sprint decomposition.** For a more complex version, break the work into sprints (asset selection → weight optimisation → tail-risk hedging → final review), each with its own generator ↔ evaluator loop.

## Disclaimer

This is a conceptual exploration of an AI engineering pattern. It is **NOT** financial advice. The portfolio allocations produced are illustrative and must not be used for actual investment decisions.
