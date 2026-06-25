# `whatif/` — deterministic what-if scenario tool

Research-mode tool: take a completed `harness_output.json`, edit the final
portfolio (swap / set / drop a security), and see the **deterministic**
before/after deltas — return, drawdown, and the loss-cap verdict — **without
rerunning the LLM pipeline**.

Standalone, like `viz/` — it reads the static JSON as a black box and never
touches the harness. It reuses the harness's allocation-first, LLM-free eval
core (`loss_floor` / `risk` / `correlation` / `pricing` / `compute_backtest`)
and `report.py`'s section renderers. **No API key needed.**

## Usage

```bash
# Swap a holding for another (the headline use): what does it do to my risk?
uv run python whatif/whatif.py harness_output.json --substitute VTI=NVDA -o vti_nvda

# Pin a weight (others renormalize to fill the rest) / drop a holding
uv run python whatif/whatif.py harness_output.json --set GLD=0.10 --remove VNQ

# Multiple edits compose; trailing-window length is configurable
uv run python whatif/whatif.py harness_output.json \
    --substitute VTI=QQQ --set SGOV=0.25 --trailing-years 7 -o stress
```

Edits (applied in order, then renormalized **once** — never silently):
- `--substitute A=B` — move A's weight to B (added to B if already held; errors if A absent).
- `--set X=w` — pin X at fraction `w`; the other holdings renormalize to fill `1−w`.
- `--remove A` — drop A; the rest renormalize.

## What it reports

For **both** the baseline and the edited book (recomputed from data — never the
baseline's LLM-claimed numbers):

- **Loss-cap verdict** (the headline): worst stress-year gross loss vs the run's
  cap, and whether the edit **flips it to BREACHED**.
- Stress-year (2008/2020/2022) gross return + drawdown — `run_loss_floor_check`.
- Forward Monte-Carlo distribution per horizon — `run_risk_profile`.
- Trailing-window return / volatility / drawdown — `compute_backtest`.
- New |ρ| ≥ 0.85 correlation pairs — `run_correlation`.
- Lot-size feasibility — `run_pricing`.
- **Concentration**: effective-N (breadth) **plus a single-name flag** — because
  effective-N on ticker weights is blind to fund→single-stock swaps (a 14% ETF
  and a 14% stock score identically); the flag (yfinance `quoteType`) and the
  volatility/loss-cap deltas carry that risk.

## Output

- `whatif_<label>.json` — the modified portfolio in the **standard harness
  schema**, so `viz/ui_designer.py` renders it like any run (the LLM-only
  sections — `final_evaluation`, `iteration_history`, `expected_*` — are
  null/empty since no LLM ran, and the designer hides them). **No comparison is
  in this file** — it's purely the resulting portfolio.
- `whatif_<label>.md` (+ CLI output) — the before/after comparison (loss-cap
  verdict, return/drawdown/concentration deltas). This is where the
  "what does the swap do?" answer lives.
- The baseline `harness_output.json` is **read-only** — never overwritten.

Full design + locked decisions: `SESSION_NOTES.md` → "What-If scenario tool".

> v2 ideas (deterministic core unchanged): REPL, a natural-language front-end,
> `--evaluate` (run the real LLM Evaluator on the edited book), multi-variant
> A/B ranking, and a server endpoint.
