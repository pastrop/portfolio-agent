# `whatif/` — deterministic portfolio risk analysis

Research-mode tool with **two modes**, both deterministic and needing **no API key**:

1. **Edit a completed run** — take a `harness_output.json`, swap / set / drop a
   security, and see the **before/after** deltas (return, drawdown, loss-cap
   verdict) — without rerunning the LLM pipeline.
2. **Analyze a portfolio you supply** — hand it a fresh book (`--portfolio` or
   `--portfolio-file`) and get a **standalone** risk report.

Standalone, like `viz/` — it reads its inputs as black boxes and never touches
the harness. It reuses the harness's allocation-first, LLM-free eval core
(`loss_floor` / `risk` / `correlation` / `pricing` / `compute_backtest`) and
`report.py`'s section renderers.

## Usage

```bash
# --- EDIT a completed run (before/after) ---
# Swap a holding: what does it do to my risk?
uv run python whatif/whatif.py harness_output.json --substitute VTI=NVDA -o vti_nvda
# Pin a weight (others renormalize) / drop a holding; edits compose
uv run python whatif/whatif.py harness_output.json --set GLD=0.10 --remove VNQ

# --- ANALYZE a portfolio you supply (standalone) ---
# Inline (fractions, sum ~1.0):
uv run python whatif/whatif.py --portfolio "SCHD=0.24,VTV=0.20,SPY=0.20,SGOV=0.36" -o my_book
# From a file:
uv run python whatif/whatif.py --portfolio-file my_book.json
```

### Edit mode

Edits apply in order, then the book renormalizes **once** (never silently):
- `--substitute A=B` — move A's weight to B (added to B if already held; errors if A absent).
- `--set X=w` — pin X at fraction `w`; the others renormalize to fill `1−w`.
- `--remove A` — drop A; the rest renormalize.

### Direct mode — the portfolio file

`--portfolio-file` takes a JSON file in either of two shapes:

```jsonc
// 1) bare map: ticker -> weight (fractions)
{ "SCHD": 0.24, "VTV": 0.20, "SPY": 0.20, "SGOV": 0.36 }
```
```jsonc
// 2) self-contained scenario: holdings + its own assumptions
{
  "allocations": { "SCHD": 0.24, "VTV": 0.20, "SGOV": 0.56 },
  "max_loss": 0.05,
  "horizon_years": 5,
  "capital": 100000
}
```

Assumptions resolve as **explicit CLI flag → file/baseline value → default**.
Direct-mode defaults: `--max-loss 0.05`, `--horizon-years 5`, `--capital 100000`.

## What it reports

Recomputed from data (never the baseline's LLM-claimed numbers):
- **Loss-cap verdict** (the headline): worst stress-year gross loss vs the cap, and — in edit mode — whether the edit **flips it to BREACHED**.
- Stress-year (2008/2020/2022) gross return + drawdown — `run_loss_floor_check`.
- Forward Monte-Carlo distribution per horizon — `run_risk_profile`.
- Trailing-window return / volatility / drawdown — `compute_backtest`.
- |ρ| ≥ 0.85 correlation pairs — `run_correlation`.
- Lot-size feasibility — `run_pricing`.
- **Concentration**: effective-N (breadth) **plus a single-name flag** — effective-N on ticker weights is blind to fund→single-stock swaps (a 14% ETF and a 14% stock score identically); the flag (yfinance `quoteType`) + the vol/loss-cap numbers carry that risk.

## Output

- `whatif_<label>.json` — the portfolio in the **standard harness schema**, so `viz/ui_designer.py` renders it like any run (the LLM-only sections — `final_evaluation`, `iteration_history`, `expected_*` — are null/empty since no LLM ran, and the designer hides them). No comparison is in this file.
- `whatif_<label>.md` (+ CLI output) — edit mode: the before/after comparison; direct mode: a single-portfolio risk report.
- Any `harness_output.json` input is **read-only** — never overwritten.

Full design + decisions: `SESSION_NOTES.md` → "What-If scenario tool".

> v2 ideas (deterministic core unchanged): a `mode == "whatif"` designer branch for a true before/after dashboard, REPL, NL front-end, `--evaluate` (LLM QA on the book), multi-variant A/B, server endpoint.
