# How `whatif.py` verifies a portfolio

This document explains **how a portfolio is checked** by `whatif.py` — the
verification techniques, what each one measures, how it is computed, and where
its limits are. It applies to both modes (editing a completed run, or analyzing
a portfolio you supply): in either case the same engine runs on a plain
`{ticker: weight}` allocation.

## Guiding principles

Every check is:

- **Deterministic and data-grounded.** All numbers come from real market
  history (Yahoo Finance via `yfinance`), not from a language model. There is
  **no LLM and no API key** in this path. The same portfolio always produces the
  same verdict (the Monte-Carlo step is seeded).
- **Judged on the *delivered* book.** The portfolio is verified exactly as
  listed — the checks never credit a hedge, overlay, or rebalancing rule the
  book does not actually hold (see *Loss-floor compliance* below).
- **Fail-soft.** No single data problem aborts the analysis. A ticker with thin
  history, an offline blip, or an un-priceable leg degrades that one metric
  (and says so) rather than crashing the run.
- **Honest about coverage.** Wherever history is incomplete or a substitute was
  used, the output states it (coverage %, proxy substitutions, the limiting
  ticker, dropped legs).

The single entry point is `evaluate()`, which runs the six independent
techniques below on the allocation and collects their results.

## The data foundation

- **Source:** daily total-return history (dividends reinvested) from `yfinance`.
- **Proxy substitution for coverage.** Many ETFs are young (e.g. `SGOV` 2020,
  `DBMF` 2019), so they have no 2008 data. For the historical checks, such
  tickers are swapped for long-history asset-class proxies via a curated
  `RISK_PROXY_MAP` (~15 entries, e.g. `SGOV→SHV`, `SCHP→TIP`, `MUB→VWITX`) so the
  sample can span the 2008 crisis. A proxy approximates the *asset class*, not
  the exact fund; every substitution is reported.
- **Coverage weight.** Each backtest reports the fraction of portfolio weight
  that actually had data in the window, so a partially-covered result is never
  mistaken for a full one.

---

## 1. Historical stress-window backtest → the loss-cap verdict *(the headline)*

**Question it answers:** *"In the worst real years on record, would this book —
as held — have stayed within its annual loss cap?"*

**Module:** `loss_floor.run_loss_floor_check`.

**How it works:**

1. The delivered holdings (proxy-substituted for coverage) are backtested over
   three **calendar-year stress windows**: **2008** (Global Financial Crisis),
   **2020** (COVID crash), and **2022** (rate-hike drawdown).
2. For each year it computes the **gross calendar-year total return** (Jan 1 →
   Dec 31) and the intra-year max drawdown. The annual loss cap is a
   *calendar-year* limit, so the calendar-year total return is the number
   compared against it.
3. A year whose data coverage is below **90%** is treated as *unjudgeable* and
   excluded from the verdict (rather than reported as a misleadingly shallow
   loss).
4. The **worst gross annual loss** across the reliably-covered years is compared
   to the cap (default −5%). `organic_pass` is true when the worst year is
   within the cap **on the book's own holdings**.

**The un-held-hedge guard.** This is the key honesty check. If a book breaches
the cap on its raw holdings, the analysis asks whether it actually *holds* a
hedge that could contain the loss. It detects only genuine
options/derivative legs (a put/collar/option-overlay line item, or a dedicated
tail-hedge ETF) — deliberately **not** colloquial "hedges" like gold, which is
often described as a "tail hedge" but is not one. If the book breaches gross
**and** holds no real hedge, it is flagged `relies_on_unheld_mechanism`: its
"within cap" story depends on something not in the portfolio.

**Output:** worst gross annual loss + year, per-year returns/drawdowns/coverage,
`organic_pass`, and the un-held-hedge flag. In a what-if edit, this is also where
the "the swap **flips** the cap to BREACHED" verdict comes from.

---

## 2. Forward return distribution → block-bootstrap Monte-Carlo

**Question it answers:** *"Over my holding horizon, what's the range of outcomes
— the typical result, the chance of ending down, and the bad tails?"*

**Module:** `risk.run_risk_profile` (constants: 252 trading days/yr, 126-day
blocks, 20,000 paths/horizon, seed 7, history from 2006-01-01).

It manufactures tens of thousands of plausible multi-year paths out of real
history and reports where they land.

### Stage A — build the historical "engine" (one daily portfolio-return series)

Before any simulation, the book is reduced to a single 1-D series of daily
returns:

1. **Proxy + weights.** Young ETFs are swapped for long-history proxies and the
   weights aggregated onto each proxy.
2. **Download** daily dividend-adjusted closes from **2006-01-01** (so the
   sample spans the 2008 GFC — "the GFC is the prize").
3. **Common history.** Keep only days where *every* holding has data (inner
   join); require ≥ `2 × 126 = 252` overlapping days, or bail with an error.
4. **Renormalize** weights over what survived — a dropped un-priceable leg (e.g.
   a synthetic option overlay) simply isn't modeled, which makes the result
   **conservative** (`coverage_weight` records how much was kept).
5. **Collapse to the portfolio's daily returns:**
   ```python
   rets = px[present].pct_change().dropna()   # daily % return, per holding
   port = (rets @ weights).to_numpy()          # one number per day = the book's return that day
   ```

**Why this matters.** `port[t]` is the actual return your exact book would have
had on real day *t*. Because each day's value already reflects *how all the
holdings moved together that day*, the **contemporaneous cross-asset correlation
is baked in for free** — there is no covariance matrix and no per-asset
simulation; the bootstrap simply resamples a single 1-D array. (Implicit
assumption: fixed target weights ⇒ a daily-rebalanced book, not buy-and-hold
weight drift.)

It also records the sample window, `sample_years`, `includes_2008`, the
`limiting_ticker` (whose inception cut the window short), and the realized
annualized return/vol here.

### Stage B — why block bootstrap, then how

**Why not the obvious alternatives:**

- *A normal / parametric model* understates fat tails — real crashes are far
  worse than a Gaussian predicts.
- *Resampling individual days (IID bootstrap)* destroys **volatility
  clustering** (calm and turbulent days arrive in streaks) and autocorrelation →
  unrealistically smooth paths with thin tails.
- *Block bootstrap* resamples **contiguous runs** of real days, so each run keeps
  its real day-to-day structure (turbulent stretches stay turbulent). It's the
  standard fix for serially-dependent financial data.

**The mechanics (`_bootstrap_terminal`).** For a horizon of `h` years:

```
T  = h * 252            # trading days to fill   (5y → 1260)
nb = ceil(T / 126)      # 126-day blocks needed  (5y → 10)
```

For **each** of the 20,000 paths: draw `nb` random start indices (uniform in
`[0, n − 126)`); each start selects a **contiguous 126-day (~6-month) slice** of
`port`; stitch the slices end-to-end, trim to exactly `T` days, and compound to
one terminal return `∏(1 + rₜ) − 1`.

A single 5-year path is 10 real half-year chunks taken from random points in
2006–today and glued together:

```
history:  2006 ─────────────────────────────────────── today    (n daily returns)
                │126d│        │126d│   │126d│  …   (random slices)
one path =  [126d][126d][126d][126d][126d][126d][126d][126d][126d][126d]
            └────────────── 10 blocks → trim to 1260 → compound → 1 terminal return ──────────────┘
repeat 20,000× → 20,000 terminal returns
```

Inside each block the real structure is intact; only at the **9 seams** between
blocks is continuity artificial — so most of each path is genuine contiguous
history. (Block-length tradeoff: longer blocks preserve more structure but offer
fewer distinct pieces to recombine; shorter blocks give more variety but more
seams. 126 ≈ 6 months is the chosen middle.) It is vectorized in chunks of 4,000
paths with numpy fancy-indexing, so all 20,000 run fast.

### Stage C — read the distribution

Per horizon, from the 20,000 terminal returns:

- **median** — the typical (50th-percentile) outcome.
- **prob_end_down** — fraction of paths ending below 0 (chance you're underwater
  at the horizon).
- **bad_5th** — 5th-percentile terminal return (a 1-in-20 bad case).
- **severe_1st** — 1st-percentile (1-in-100, the deep tail).

These are **terminal outcomes only** — deliberately no "worst dip along the way"
column.

### Horizon grid

`bracket_horizons` surrounds the configured horizon with a shorter (~⅓) and a
longer (~2×) point plus a 1-year anchor (so a 5-year target reports ~`(1, 5,
10)`). A no-flag run keeps the fixed `(1, 3, 5, 10)`.

### Assumptions & limits

- **Stationarity:** it assumes future 6-month chunks resemble 2006→today. It
  *can* stitch several crisis blocks into a worse-than-any-single-historical-year
  path (good for tail estimation) but **cannot invent a regime that never
  occurred** in the sample.
- **No imposed drift:** the median inherits the sample period's actual drift —
  it's "what the distribution looks like *if the future rhymes with 2006–now*,"
  not a forward expected-return forecast.
- **Frictionless daily rebalancing:** fixed target weights, no costs / taxes /
  slippage.
- **Proxies approximate asset classes**, not exact funds.

**Reproducibility & honesty.** Seeded (`RISK_SEED=7`) so reports are stable;
never raises (any failure populates an `error` field); proxy substitutions,
`coverage_weight`, `includes_2008`, and `limiting_ticker` are all in the output.

---

## 3. Trailing-window backtest → recent realized performance

**Question it answers:** *"How would this exact book have actually done over the
last few years?"*

**Module:** `tools.compute_backtest` over a trailing window (default 5 years).

**How it works:** aligns the holdings' daily returns over the window, forms the
weighted portfolio return series, and reports the realized **total return**,
**annualized volatility** (daily σ × √252), and **max drawdown** (peak-to-trough
on the cumulative path), plus coverage. Unlike the stress and Monte-Carlo checks,
this uses the **real tickers** (no proxying) — it's "what literally happened
recently."

---

## 4. Pairwise correlation → diversification / hidden redundancy

**Question it answers:** *"Are these holdings actually diversified, or are some
of them the same bet in different wrappers?"*

**Module:** `correlation.run_correlation`.

**How it works:** downloads ~5 years of daily returns for the **real** holdings
(no proxying here — exact tickers), requires **≥120 overlapping daily returns**
(~6 months) per pair, and computes the pairwise correlation matrix. It then surfaces every pair
with **|ρ| ≥ 0.50**, flagging the subset at **|ρ| ≥ 0.85** as **highly
redundant** — e.g. two intermediate-bond sleeves, or two broad-equity sleeves,
that move together and so add little real diversification.

This is computed from data on purpose: an earlier LLM "Advisor" recalled these
correlations from memory and was systematically wrong (it over-stated the cash
sleeve and under-stated the bond-duration cluster), so it was replaced by this
deterministic step.

---

## 5. Concentration → breadth and single-name risk

**Question it answers:** *"Is the book spread out, or piled into a few line items
— or worse, into a single company?"*

**Module:** `whatif._concentration`.

Two complementary measures:

- **Effective-N (1/HHI).** The Herfindahl index is the sum of squared weights;
  its reciprocal is the "effective number of equal positions." Ten equal 10%
  holdings → effective-N ≈ 10; everything in one line → 1. This captures
  *breadth across line items*.
- **Single-name flag.** Effective-N has a blind spot: it works on *ticker
  weights*, so a 14% S&P 500 ETF and a 14% single stock score **identically** —
  it cannot see that one is 500 companies and the other is one. So concentration
  also classifies each holding via `yfinance`'s `quoteType` and reports the
  total **single-name (individual stock) weight** and which tickers they are.
  For a fund→single-stock change, this flag — together with the volatility and
  loss-cap impact — is what surfaces the real risk that breadth alone misses.

---

## 6. Implementability → lot-size feasibility

**Question it answers:** *"Can this actually be bought with whole shares at the
given capital, and how far does that push it off the target weights?"*

**Module:** `pricing.run_pricing`.

Fetches the latest price per holding and runs a **whole-share lot-size** check
against the configured capital (default $100,000): how much gets invested, the
**leftover cash**, the **max weight drift** from target caused by share
granularity, and any tickers it couldn't price. It's a practicality check —
small books or high-priced shares can drift noticeably from target weights.

---

## How the techniques fit together

The checks form a layered picture, with one clear headline and several context
layers:

- **Headline — pass/fail:** the **loss-cap verdict** (technique 1). "Worst
  stress-year loss X% vs the ≤cap" is the single most important answer, and the
  un-held-hedge flag keeps it honest.
- **Forward risk:** the **Monte-Carlo distribution** (2) — typical outcome,
  chance of loss, and tails over the horizon.
- **Recent reality:** the **trailing backtest** (3).
- **Structure:** **correlation** (4) and **concentration + single-name** (5)
  explain *why* the risk numbers look the way they do (redundant bets,
  over-concentration, hidden single-stock exposure).
- **Practicality:** **lot-size feasibility** (6).

In what-if **edit** mode every technique is computed for both the before and the
after book and the deltas are reported, so you see exactly what a change does to
each dimension. In **direct** mode the same techniques run once, as a standalone
risk report.

## What this verification deliberately does *not* do

Being explicit about the boundaries:

- **No forward "expected return" claim.** Returns here are *historical* (stress /
  trailing backtests) or *distributional* (Monte-Carlo median) — never a forward
  point estimate. That kind of assertion only comes from the LLM construction
  pipeline, not this deterministic path.
- **No look-through into funds.** Concentration sees line items, not the
  underlying holdings of an ETF — which is exactly why the single-name flag
  exists as a separate signal.
- **Annual cap, not a horizon-aware loss probability.** The loss-cap check is a
  calendar-year limit on three historical stress years, not a modeled
  `P(loss > X% over H years)`.
- **No transaction costs, taxes, or slippage**, and **no rebalancing path** —
  weights are treated as static over each backtest window.
- **Proxies approximate asset classes**, not exact funds, in the history-spanning
  checks; the correlation and trailing checks use the real tickers instead.

All of these are disclosed in the outputs (coverage, proxy substitutions,
dropped legs, the un-held-hedge flag) so the reader can judge how much to trust
any single number.
