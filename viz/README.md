# `viz/` — on-demand UI designer

Research-mode tool: turn a `harness_output.json` file + a natural-language request into a self-contained interactive HTML dashboard.

This is a **standalone exploration** — completely decoupled from the harness pipeline. It reads the static JSON file and produces a single HTML file. No build step, no server, no scaffolding.

Two modes:
- **One-shot** (default): generate → save → open in browser → done.
- **Interactive** (`--interactive` / `-i`): generate the initial dashboard, then enter a REPL where you can ask Claude to refine the file in place ("make the chart bigger", "switch to a dark theme", "add a tab for the advisor pairs"). Uses the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview) under the hood — Claude reads the current HTML, applies targeted `Edit` calls, and you reload the browser to see the change.

## How it works

```
$ uv run viz/ui_designer.py harness_output.json

What would you like the dashboard to show?
(One sentence is fine; or describe a multi-section layout.)
> compare iteration scores side-by-side and show a sortable table of final allocations

→ Generating dashboard with claude-opus-4-7 (max_tokens=32768) ...
✓ Dashboard saved to: viz/generated/dashboard_20260524_143012.html
  Size: 14,832 bytes
  Opening in browser ...
```

What happens under the hood:

1. CLI loads the JSON file
2. Prompts you for what to show (or accepts `--prompt`)
3. Sends the JSON + request to Claude with a system prompt that frames the task as "generate a complete interactive HTML dashboard, single file, Tailwind + Chart.js / Plotly via CDN, embed data inline"
4. Extracts the HTML from the response (markdown-fence aware fallback)
5. Saves to `viz/generated/dashboard_<timestamp>.html` (gitignored)
6. Opens it in your default browser

## Design philosophy

**Maximum expressiveness, good-enough robustness.** The LLM gets the entire web platform as a canvas — Tailwind for styling, Chart.js / Plotly.js for charts, vanilla JS for interactivity, marked.js for rendering long prose fields. The only hard constraint is "single self-contained file" so the output is shareable / inspectable.

The system prompt nudges Claude toward expressive UI patterns (tabs, drill-downs, hover tooltips, conditional formatting, sortable tables, side-by-side comparisons) and away from "just dump the JSON into a table". It also has explicit "be honest about the data" instructions — if the user asks for something not in the JSON (e.g., Sharpe ratio), Claude is told to note the absence rather than fabricate.

## Usage

> Note: `uv run` auto-detects Python files, so the `python` keyword is unnecessary (`uv run viz/ui_designer.py ...` works the same as `uv run python viz/ui_designer.py ...`). Examples below use the shorter form.

```bash
# 1. Interactive — tool prompts for the request
uv run viz/ui_designer.py harness_output.json

# 2. One-shot — pass the request as a flag
uv run viz/ui_designer.py harness_output.json \
    --prompt "compare iteration scores side by side"

# 3. Cheaper model (Sonnet) for quick experiments
uv run viz/ui_designer.py harness_output.json \
    --model sonnet --prompt "show me the pricing rows as a sortable table"

# 4. Cheapest model (Haiku) for one-shot smoke tests
uv run viz/ui_designer.py harness_output.json \
    --model haiku --prompt "just the final portfolio as a donut chart"

# 5. Save to a specific path
uv run viz/ui_designer.py harness_output.json \
    -o /tmp/dash.html

# 6. Generate without auto-opening the browser (useful for scripting / CI)
uv run viz/ui_designer.py harness_output.json --no-open

# 7. Very large dashboard — bump the token ceiling and switch to Sonnet
#    (Opus caps at 32k output tokens; Sonnet supports up to 64k)
uv run viz/ui_designer.py harness_output.json \
    --model sonnet --max-tokens 65536 \
    --prompt "comprehensive multi-tab dashboard: overview, per-iteration deep-dive, refinement before/after, advisor + pricing"

# 8. Combine everything — explicit prompt, custom output, no browser, Sonnet
uv run viz/ui_designer.py harness_output.json \
    --model sonnet \
    --prompt "executive summary at the top, then an accordion for each iteration" \
    -o ~/Desktop/portfolio_dashboard.html \
    --no-open

# 9. Interactive refinement — generate once, then iterate
uv run viz/ui_designer.py harness_output.json \
    --interactive \
    --prompt "compare iteration scores side by side"
# Once the dashboard opens, you can type follow-ups at the `refine>` prompt:
#   refine> make the comparison a single grouped bar chart
#   refine> use a darker accent — the green is hard to read
#   refine> add a section at the top with the headline pass/fail verdict
# Reload the browser to see each change.  Empty line / 'exit' / Ctrl+D to leave.
```

## CLI flags

| Flag | Effect |
|---|---|
| `--prompt`, `-p` | Natural-language request. If omitted, the tool prompts interactively. |
| `--output`, `-o` | Output HTML path. Default: `viz/generated/dashboard_<timestamp>.html` |
| `--model` | Default `claude-opus-4-7`. Accepts the aliases `haiku`, `sonnet`, `opus`, or a full model ID. |
| `--max-tokens` | Output token budget. Default `32768`. It's a *ceiling*, not a quota — you only pay for tokens actually generated, so set it high. Opus caps at 32k; Sonnet supports up to 64k. Bump (and switch to Sonnet) if the dashboard truncates. |
| `--no-open` | Save the HTML but don't auto-open in the browser. |
| `--interactive`, `-i` | After the initial generation, enter a REPL backed by the Claude Agent SDK. Each turn reads the current dashboard, applies targeted edits, and prints a one-line summary; you reload the browser to see the change. Session writes are scoped to the dashboard file (every other path is refused). |

## Example requests worth trying

Single-focus drill-downs:
- *"show me only the final portfolio's allocations as a pie chart with descriptions on hover"*
- *"big-number summary at the top — expected return, max drawdown, pass/fail — then the iteration history as a timeline"*
- *"show me the advisor's correlation pairs as a heatmap"*

Multi-section dashboards:
- *"three tabs: Overview (headline metrics), Iterations (compare all three side by side), Risk (pricing + advisor)"*
- *"executive summary at the top, then a deep-dive accordion for each iteration"*

Targeted analyses:
- *"highlight which iteration had the best balance of return and drawdown using a scatter plot"*
- *"show me how the refiner changed the portfolio — before / after table with deltas"*

## Cost / latency

Per generation (rough): ~12k input tokens (the JSON is ~50KB) + ~5k output tokens = roughly $1 on Opus, $0.20 on Sonnet, $0.07 on Haiku. Wall clock: 15-30 seconds depending on model.

## What works, what doesn't (research notes)

This is an exploration to see whether on-demand LLM-generated UI is conceptually viable for surfacing structured data. Things worth observing as you experiment:

- **Does Claude pick the right layout?** Or does it default to "tables for everything"?
- **Does it use the expressive vocabulary** (tabs, drill-downs, conditional colour)?
- **Is the data honest?** Does it match the JSON, or does it confabulate?
- **Are the charts well-chosen?** Bar vs. pie vs. scatter — is the choice motivated?
- **Does interactivity work?** Sortable tables, hover tooltips, tab switching?
- **What kinds of requests fail?** Vague vs. specific; visual vs. analytical?

## Limitations

- Generated dashboards are static — no live data, no two-way interaction with the harness.
- Claude can write broken JS/HTML; usually visible in the browser console. Re-run with a clearer prompt if so.
- Interactive mode does not auto-refresh the browser — reload manually after each edit. A websocket / hot-reload bridge would be a natural next step.
- No prompt caching yet on the one-shot path. If you iterate on requests for the same JSON, the JSON gets re-sent each time. (The Agent SDK in `--interactive` mode keeps the conversation in a single session, so the file is only sent once per session.)
- Generated dashboards are gitignored on purpose — they're not source of truth, they're disposable views.
