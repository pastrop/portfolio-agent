"""
On-demand UI designer for harness_output.json.

Research-mode tool: takes the harness's static JSON trace + a natural-language
request, asks Claude to write a complete self-contained interactive HTML
dashboard, saves it, and opens it in the browser.

Usage:
    uv run viz/ui_designer.py path/to/harness_output.json \
        --prompt "compare iteration scores side by side and show a sortable
                  table of final allocations"

    # Interactive — prompt for the request:
    uv run viz/ui_designer.py path/to/harness_output.json

Design philosophy: maximum expressiveness, good-enough robustness.  The
LLM gets the whole web platform as a canvas (Tailwind + Chart.js + Plotly
via CDN, vanilla JS for interactivity) and is constrained only by the
"single self-contained file" requirement so the output is shareable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import textwrap
import webbrowser
from datetime import datetime
from pathlib import Path

import anthropic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "claude-opus-4-7"
# Dashboards can be sizable — Tailwind classes, Chart.js scripts, embedded
# data.  max_tokens is a CEILING not a quota — you only pay for tokens
# actually generated, so set it high.  32k covers comprehensive multi-
# section dashboards on Opus.  For very large dashboards switch to
# Sonnet (--model sonnet) which supports up to 64k output.
DEFAULT_MAX_TOKENS = 32_768

_MODEL_ALIASES: dict[str, str] = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",
}


# ---------------------------------------------------------------------------
# System prompt — the heart of the project
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert front-end UI designer.  You generate
    self-contained interactive HTML dashboards that visualise the
    output of a portfolio-optimization harness.

    INPUT YOU WILL RECEIVE:
      • A natural-language request describing what the user wants to see.
      • A JSON trace from the harness — see SCHEMA below for the shape.

    OUTPUT FORMAT — STRICT:
      • A COMPLETE HTML document.  Begin with `<!DOCTYPE html>` and end
        with `</html>`.  NOTHING before, NOTHING after.
      • NO markdown fences (no ```html, no ```).
      • NO prose preamble, no explanatory notes.  The browser will
        render exactly what you emit.

    HARD CONSTRAINTS:
      • SINGLE self-contained file.  No external assets that you author.
      • Tailwind CSS via CDN:
          <script src="https://cdn.tailwindcss.com"></script>
      • Chart.js via CDN (preferred for most charts):
          <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      • Plotly.js via CDN (if you need something Chart.js can't do —
        sunbursts, treemaps, 3D, heatmaps, etc.):
          <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
      • marked.js via CDN if you need to render long prose / markdown
        fields (rationale, critique, etc.):
          <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
      • EMBED the data directly: inline the JSON inside a
          <script>const data = { ... };</script>
        block.  Do NOT fetch from any URL.  Do NOT reference local files.
      • Responsive — must look reasonable on a laptop screen.
      • Works offline once loaded (after the CDN scripts hydrate).

    DESIGN PRINCIPLES — BE EXPRESSIVE:
      The user wants to see what is POSSIBLE.  Don't just dump JSON
      into a table.  Specifically:
        • Lead with the HEADLINE.  What is the single most important
          number / verdict?  Show it big at the top.
        • Use the full vocabulary of a modern dashboard: tabs,
          collapsible sections, sortable / filterable tables, hover
          tooltips, conditional formatting (green/red on deltas),
          side-by-side comparisons, drill-downs, sparklines.
        • Tell a STORY.  A dashboard for "did the portfolio pass QA?"
          should look very different from one for "compare all three
          iterations head-to-head".  Match the layout to the question.
        • Pick a thoughtful colour palette.  Default to dark
          background + high-contrast accents if the data is dense.
        • Use whitespace.  No 14-point Comic Sans dumps.

    BE HONEST ABOUT THE DATA:
      • If the user asks for something the JSON doesn't contain
        (e.g., "show me the Sharpe ratio" — not computed by the
        harness), include a visible note in the dashboard saying
        the data isn't present, rather than fabricating values.
      • Compute derived values (means, deltas, sums) only when
        unambiguous from the source data.

    SCHEMA OF THE harness_output.json YOU'LL RECEIVE:
    {
      "model": str,                       # e.g. "claude-opus-4-7"
      "max_iterations": int,              # e.g. 3
      "pass_threshold": int,              # e.g. 7
      "target_max_loss": float,           # e.g. 0.05  (5% loss cap)
      "mode": str,                        # "optimized" | "preservation"
                                          #   (missing ⇒ treat as "optimized")
      "horizon_years": int,               # investment horizon, e.g. 10
      "horizon_posture": str,             # derived risk-posture label, e.g.
                                          #   "Growth" / "Balanced growth" /
                                          #   "Conservative balanced" /
                                          #   "Capital preservation"
      "spec": {                           # output of the Planner
        "objective": str | dict,
        "constraints": str | dict,
        "asset_universe": str | dict,
        "risk_budget": str | dict,
        "evaluation_criteria": str | dict,
        "raw_text": str                   # full original JSON the planner emitted
      },
      "iteration_history": [              # one entry per Generator/Evaluator round
        {
          "iteration": int,               # 1-based
          "allocations": {ticker: weight, ...},
          "descriptions": {ticker: "plain-english", ...},
          "expected_return": float,
          "expected_max_drawdown": float,
          "scores": {
            "constraint_compliance": int,
            "return_potential": int,
            "diversification": int,
            "implementability": int,
            "methodology_rigour": int
          },
          "average_score": float,
          "passed": bool,
          "critique_snippet": str,        # first 300 chars of critique
          "selected": bool                # the one the Selector kept
        }, ...
      ],
      "selected_iteration": int | null,   # null if no iteration passed (then iter 1 is fallback)
      "final_proposal": {                 # the portfolio chosen as final
        "allocations": {ticker: weight, ...},
        "descriptions": {ticker: str, ...},
        "expected_annual_return": float,
        "expected_max_drawdown": float,
        "methodology": str,
        "rationale": str,
        "raw_text": str
      },
      "final_evaluation": {
        "passed": bool,
        "scores": {...same 5 keys as above...},
        "average_score": float,
        "critique": str                   # FULL critique, can be very long
      },
      "refinement": {                     # optional Refiner pass
        "performed": bool,
        "promoted": bool,                 # was the refined version promoted to final?
        "improvements": {
          "score_deltas": {criterion: {"before": int, "after": int, "delta": int}, ...},
          "average_score_before": float, "average_score_after": float, "average_score_delta": float,
          "expected_return_before": float, "expected_return_after": float,
          "expected_max_drawdown_before": float, "expected_max_drawdown_after": float,
          "passed_before": bool, "passed_after": bool,
          "allocation_changes": [{ticker, before, after, delta, kind: "added"|"removed"|"changed"}, ...]
        },
        "refined_proposal": {...same shape as final_proposal...},
        "refined_evaluation": {...same shape as final_evaluation...}
      },
      "correlation": {                    # computed pairwise-correlation snapshot (no LLM)
        "performed": bool,
        "skipped_reason": str | null,
        "window_years": int,              # trailing window used
        "sample_start": str, "sample_end": str, "sample_days": int,
        "frequency": "daily",
        "coverage_weight": float,         # fraction of the book priced (0-1)
        "modeled_tickers": [ticker, ...],
        "dropped_tickers": [ticker, ...], # no priceable history (e.g. option overlays)
        "pairs": [                        # |rho| >= 0.5, sorted strongest first
          {"a": ticker_a, "b": ticker_b, "rho": float, "high": bool}, ...
        ],
        "high_pairs_count": int,          # pairs at |rho| >= 0.85 (highly redundant)
        "error": str | null
      },
      "pricing": {                        # yfinance lot-size feasibility
        "performed": bool,
        "capital": float,                 # assumed investable USD
        "total_invested": float,
        "leftover_cash": float,
        "max_abs_drift": float,           # largest |actual_weight - target| across rows
        "rows": [
          {"ticker": str, "weight": float, "status": "ok"|"error",
           "price": float | null, "target_dollars": float,
           "shares": int, "actual_dollars": float,
           "actual_weight": float, "weight_drift": float, "error": str | null}, ...
        ],
        "failed_tickers": [ticker, ...],
        "disclaimer": str,
        "fetched_at": str                 # ISO timestamp
      },
      "risk_profile": {                   # Monte-Carlo return distribution (no LLM)
        "performed": bool,
        "skipped_reason": str | null,     # set when skipped (e.g. --no-risk / --test)
        "sample_start": str, "sample_end": str,   # historical window bootstrapped
        "sample_years": float,
        "includes_2008": bool,            # does the window span the 2008 crisis?
        "limiting_ticker": str | null,    # holding that constrained the window start
        "annualized_return": float,       # realized in the sample (geometric)
        "annualized_vol": float,          # realized in the sample
        "coverage_weight": float,         # fraction of the book actually modeled (0-1)
        "proxy_substitutions": [          # young ETF -> long-history asset-class proxy
          {"original": ticker, "proxy": ticker}, ...
        ],
        "dropped_tickers": [ticker, ...], # non-priceable legs (e.g. option overlays)
        "horizons": [                     # one row per holding period
          {"horizon_years": int,
           "median": float,               # typical (50th-pct) total compounded return
           "mean": float,
           "prob_end_down": float,        # P(end underwater) at this horizon, 0-1
           "bad_5th": float,              # 5th-pctile (1-in-20) total return
           "severe_1st": float}, ...      # 1st-pctile (1-in-100) total return
        ],
        "n_sims": int, "block_days": int,
        "disclaimer": str,
        "error": str | null               # set if the profile could not run
      }
    }

    Some fields may be missing or null on any given run (e.g., refinement
    may be skipped via --no-refine; risk_profile may be skipped via
    --no-risk, or carry an `error` when offline).  Handle absence
    gracefully — hide the corresponding section rather than rendering
    empty placeholders.

    When `mode == "preservation"` (short horizon — the optimizer was
    deliberately bypassed), tell the PRESERVATION STORY: the deterministic
    capital-preservation template (final_proposal.allocations), the
    redirect_message explaining why the horizon was redirected, plus the
    pricing and risk_profile sections.  Do NOT render empty QA / iteration
    sections in this mode — iteration_history is [], refinement is
    skipped, and final_proposal.expected_annual_return /
    expected_max_drawdown are null (guard any formatting on them).

    Now generate the dashboard.
""")


# ---------------------------------------------------------------------------
# Refinement system prompt — used by the interactive REPL after the initial
# dashboard has been generated.  The agent's job here is targeted edits, not
# regeneration: read the current file, apply a minimal patch, describe it.
# ---------------------------------------------------------------------------
REFINE_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an expert front-end UI designer continuing to refine an
    interactive HTML dashboard for a portfolio-optimization harness.

    The dashboard file lives at:
      {dashboard_path}

    HOW TO WORK:
      • Always Read the file BEFORE editing — it is the ground truth.
      • Prefer targeted Edit calls (small old_string/new_string ranges)
        over rewriting the whole file.  Reach for Write only if the user
        explicitly asks for a wholesale restructure.
      • Preserve the inline `<script>const data = {{ ... }};</script>`
        block — that is the harness JSON.  Never modify the data itself.
      • Preserve the CDN script tags in <head> (Tailwind, Chart.js,
        Plotly, marked) unless the user explicitly asks to remove them.
      • After applying your edits, give a one- or two-sentence summary
        of what changed.  The user reloads the browser themselves.

    DESIGN PRINCIPLES (unchanged from the initial generation):
      • Be expressive — tabs, drill-downs, conditional formatting,
        sortable tables, hover tooltips, side-by-side comparisons.
      • Be honest — if the user asks for data the JSON does not
        contain (e.g. Sharpe ratio), add a visible note in the
        dashboard rather than fabricating values.
      • Single self-contained file — no external assets you author.
""")


# ---------------------------------------------------------------------------
# Interactive refinement loop (Claude Agent SDK)
# ---------------------------------------------------------------------------
# Layered on top of the existing one-shot generator.  Turn 1 still goes
# through the raw Anthropic streaming path (proven, ~30s on Opus).  From
# turn 2 onwards we hand the file to a ClaudeSDKClient session restricted
# to Read / Edit / Write on the dashboard file itself.  The SDK manages
# the tool loop, the file edits, and the conversation history; we just
# print results and re-prompt.
async def interactive_refinement(dashboard_path: Path, model: str) -> int:
    """REPL: read user requests, dispatch Read/Edit via SDK, loop until exit."""
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            PermissionResultAllow,
            PermissionResultDeny,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
        )
    except ImportError:
        print(
            "ERROR: --interactive requires the `claude-agent-sdk` package. "
            "Install with `uv add claude-agent-sdk` (already declared in "
            "pyproject.toml — run `uv sync`).",
            file=sys.stderr,
        )
        return 2

    dashboard_abs = dashboard_path.resolve()

    def is_dashboard_path(p: object) -> bool:
        if not isinstance(p, str) or not p:
            return False
        try:
            return Path(p).resolve() == dashboard_abs
        except OSError:
            return False

    async def can_use_tool(tool_name, input_data, _context):
        # Read / Edit / Write are allowed, but ONLY on the dashboard file
        # itself.  Everything else is denied with a clear message so the
        # model can adjust on the next round.
        if tool_name in {"Read", "Edit", "Write"}:
            if is_dashboard_path(input_data.get("file_path")):
                return PermissionResultAllow(updated_input=input_data)
            return PermissionResultDeny(
                message=(
                    f"This session only has access to {dashboard_abs}. "
                    f"Refused {tool_name} on "
                    f"{input_data.get('file_path')!r}."
                ),
                interrupt=False,
            )
        return PermissionResultDeny(
            message=f"Tool {tool_name!r} is not available in refinement mode.",
            interrupt=False,
        )

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=REFINE_SYSTEM_PROMPT_TEMPLATE.format(
            dashboard_path=str(dashboard_abs),
        ),
        allowed_tools=["Read", "Edit", "Write"],
        can_use_tool=can_use_tool,
        cwd=str(dashboard_abs.parent),
        # We're not asking the model to plan complex workflows — a single
        # refinement should converge in 2-4 tool calls (Read, then 1-3
        # Edits).  Cap it so a runaway loop doesn't burn budget.
        max_turns=15,
    )

    print()
    print("─" * 70)
    print("Entering interactive refinement mode.")
    print(f"Dashboard:  {dashboard_abs}")
    print(f"Model:      {model}")
    print("Describe what to change, then reload the browser to see edits.")
    print("Empty line, 'exit', 'quit', or Ctrl+D to leave.")
    print("─" * 70)

    try:
        async with ClaudeSDKClient(options=options) as client:
            while True:
                print()
                try:
                    user_input = input("refine> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n(exiting interactive mode)")
                    return 0
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("(exiting interactive mode)")
                    return 0

                await client.query(user_input)
                edits_made = 0
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                text = block.text.strip()
                                if text:
                                    print(text)
                            elif isinstance(block, ToolUseBlock):
                                if block.name in {"Edit", "Write"}:
                                    edits_made += 1
                                # One-line tool-call trace so the user can
                                # see what the agent is doing.
                                fp = block.input.get("file_path", "") if isinstance(block.input, dict) else ""
                                suffix = f" {Path(fp).name}" if fp else ""
                                print(f"  · {block.name}{suffix}")
                    elif isinstance(message, ResultMessage):
                        if edits_made:
                            print(
                                f"\n↻ Reload the dashboard in your browser "
                                f"({edits_made} change{'s' if edits_made != 1 else ''} applied)."
                            )
    except KeyboardInterrupt:
        print("\n(interrupted)")
        return 130


# ---------------------------------------------------------------------------
# HTML extraction — be lenient about whatever wrapper the model emits
# ---------------------------------------------------------------------------
def extract_html(response_text: str) -> str:
    """
    Pull the HTML document out of the model's response.  Tries, in order:

      1. Markdown-fenced block:  ```html  ...  ```
      2. Bare DOCTYPE-to-/html match.
      3. Return the response verbatim (last-ditch).

    The model is instructed to emit raw HTML with no fences, but defence
    in depth is cheap.
    """
    # Attempt 1: ```html fenced block
    fenced = re.search(
        r"```(?:html)?\s*(<!DOCTYPE\s+html.*?</html>)\s*```",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1)

    # Attempt 2: bare DOCTYPE-to-/html
    bare = re.search(
        r"<!DOCTYPE\s+html.*?</html>",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if bare:
        return bare.group(0)

    # Attempt 3: give up and return the whole thing (will be obvious
    # in the browser if something's off)
    return response_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "On-demand UI designer for harness_output.json — turn a "
            "natural-language request into an interactive HTML dashboard."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              # Interactive — tool prompts for the request
              uv run viz/ui_designer.py harness_output.json

              # One-shot via --prompt
              uv run viz/ui_designer.py harness_output.json \\
                  --prompt "compare iteration scores side by side"

              # Use Sonnet instead of Opus to cut cost
              uv run viz/ui_designer.py harness_output.json \\
                  --model sonnet --prompt "show me the pricing rows as a sortable table"

              # Save to a specific path (default: viz/generated/dashboard_<timestamp>.html)
              uv run viz/ui_designer.py harness_output.json \\
                  -o /tmp/dash.html
        """),
    )
    parser.add_argument(
        "json_path",
        help="Path to the harness_output.json file to visualise.",
    )
    parser.add_argument(
        "--prompt", "-p",
        help=(
            "Natural-language description of what to show. If omitted, "
            "the tool prompts for it interactively."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        help=(
            "Output HTML file path. Default: "
            "viz/generated/dashboard_<timestamp>.html"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Model to use (default {DEFAULT_MODEL}). Accepts short "
            f"aliases (haiku, sonnet, opus) or a full Anthropic ID."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Output token budget (default {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Save the dashboard but don't auto-open it in a browser.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help=(
            "After the initial generation, enter a REPL where you can ask "
            "for incremental refinements ('make the chart bigger', 'switch "
            "to a dark theme', etc.).  Uses the Claude Agent SDK to apply "
            "targeted edits to the dashboard file in place."
        ),
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set. "
            "Export it (e.g. `export ANTHROPIC_API_KEY=sk-ant-...`)."
        )

    json_path = Path(args.json_path)
    if not json_path.exists():
        sys.exit(f"ERROR: file not found: {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: {json_path} is not valid JSON: {exc}")

    # Get the user's request
    if args.prompt:
        user_request = args.prompt.strip()
    else:
        print("What would you like the dashboard to show?")
        print("(One sentence is fine; or describe a multi-section layout.)")
        try:
            user_request = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(cancelled)")
            return 1
    if not user_request:
        sys.exit("ERROR: no request given")

    model = _MODEL_ALIASES.get(args.model, args.model)
    print(f"\n→ Generating dashboard with {model} (max_tokens={args.max_tokens}) ...")
    print(f"  Request: {user_request}\n")

    user_message = (
        f"USER REQUEST:\n{user_request}\n\n"
        f"DATA (the full harness_output.json):\n"
        f"```json\n{json.dumps(data, indent=2, default=str)}\n```\n\n"
        f"Generate the complete HTML dashboard now.  Remember: begin with "
        f"<!DOCTYPE html>, end with </html>, nothing before, nothing after."
    )

    client = anthropic.Anthropic()
    # Stream the response.  Anthropic's SDK refuses non-streaming calls
    # whose theoretical wall-clock could exceed 10 minutes (any
    # max_tokens above ~24k for current models triggers this).  Streaming
    # avoids the timeout, lets us show live progress, and yields the same
    # final Message object so the downstream code is unchanged.
    print("  Streaming response", end="", flush=True)
    bytes_since_dot = 0
    with client.messages.stream(
        model=model,
        max_tokens=args.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for chunk in stream.text_stream:
            # One dot per ~2KB of generated text — visible heartbeat
            # without flooding the terminal with HTML.
            bytes_since_dot += len(chunk)
            if bytes_since_dot >= 2000:
                print(".", end="", flush=True)
                bytes_since_dot = 0
        resp = stream.get_final_message()
    print(" done")
    response_text = resp.content[0].text

    if getattr(resp, "stop_reason", None) == "max_tokens":
        # Suggest doubling.  If we're already above 32k, also recommend
        # switching to Sonnet (which supports 64k output vs Opus's 32k).
        suggested = args.max_tokens * 2
        model_hint = ""
        if args.max_tokens >= 32_000 and "opus" in model.lower():
            model_hint = (
                " On Opus the output cap is 32k tokens — switch to "
                "`--model sonnet` for up to 64k."
            )
        print(
            f"  ⚠️  Response hit max_tokens={args.max_tokens} — the "
            f"dashboard may be truncated. Re-run with "
            f"--max-tokens {suggested}.{model_hint}"
        )

    html = extract_html(response_text)

    # Annotate the file with the request + model + timestamp for debugging
    # / reproducibility.  Tucked into an HTML comment so it doesn't render.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    annotation = textwrap.dedent(f"""\
        <!--
            Generated by viz/ui_designer.py
            Source data: {json_path.name}
            Model:       {model}
            Generated:   {timestamp}
            Request:     {user_request}
        -->
    """)
    # Inject the annotation right after <!DOCTYPE html> for visibility
    if html.lstrip().lower().startswith("<!doctype"):
        first_line_end = html.find(">") + 1
        html = html[:first_line_end] + "\n" + annotation + html[first_line_end:]

    # Pick output path
    if args.output:
        output_path = Path(args.output)
    else:
        generated_dir = Path(__file__).parent / "generated"
        generated_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = generated_dir / f"dashboard_{ts}.html"

    output_path.write_text(html, encoding="utf-8")
    print(f"\n✓ Dashboard saved to: {output_path}")
    print(f"  Size: {len(html):,} bytes")

    if not args.no_open:
        url = output_path.resolve().as_uri()
        print(f"  Opening in browser ...")
        webbrowser.open(url)

    if args.interactive:
        return asyncio.run(interactive_refinement(output_path, model))

    return 0


if __name__ == "__main__":
    sys.exit(main())
