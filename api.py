"""
Anthropic API client + small helpers shared by ``agents.py`` and the
orchestrator in ``harness.py``.

What lives here:

* The singleton ``anthropic.Anthropic`` client.
* ``call_claude`` — the Messages API wrapper with exponential-backoff
  retry on transient errors and an explicit truncation warning when
  the response hits ``stop_reason == "max_tokens"``.
* ``_parse_json_response`` — fail-soft JSON parser used by every
  agent.  Returns ``{}`` on unparseable output so the pipeline degrades
  through dataclass defaults rather than crashing.
* Model / token / retry constants.  These are deliberately
  module-level mutables so the harness CLI can patch them (e.g.,
  ``api.MODEL = ...`` from ``--model X`` or ``--test``).
  ``call_claude`` resolves them at call time, so patches take effect
  immediately without having to re-import anything.

Has no dependency on ``models``, ``agents``, ``harness``, ``pricing``,
or ``report`` — strictly a leaf module on the project's dependency graph.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from typing import Any, Callable

import anthropic


# ---------------------------------------------------------------------------
# Model selection (per-agent; patched by the CLI in harness.py)
# ---------------------------------------------------------------------------
MODEL = "claude-opus-4-7"
# Per-agent override.  The Planner is mostly recall + JSON structuring, so
# it doesn't need Opus.  Generator / Evaluator / Refiner do the real
# reasoning work and stay on MODEL (Opus by default).
PLANNER_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Token budgets
# ---------------------------------------------------------------------------
MAX_TOKENS = 4096
# ---- Heavy reasoning-agent headroom (Generator / Evaluator / Refiner) ----
# These agents emit structured JSON AFTER a chunk of reasoning, and when
# adaptive "thinking" is enabled those thinking tokens count against
# max_tokens.  A powerful thinker like Fable 5 can spend the entire 4096
# default on thinking and emit ZERO text (stop_reason=max_tokens, 0-char
# response -> `_parse_json_response` falls back to an empty dict and the
# agent's output is lost).  max_tokens is a CEILING, not a target — you are
# billed for tokens actually generated, and adaptive thinking self-regulates —
# so a generous ceiling is free insurance.  NB: on Opus 4.7/4.8 adaptive
# thinking is OFF unless the request sets thinking={"type":"adaptive"} (this
# harness does not — yet), so today this headroom is slack on Opus and load-
# bearing mainly for Fable; see GENERATOR_EFFORT / EVALUATOR_EFFORT below.
GENERATOR_MAX_TOKENS = 16384   # full portfolio JSON (allocations + descriptions
                               # + methodology + rationale) after reasoning
REFINER_MAX_TOKENS = 16384     # portfolio JSON + point-by-point critique reply
EVALUATOR_MAX_TOKENS = 16384   # multi-round backtest narration + scores/critique
# The Planner runs on Sonnet and is NOT affected by --reasoning-model; its
# spec (objective / constraints / asset_universe / risk_budget /
# evaluation_criteria) plus web_search narration fits comfortably in 8192.
PLANNER_MAX_TOKENS = 8192

# ---------------------------------------------------------------------------
# Effort (output_config.effort) for the heavy reasoning agents
# ---------------------------------------------------------------------------
# `effort` (GA on Opus 4.5+, Sonnet 4.6, Fable 5 — NOT Haiku 4.5 / Sonnet 4.5)
# trades token spend for thoroughness: it tunes how deeply the model reasons
# (when thinking is on) AND how thoroughly it acts.  At "max" the Generator
# reasons harder before emitting, and the Evaluator and Refiner run more of
# their compute_backtest tool calls to verify against real history.  The API
# default is "high"; the ladder is low < medium < high < xhigh < max, so "max"
# is the ceiling.  Use "max" when correctness matters more than cost — which it
# does for portfolio construction + QA.  Applied via call_claude(effort=...)
# and gated by _model_supports_effort so a --test (Haiku) run doesn't 400.
# Pairs best with adaptive thinking; see the max_tokens note above.
GENERATOR_EFFORT = "max"
EVALUATOR_EFFORT = "max"
REFINER_EFFORT = "max"   # the post-loop finisher; it has compute_backtest and
                         # is the only construction agent that sees ground truth


# ---------------------------------------------------------------------------
# Retry / backoff config for transient Anthropic API errors
# ---------------------------------------------------------------------------
# Sequence of attempts and sleeps when an attempt fails with a retryable
# error (529 Overloaded, 429 rate limit, 5xx, connection / timeout errors):
#
#   attempt 1  →  sleep   2s (±25% jitter) →
#   attempt 2  →  sleep   4s  →
#   attempt 3  →  sleep   8s  →
#   attempt 4  →  sleep  16s  →
#   attempt 5  →  sleep  32s  →
#   attempt 6  →  give up and re-raise
#
# Max wall-clock spent on retries before giving up ≈ 62s (+ jitter).
SDK_MAX_RETRIES = 5                              # SDK-level retries (fast transient blips)
RETRY_MAX_ATTEMPTS = 6                           # outer wrapper attempts on top of SDK
RETRY_INITIAL_BACKOFF_SECONDS = 2.0
RETRY_MAX_BACKOFF_SECONDS = 32.0
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504, 529}


# ---------------------------------------------------------------------------
# Client construction (fires at import time — fails fast on missing key)
# ---------------------------------------------------------------------------
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit(
        "ERROR: ANTHROPIC_API_KEY environment variable is not set. "
        "Export it (e.g. `export ANTHROPIC_API_KEY=sk-ant-...`) and try again."
    )

# Bump SDK retries above the default of 2.  The SDK handles fast transient
# blips silently; our outer wrapper (in call_claude) handles longer overload
# events with visible progress logs.
client = anthropic.Anthropic(max_retries=SDK_MAX_RETRIES)


# ---------------------------------------------------------------------------
# Internal: classify which exceptions should trigger a retry
# ---------------------------------------------------------------------------
def _is_retryable(exc: BaseException) -> bool:
    """
    Return True for transient Anthropic API errors worth retrying:
      • 429 Rate-limit, 5xx, 529 Overloaded — server-side back-pressure.
      • Connection / timeout errors — network blips.
    Auth, validation, and other 4xx errors are NOT retried.
    """
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return getattr(exc, "status_code", None) in RETRYABLE_HTTP_STATUS
    return False


# ---------------------------------------------------------------------------
# Internal: single-round-trip Messages API call with exponential-backoff retry
# ---------------------------------------------------------------------------
def _messages_create_with_retry(**kwargs: Any) -> Any:
    """
    One call to ``client.messages.create(**kwargs)`` wrapped in the same
    exponential-backoff retry policy used elsewhere in this module.

    Retry is per-roundtrip on purpose: when ``call_claude`` runs a
    multi-round tool-use loop, a transient blip on round N retries
    ONLY round N — earlier conversation state is preserved.  Wrapping
    the whole loop would restart from scratch and re-issue tool calls,
    which would be wrong (handlers can be non-idempotent).
    """
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            resp = client.messages.create(**kwargs)
            # Surface truncation explicitly — otherwise it presents as
            # malformed JSON downstream and is hard to diagnose.  Include
            # the model name so mixed-model runs are easier to debug.
            if getattr(resp, "stop_reason", None) == "max_tokens":
                print(
                    f"  ⚠️  {kwargs.get('model')}: response hit "
                    f"max_tokens={kwargs.get('max_tokens')} — output likely "
                    f"truncated. Consider raising max_tokens for this call.",
                    flush=True,
                )
            return resp
        except Exception as exc:
            if attempt == RETRY_MAX_ATTEMPTS or not _is_retryable(exc):
                raise
            backoff = min(
                RETRY_INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                RETRY_MAX_BACKOFF_SECONDS,
            )
            backoff *= 1 + random.random() * 0.25  # +0–25% jitter
            kind = type(exc).__name__
            status = getattr(exc, "status_code", None)
            status_suffix = f" (HTTP {status})" if status else ""
            print(
                f"  ⚠️  Anthropic API {kind}{status_suffix} on attempt "
                f"{attempt}/{RETRY_MAX_ATTEMPTS} — retrying in {backoff:.1f}s",
                flush=True,
            )
            time.sleep(backoff)
    # The loop only exits via return or raise — this is unreachable but
    # satisfies static analysers.
    raise RuntimeError("_messages_create_with_retry exited unexpectedly")


# ---------------------------------------------------------------------------
# Internal: extract the final text block from a response
# ---------------------------------------------------------------------------
def _extract_text(resp: Any) -> str:
    """
    Pull the LAST text block out of a response's ``content`` list.

    Pre-tool-use this was always just ``resp.content[0].text``.  With
    tool use a single response may have multiple content blocks (text +
    tool_use, or — for server-side tools like web_search — intermediate
    text + server_tool_use + web_search_tool_result + final text).
    The user-visible answer is the LAST text block; everything before
    it is reasoning + tool I/O.

    Returns "" if no text block exists (rare; means the model only
    emitted tool_use without any commentary).
    """
    texts: list[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            t = getattr(block, "text", "")
            if isinstance(t, str):
                texts.append(t)
    return texts[-1] if texts else ""


# ---------------------------------------------------------------------------
# Internal: marshal a handler's return value into a tool_result string
# ---------------------------------------------------------------------------
def _tool_result_content(value: Any) -> str:
    """JSON-encode non-string handler results so the model can read them."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Internal: which models accept output_config.effort
# ---------------------------------------------------------------------------
def _model_supports_effort(model: str) -> bool:
    """
    Whether ``output_config.effort`` is accepted by ``model``.

    Effort is GA on Opus 4.5+, Sonnet 4.6, and Fable 5, but Haiku 4.5 and
    Sonnet 4.5 reject it with a 400.  This harness only ever runs
    opus-4-7 / sonnet-4-6 / fable-5 / haiku-4-5 (the last via ``--test``), so
    excluding Haiku is what matters in practice; sonnet-4-5 is excluded too for
    safety.  Resolved at call time because ``--test`` patches ``MODEL`` to
    Haiku globally.
    """
    m = model.lower()
    return "haiku" not in m and "sonnet-4-5" not in m


# ---------------------------------------------------------------------------
# Public: Messages API wrapper (with optional tool-use loop)
# ---------------------------------------------------------------------------
def call_claude(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = MAX_TOKENS,
    effort: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    max_tool_rounds: int = 8,
) -> str:
    """
    Send ``system`` + ``user`` to Claude and return the model's final
    text response, with exponential-backoff retry on transient errors
    (529 Overloaded, 429 rate limits, 5xx, connection blips).
    Re-raises immediately on terminal errors (auth, bad request, etc.)
    so we fail fast on real bugs.

    ``model`` and ``max_tokens`` are agent-tunable knobs.  Both are
    resolved at call time so CLI patches to ``MODEL`` still apply.
    Default: ``MODEL`` (Opus) and ``MAX_TOKENS`` (4096).

    ``effort`` (optional ``low``/``medium``/``high``/``xhigh``/``max``) sets
    ``output_config.effort`` to trade token spend for reasoning depth +
    thoroughness.  It is added to the request ONLY when the effective model
    accepts it (see ``_model_supports_effort``), so passing ``effort="max"``
    is safe even under ``--test`` (Haiku), where it is silently dropped.

    Tool use (optional)
    -------------------
    If ``tools`` is provided, the model can call them.  Supports BOTH
    flavors of Anthropic tool use:

      * **Client-side tools.**  Define a tool with a name + JSON schema
        and a Python handler.  The model emits a ``tool_use`` block;
        we execute the handler, send the ``tool_result`` back, loop
        until ``stop_reason != "tool_use"`` or ``max_tool_rounds`` is
        hit.  Per-round retry preserves conversation state across
        transient blips.

      * **Server-side tools.**  Tools like ``web_search`` declared via
        ``{"type": "web_search_20250305", "name": "web_search", ...}``
        execute on Anthropic's infrastructure.  No handler needed —
        the search runs inline and ``server_tool_use`` /
        ``web_search_tool_result`` blocks appear in the same response.
        These don't trigger our loop (stop_reason is ``end_turn``),
        they're just non-text blocks we ignore when extracting the
        final answer.

    ``tool_handlers`` maps tool name → callable that takes the parsed
    ``input`` dict and returns the result.  Return value is
    JSON-encoded if it isn't already a string.  Handler exceptions and
    unknown-tool errors are reported back to the model via
    ``tool_result(is_error=True)`` so the model can recover (e.g., fix
    bad inputs and retry) rather than the whole pipeline crashing.

    ``max_tool_rounds`` caps the client-side loop.  Default 8.
    """
    # Resolve the model at call time, not at function-definition time —
    # otherwise the CLI's --model / --test patches to the MODEL global
    # would be invisible here.
    effective_model = model if model is not None else MODEL

    create_kwargs: dict[str, Any] = {
        "model": effective_model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if tools:
        create_kwargs["tools"] = tools
    if effort and _model_supports_effort(effective_model):
        # output_config.effort tunes reasoning depth + thoroughness.  Silently
        # skipped on models that reject it (Haiku under --test) so the smoke
        # test still runs; resolved against effective_model (post --test patch).
        create_kwargs["output_config"] = {"effort": effort}

    # Fast path: no tools means single round-trip, behavior identical
    # to the pre-tool-use version of this wrapper.
    if not tools:
        resp = _messages_create_with_retry(**create_kwargs)
        return _extract_text(resp)

    # Client-side tool-use loop.  Server-side tools (web_search etc.)
    # don't trigger this loop because they execute inline and return
    # stop_reason="end_turn" on the first round.
    handlers = tool_handlers or {}
    resp: Any = None
    for round_idx in range(1, max_tool_rounds + 1):
        resp = _messages_create_with_retry(**create_kwargs)

        if getattr(resp, "stop_reason", None) != "tool_use":
            return _extract_text(resp)

        # Collect every tool_use block the model emitted (it may emit
        # several in parallel — we execute them all and send back one
        # batched user turn with all the tool_result blocks).
        tool_uses = [
            b for b in (getattr(resp, "content", []) or [])
            if getattr(b, "type", None) == "tool_use"
        ]
        if not tool_uses:
            # stop_reason claimed tool_use but there are no tool_use
            # blocks — defensive bail-out, return whatever text exists.
            return _extract_text(resp)

        # Append the assistant turn to the conversation so the next
        # request carries full context.
        create_kwargs["messages"].append(
            {"role": "assistant", "content": resp.content}
        )

        # Execute each tool and collect tool_result blocks.
        tool_results: list[dict[str, Any]] = []
        for tu in tool_uses:
            name = getattr(tu, "name", "")
            tu_id = getattr(tu, "id", "")
            inp = getattr(tu, "input", {}) or {}
            handler = handlers.get(name)
            if handler is None:
                msg = f"ERROR: no handler registered for tool '{name}'"
                print(f"  ⚠️  tool '{name}' (round {round_idx}): {msg}",
                      flush=True)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": msg,
                    "is_error": True,
                })
                continue
            try:
                # Log the call so the user can see the tool-use trace.
                # Keep input compact — just key names, not full values.
                print(
                    f"  🔧 tool '{name}' (round {round_idx}) "
                    f"called with keys: {list(inp.keys())}",
                    flush=True,
                )
                result = handler(inp)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": _tool_result_content(result),
                })
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                print(f"  ⚠️  tool '{name}' raised: {msg}", flush=True)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": msg,
                    "is_error": True,
                })

        create_kwargs["messages"].append(
            {"role": "user", "content": tool_results}
        )

    # max_tool_rounds exhausted — the model is still asking for more
    # tool calls.  Return whatever final text exists; the caller's JSON
    # parser will degrade gracefully if nothing usable came back.
    print(
        f"  ⚠️  tool-use loop hit max_tool_rounds={max_tool_rounds}; "
        f"returning last response (model may not be finished)",
        flush=True,
    )
    return _extract_text(resp)


# ---------------------------------------------------------------------------
# Public: best-effort JSON parse for model responses
# ---------------------------------------------------------------------------
def _parse_json_response(raw: str, *, agent: str = "agent") -> dict[str, Any]:
    """
    Parse a model response that's supposed to be JSON.  Three attempts,
    in order of preference:

      1. Strict ``json.loads`` on the whole string.
      2. Markdown-fenced block: extract the JSON between ``​```json``
         (or bare ``​```) fences, then strict parse.  Catches the common
         case where the model wraps JSON in a code block, often with a
         prose preamble before it — the greedy regex (attempt 3) would
         over-grab in that case and include the closing fence.
      3. Greedy ``{...}`` regex extract, then strict parse.  Last-ditch
         fallback for unfenced JSON with surrounding prose.

    If ALL THREE fail (e.g., the response was truncated mid-string by
    ``max_tokens`` and every extracted span is incomplete), this function
    logs a clear diagnostic and returns ``{}`` so the caller can degrade
    gracefully — the orchestrator already handles empty/missing fields
    via dataclass defaults.

    The ``agent`` label is only used for the log message.
    """
    # Attempt 1: strict parse on the whole response
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        pass

    # Attempt 2: ```json fenced block (handles preamble + fences cleanly).
    # ``` may or may not carry a "json" language tag; we accept both.
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        try:
            return json.loads(fence_match.group(1), strict=False)
        except json.JSONDecodeError:
            pass

    # Attempt 3: greedy {...} (existing fallback)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(), strict=False)
        except json.JSONDecodeError:
            pass

    # Total failure — log and degrade gracefully.
    print(
        f"  ⚠️  {agent}: could not parse JSON from response "
        f"({len(raw):,} chars). Returning empty dict so the pipeline "
        f"continues. First 300 chars of response follow:\n"
        f"  {raw[:300]!r}",
        flush=True,
    )
    return {}
