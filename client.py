"""
Tiny CLI client for the portfolio-agent server.

Submits a run, polls until it finishes, prints a result summary, and
optionally saves the markdown report locally.

Usage examples
--------------

    # Smoke test (Haiku, 1 iteration, no refine/price/correlation)
    uv run python client.py --test "smoke test goal"

    # Full default run
    uv run python client.py "Optimise a portfolio for a US retail investor..."

    # Force all agents to Sonnet, save the report locally
    uv run python client.py --model sonnet --save-report ./report.md \\
        "Build an aggressive growth portfolio..."

    # Point at a remote server
    uv run python client.py --server http://10.0.0.5:8000 "goal here"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests


DEFAULT_SERVER = "http://127.0.0.1:8000"
DEFAULT_POLL_SECONDS = 5.0
# 10s is enough for a healthy local server; long enough that a transient
# network blip doesn't immediately fail the script.
HTTP_TIMEOUT_SECONDS = 10.0


def _build_request_body(args: argparse.Namespace) -> dict:
    """Construct the POST body, omitting unset optional fields so the
    server's own defaults apply (per-agent model mix, MAX_ITERATIONS, ...)."""
    body: dict = {
        "goal": args.goal,
        "test": args.test,
        "refine": not args.no_refine,
        "price": not args.no_price,
        "correlation": not args.no_correlation,
        "capital": args.capital,
    }
    if args.model is not None:
        body["model"] = args.model
    if args.iterations is not None:
        body["iterations"] = args.iterations
    return body


def _post_run(server: str, body: dict) -> dict:
    url = f"{server.rstrip('/')}/runs"
    r = requests.post(url, json=body, timeout=HTTP_TIMEOUT_SECONDS)
    if r.status_code >= 400:
        # Surface validation / server errors verbatim so the user can see
        # exactly what the server complained about.
        sys.exit(f"ERROR submitting run: HTTP {r.status_code} — {r.text}")
    return r.json()


def _poll_until_done(server: str, run_id: str, poll_seconds: float) -> dict:
    url = f"{server.rstrip('/')}/runs/{run_id}"
    last_status: str | None = None
    while True:
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
            r.raise_for_status()
        except requests.RequestException as exc:
            # Transient: report it but keep polling — the run itself is
            # still happening on the server even if our poll round-trip
            # blipped.
            print(f"  poll error ({exc}); retrying in {poll_seconds}s", file=sys.stderr)
            time.sleep(poll_seconds)
            continue

        view = r.json()
        status = view["status"]
        if status != last_status:
            print(f"  [{time.strftime('%H:%M:%S')}] {status}")
            last_status = status
        if status in ("succeeded", "failed"):
            return view
        time.sleep(poll_seconds)


def _print_summary(view: dict) -> None:
    summary = view.get("result_summary") or {}
    print()
    print("RUN SUCCEEDED")
    print(f"  selected iteration         : {summary.get('selected_iteration')}")
    print(f"  passed QA                  : {summary.get('passed_qa')}")
    avg = summary.get("final_average_score")
    ret = summary.get("final_expected_return")
    dd = summary.get("final_expected_max_drawdown")
    if avg is not None:
        print(f"  final avg score            : {avg:.2f}")
    if ret is not None:
        print(f"  final expected return      : {ret:.2%}")
    if dd is not None:
        print(f"  final expected max drawdown: {dd:.2%}")


def _save_artifact(server: str, run_id: str, kind: str, dest: Path) -> None:
    """Stream-download one of the per-run artifacts to a local path."""
    url = f"{server.rstrip('/')}/runs/{run_id}/{kind}"
    r = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS * 3, stream=True)
    if r.status_code >= 400:
        print(f"  could not fetch {kind}: HTTP {r.status_code} — {r.text}",
              file=sys.stderr)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"  saved {kind} → {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a portfolio run to the portfolio-agent server and wait for the result.",
    )
    parser.add_argument("goal", help="Free-form investment goal handed to the Planner agent.")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"Base URL of the server (default {DEFAULT_SERVER}).")
    parser.add_argument("--model", default=None,
                        help="Override all agents. Aliases: haiku|sonnet|opus, or a full model ID. "
                             "Omit for the per-agent mix.")
    parser.add_argument("--test", action="store_true",
                        help="Mirror --test on the CLI: all-Haiku, 1 iteration, skip refine/price/correlation.")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override MAX_ITERATIONS. Ignored when --test is also set.")
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--no-price", action="store_true")
    parser.add_argument("--no-correlation", action="store_true")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Investable capital in USD (default 100000).")
    parser.add_argument("--poll", type=float, default=DEFAULT_POLL_SECONDS,
                        help=f"Polling interval in seconds (default {DEFAULT_POLL_SECONDS}).")
    parser.add_argument("--save-report", metavar="PATH", default=None,
                        help="After success, download harness_output.md to PATH.")
    parser.add_argument("--save-trace", metavar="PATH", default=None,
                        help="After success, download harness_output.json to PATH.")
    args = parser.parse_args()

    body = _build_request_body(args)
    print(f"Submitting run to {args.server}/runs …")
    created = _post_run(args.server, body)
    run_id = created["run_id"]
    print(f"  run_id        = {run_id}")
    print(f"  artifacts_dir = {created['artifacts_dir']}")
    print()
    print(f"Polling every {args.poll}s …")

    view = _poll_until_done(args.server, run_id, args.poll)

    if view["status"] == "failed":
        print()
        print("RUN FAILED")
        print(f"  {view.get('error')}")
        print()
        print(f"  full stdout: GET {args.server}/runs/{run_id}/stdout")
        print(f"  on disk:     {view['artifacts_dir']}")
        sys.exit(2)

    _print_summary(view)
    print()
    print(f"Server-side artifacts: {view['artifacts_dir']}")
    print(f"  trace:  GET {args.server}/runs/{run_id}/trace")
    print(f"  report: GET {args.server}/runs/{run_id}/report")
    print(f"  stdout: GET {args.server}/runs/{run_id}/stdout")

    if args.save_report:
        print()
        _save_artifact(args.server, run_id, "report", Path(args.save_report))
    if args.save_trace:
        if not args.save_report:
            print()
        _save_artifact(args.server, run_id, "trace", Path(args.save_trace))


if __name__ == "__main__":
    main()
