# `server/` — architecture reference

A thin FastAPI layer that exposes `harness.run_harness` over HTTP for
local-research use. This document explains how the pieces fit together
and *why* the design is shaped this way. Pair with
[`../SESSION_NOTES.md`](../SESSION_NOTES.md) for the broader project
context.

## 1. Purpose and constraints

The server has one job: let a remote caller submit an investment goal,
have the existing multi-agent harness run against it, and retrieve the
result. Three constraints shaped every decision:

1. **No changes to the harness.** `harness.py`, `agents.py`, `api.py`,
   `pricing.py`, `report.py` must remain untouched. The CLI continues to
   work exactly as before. The server is *additive*.
2. **Single process, in-memory state.** This is the research phase — no
   database, no message broker, no multi-machine deployment. The
   registry of runs lives in process memory; on-disk artifact
   directories survive restarts but the registry does not.
3. **Long runs are normal.** A default Opus run takes minutes. A
   synchronous HTTP endpoint that blocks until completion would tie up
   client connections and hit timeouts. So all runs are asynchronous:
   `POST /runs` returns immediately with a `run_id`, the caller polls
   `GET /runs/{id}` (or, in Phase 2, subscribes to an SSE stream).

These three constraints are not arbitrary — each has a "cost" if
violated. Constraint #1 means the harness's mutable module globals
(`api.MODEL`, `api.PLANNER_MODEL`, `api.ADVISOR_MODEL`,
`harness.MAX_ITERATIONS`) and `print()`-based progress reporting must
be handled at the server's edge, not by refactoring the harness.

## 2. Module map

```
server/
├── __init__.py     package marker
├── main.py         FastAPI app + lifespan (starts/stops the JobManager)
├── routes.py       HTTP route handlers — translates HTTP ↔ JobManager calls
├── jobs.py         JobRecord + JobManager: registry, queue, worker loop
├── runner.py       QueueWriter + execute_run_sync: stdout capture,
│                   global snapshot/restore, harness invocation, artifact writing
├── storage.py      runs-dir resolution, run-ID generator, artifact name table
└── schemas.py      Pydantic request/response models + MODEL_ALIASES
```

Dependency graph (arrows point from consumer to provider):

```
main.py ──→ routes.py ──→ jobs.py ──→ runner.py ──→ {api, harness, report}
                │            │            │
                └─→ schemas.py            └─→ storage.py
                └─→ storage.py
```

Acyclic. No file in `server/` is imported by any module outside it.

## 3. Request lifecycle

```
┌──────────────┐    POST /runs body    ┌────────────────┐
│  HTTP client │─────────────────────→ │  route handler │
└──────────────┘                       │ (routes.py)    │
        ▲                              └────────────────┘
        │ 202 Accepted                          │
        │ {run_id, status:"queued", ...}        │ JobManager.submit(req)
        └──────────────────────────────────────╴│
                                                ▼
                          ┌─────────────────────────────────────┐
                          │  JobManager (jobs.py)               │
                          │                                     │
                          │  • new run_id                       │
                          │  • mkdir runs/<run_id>/             │
                          │  • write request.json               │
                          │  • create JobRecord(status=queued)  │
                          │  • pending.put_nowait(run_id)       │
                          │       │                             │
                          │       ▼                             │
                          │  ┌─────────────────────────────┐    │
                          │  │  worker_loop (asyncio task) │    │
                          │  │  awaits pending.get()       │    │
                          │  └────────────┬────────────────┘    │
                          └───────────────┼─────────────────────┘
                                          │ run_in_executor(executor, ...)
                                          ▼
                          ┌─────────────────────────────────────┐
                          │  ThreadPoolExecutor(max_workers=1)  │
                          │       │                             │
                          │       ▼                             │
                          │  ┌─────────────────────────────┐    │
                          │  │  execute_run_sync (runner)  │    │
                          │  │  • snapshot globals         │    │
                          │  │  • patch globals            │    │
                          │  │  • redirect_stdout(writer)  │    │
                          │  │  • harness.run_harness(...) │    │
                          │  │  • write trace_json,        │    │
                          │  │    report_md (or error_json)│    │
                          │  │  • restore globals          │    │
                          │  │  • push sentinel to queue   │    │
                          │  └─────────────────────────────┘    │
                          └─────────────────────────────────────┘
                                          │
                                          │ (returns)
                                          ▼
                                JobRecord mutated:
                                  status = "succeeded" | "failed"
                                  result_summary populated
                                  finished_at = now

┌──────────────┐    GET /runs/{id}     ┌────────────────┐
│  HTTP client │─────────────────────→ │ route handler  │ ──→ JobManager.get(id)
│  (polling)   │ 200 + JobView         │ (routes.py)    │
└──────────────┘ ◀──────────────────── └────────────────┘
```

Two threads of execution are visible above:

* **The asyncio event loop** runs `main.py`, route handlers, and the
  `worker_loop` coroutine. It never blocks: all blocking work is
  punted to the executor via `loop.run_in_executor`.
* **The single executor thread** runs `execute_run_sync`. This is
  where `harness.run_harness` and all its blocking HTTP calls to the
  Anthropic API actually execute.

Crucially, the two threads communicate only through `JobRecord` field
mutations (which are safe for single-writer/single-reader without locks
because Python's GIL makes individual attribute writes atomic) and the
`log_queue` (thread-safe by construction).

## 4. Execution model — why single-worker

```
                ┌──────────────────────────────────────┐
                │  asyncio event loop (one thread)     │
                │                                      │
                │   • FastAPI request handlers         │
                │   • lifespan startup/shutdown        │
                │   • worker_loop coroutine            │
                │     (awaits pending queue)           │
                └──────────────────┬───────────────────┘
                                   │
                                   │ run_in_executor(...)
                                   ▼
                ┌──────────────────────────────────────┐
                │  ThreadPoolExecutor(max_workers=1)   │
                │  "harness-run" thread (one thread)   │
                │                                      │
                │   • execute_run_sync runs here       │
                │   • blocking Anthropic SDK calls     │
                │   • blocking yfinance calls          │
                │   • sys.stdout redirect lives here   │
                └──────────────────────────────────────┘
```

Why exactly one worker thread?

* **`api.MODEL` / `PLANNER_MODEL` / `ADVISOR_MODEL` and
  `harness.MAX_ITERATIONS` are mutable module globals.** If two runs
  with different model choices ran concurrently in the same process,
  they would race on these patches. Serialising at the executor level
  makes the race impossible.
* **`contextlib.redirect_stdout` modifies `sys.stdout` process-wide.**
  Two simultaneous runs both redirecting stdout to their own
  `QueueWriter` would clobber each other. Single-worker means at most
  one redirect is active at a time.
* **The harness is I/O-bound, not CPU-bound.** Parallelism would
  reduce wall-clock per-call only if the bottleneck were CPU; for
  network-bound work, a second worker doesn't help your own
  request — it only helps a *different* user's request, which isn't a
  research-phase requirement.

The unblocking comes from the *queue*: callers can submit while a run
is in flight, and their runs execute as soon as the worker is free.
The asyncio queue caps pending submissions
(`PORTFOLIO_AGENT_QUEUE_MAX`, default 8); `POST /runs` returns 429
when the queue is full.

To replace single-worker with real concurrency (Phase 3), the globals
problem must be solved first: thread `model` and `max_iterations`
through `run_harness` and `call_claude` as parameters. The stdout
problem is harder — would need a per-thread `sys.stdout` swap or a
proper structured-event refactor of the harness.

## 5. Stdout capture — how `print()` reaches the client

`harness.py`, `agents.py`, `api.py`, `pricing.py`, and `tools.py` all
use `print()` to surface progress, retry warnings, tool calls, and
scores. The server captures every line three ways at once.

```
                    print("Iteration 1 result")
                              │
                              ▼
                  ┌───────────────────────┐
                  │  sys.stdout           │  (redirected for the run)
                  │  → QueueWriter        │
                  └───────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
      ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐
      │ stdout.log  │ │ real stdout │ │ log_queue       │
      │ (per-run    │ │ (terminal   │ │ (line-buffered, │
      │  file)      │ │  mirror)    │ │  for SSE Phase  │
      │             │ │             │ │  2 consumers)   │
      └─────────────┘ └─────────────┘ └─────────────────┘
        raw chunks      raw chunks      one item per
                                        completed line
```

`QueueWriter` ([runner.py](runner.py) lines ~57-130) is a
`io.TextIOBase` subclass with three sinks:

1. **`file_handle`** — opened on `runs/<run_id>/stdout.log` in
   line-buffered text mode. Receives every raw write so `tail -f`
   shows live output.
2. **`mirror`** — captured `sys.__stdout__` at module-load time
   (before any redirect ever happens). Receives every raw write so
   the server's own terminal keeps printing the harness output, even
   though `sys.stdout` is currently pointing at the QueueWriter.
3. **`q`** — a `queue.Queue()` on the `JobRecord`. Receives one
   string per *completed line* (newline-buffered). This is the input
   for Phase 2's SSE endpoint; in Phase 1 it has no consumer and
   simply fills until the job is deleted.

The line buffering on the queue matters because `print("hi")` issues
two writes (`"hi"` and `"\n"`); we want one queue item per logical
line, not one per fragment.

A sentinel `None` is pushed to the queue in the `finally` block of
`execute_run_sync` to signal "stream is complete" to any consumer
draining it.

**Implication for the operator:** the server's terminal keeps showing
harness output during runs, regardless of whether any HTTP client is
listening. This is deliberate; the `mirror` sink is independent of the
queue sink. If you want a silent server, drop the mirror argument when
constructing the `QueueWriter`.

## 6. Global state management — snapshot and restore

The CLI's `__main__` block in `harness.py` patches `api.MODEL`,
`api.PLANNER_MODEL`, `api.ADVISOR_MODEL`, and `harness.MAX_ITERATIONS`
before calling `run_harness`. The server does the same, but with
careful bookkeeping so that:

* The patches are applied **per run**, based on the request body.
* The patches are **restored after every run**, even on exception.
* Two consecutive runs with different model choices don't bleed into
  each other.

[runner.py:execute_run_sync](runner.py) implements this:

```python
# 1. Snapshot
orig_model    = api.MODEL
orig_planner  = api.PLANNER_MODEL
orig_advisor  = api.ADVISOR_MODEL
orig_max_iter = harness.MAX_ITERATIONS

# 2. Patch (only when the request asks for an override)
override_model = _resolve_model_patches(req)  # None → no patching
if override_model is not None:
    api.MODEL = api.PLANNER_MODEL = api.ADVISOR_MODEL = override_model
if iterations is not None:
    harness.MAX_ITERATIONS = iterations

try:
    # 3. Run harness with redirected stdout
    ...
finally:
    # 4. Restore — runs even if harness raised
    api.MODEL = orig_model
    api.PLANNER_MODEL = orig_planner
    api.ADVISOR_MODEL = orig_advisor
    harness.MAX_ITERATIONS = orig_max_iter
```

The single-worker executor guarantees that no other `execute_run_sync`
is observing the patched globals during this run.

The CLI and the server are **separate Python processes**, so a
manually-invoked `python harness.py` in another terminal has its own
copy of these globals and never interacts with the server's process.
This is why CLI compatibility is "free."

## 7. On-disk artifact layout

Each run gets its own directory under the resolved runs-root
(`PORTFOLIO_AGENT_RUNS_DIR`, default `./runs/`):

```
runs/
└── 2026-05-29T01-57-51Z_9fec8913/
    ├── request.json          # written on POST /runs (before queueing)
    ├── stdout.log            # captured during run, live-tailable
    ├── harness_output.json   # written on success (== CLI's harness_output.json)
    ├── harness_output.md     # written on success (== CLI's harness_output.md)
    └── error.json            # written on failure (message + traceback)
```

Properties of this layout:

* **Sortable, unique IDs.** Format is
  `<ISO-8601-UTC>_<8-hex-uuid>`. ISO-8601 sorts lexicographically =
  chronologically. The 8-hex suffix keeps two runs in the same second
  distinguishable. Colons in the timestamp are replaced with hyphens
  so the path is filesystem-safe on Windows and shell-safe everywhere.
* **No clobbering.** Each run has its own directory, so concurrent (or
  rapid sequential) runs never overwrite each other's outputs. This
  is the main reason the server doesn't write to `harness_output.json`
  in the CWD like the CLI does.
* **`request.json` is written *before* the run is queued.** If the
  server crashes between submit and pickup, the per-run directory
  still exists with the request body intact — the run can be
  manually re-submitted from disk.
* **`error.json` is only present on failure.** Its absence (with
  status=failed) indicates the run crashed before the error-writing
  code ran — look at `stdout.log` for the traceback.
* **No automatic cleanup.** Disk fills until the operator runs
  `DELETE /runs/{id}` (which removes both registry entry and on-disk
  dir, unless `?keep_files=true`) or manually `rm -rf`s old
  directories. A retention policy is Phase 3 work.

The mapping from "artifact key" (used in `GET /runs/{id}` response and
in routing) to filename lives in
[storage.py:ARTIFACT_NAMES](storage.py).

## 8. Endpoint reference

| Method | Path | Returns | Notes |
|---|---|---|---|
| `POST`   | `/runs` | `202` + `RunCreated` | Body: `RunRequest`. Returns immediately with `run_id`. `429` if queue is full. |
| `GET`    | `/runs` | `200` + `list[JobView]` | Newest first by `created_at`. |
| `GET`    | `/runs/{id}` | `200` + `JobView` | `404` if id unknown. Includes `status`, `result_summary` (when succeeded), `error` (when failed). |
| `DELETE` | `/runs/{id}` | `200` | Removes registry entry and `rm -rf`s the dir. `?keep_files=true` to preserve the dir. `409` if status is `running`. |
| `GET`    | `/runs/{id}/trace` | `200` + `harness_output.json` | `404` until the run succeeds. |
| `GET`    | `/runs/{id}/report` | `200` + `harness_output.md` | Same. |
| `GET`    | `/runs/{id}/stdout` | `200` + `stdout.log` | Available from the moment the run starts. |
| `GET`    | `/runs/{id}/artifacts/{name}` | `200` + that file | Generic per-run file fetch. Defensive against `..` traversal and leading-dot names. |
| `GET`    | `/healthz` | `200` + `HealthResponse` | `{ok, anthropic_key_present, runs_dir, queue_max, active_runs, pending_runs}`. |

Auto-generated docs at:

* `GET /docs` — Swagger UI (interactive try-it-out forms)
* `GET /redoc` — ReDoc (read-only reference)
* `GET /openapi.json` — raw OpenAPI 3.1 spec

## 9. Configuration

Read at server startup (lifespan), never re-read at runtime:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — required — | Read by `api.py` at import. Server refuses to start without it. |
| `PORTFOLIO_AGENT_RUNS_DIR` | `./runs` | Where per-run directories live. Created if missing. |
| `PORTFOLIO_AGENT_QUEUE_MAX` | `8` | Max pending runs before `POST /runs` returns 429. |

No config file. No `--config` flag. Add env vars in your shell or in
a `.env` (sourced before launch — see `SESSION_NOTES.md`).

## 10. Extension points

### Phase 2 — SSE live progress

Already plumbed: `JobRecord.log_queue` is being populated by the
`QueueWriter` right now, with no consumer. Phase 2 adds a single new
route:

```python
@router.get("/runs/{run_id}/events")
async def stream_events(run_id, manager): ...
```

The handler drains `job.log_queue` and yields one SSE event per line
via `sse-starlette`'s `EventSourceResponse`. Stops on the sentinel
`None`. No harness changes, no runner changes.

Open questions for Phase 2:

* **Multi-subscriber fan-out.** `queue.Queue` is single-consumer. For
  multiple concurrent SSE listeners on the same run, the JobManager
  would need a list of subscriber queues and the `QueueWriter` would
  broadcast.
* **Replay buffer.** If a client connects mid-run, do they get only
  new events, or backfill from the start? Backfill needs a `deque` of
  history alongside (or instead of) the queue.

### Phase 3 — operability

* **Auth.** `X-API-Key` header gate via a FastAPI dependency. Single
  shared secret in an env var is enough for local use.
* **Persistence.** SQLite for the registry (so it survives restart).
  Or: rebuild the registry on startup by scanning `runs/` and
  re-hydrating `JobRecord`s from `request.json` + the present
  artifact files. The second option keeps the "no DB" stance.
* **Retention.** Background task that purges runs older than N days.
* **Real concurrency.** Refactor `api.MODEL` etc. into parameters
  threaded through `run_harness` → `call_claude`. Once globals are
  gone, `max_workers` can be raised and the asyncio queue can drain
  in parallel.

## 11. Known limitations

* **Registry is lost on restart.** Artifacts on disk survive, but
  `GET /runs/{id}` returns 404 until the registry is rebuilt (Phase 3).
* **No auth.** Anyone with network access to the listening port can
  POST runs and read every run's artifacts. Bind to `127.0.0.1` only
  unless you've added auth.
* **No request size limits beyond FastAPI defaults.** A very large
  `goal` string is technically possible.
* **No per-run timeout.** A harness call that hangs ties up the worker
  indefinitely. Anthropic SDK's own timeouts will eventually fire, but
  there is no server-side wall-clock cap.
* **`print()` from request handlers during a run goes into the run's
  stdout.log.** `redirect_stdout` is process-global. In practice the
  route handlers don't `print()`, and uvicorn's access logs go through
  the `logging` module (not `print`), so this is theoretical — but
  worth knowing if you add `print` debugging to a handler.
