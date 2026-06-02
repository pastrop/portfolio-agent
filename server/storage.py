"""
On-disk layout for run artifacts.

Each run gets its own directory under the resolved runs-root (default
``./runs/``, overridable via ``PORTFOLIO_AGENT_RUNS_DIR``).  No
clobbering between runs, easy to grep / archive / delete.

Per-run directory contents:

  runs/<run_id>/
    ├── request.json          POST body that started the run
    ├── stdout.log            full captured stdout (live during run)
    ├── harness_output.json   == CLI's harness_output.json
    ├── harness_output.md     == CLI's harness_output.md
    └── error.json            present only on failure (message + traceback)
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


# Canonical artifact filenames.  Routes use the keys; on-disk uses the values.
ARTIFACT_NAMES: dict[str, str] = {
    "request_json": "request.json",
    "stdout_log": "stdout.log",
    "trace_json": "harness_output.json",
    "report_md": "harness_output.md",
    "error_json": "error.json",
}


def get_runs_dir() -> Path:
    """
    Resolve the runs-root directory.  Created if missing.  Defaults to
    ``./runs/`` relative to the process CWD.
    """
    raw = os.environ.get("PORTFOLIO_AGENT_RUNS_DIR", "./runs")
    p = Path(raw).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_run_id() -> str:
    """
    Generate a sortable, unique, filesystem-safe run id.

    Format: ``YYYY-MM-DDTHH-MM-SSZ_<8-hex>``.  Colons are replaced with
    hyphens (Windows-safe, shell-safe), and the 8-hex suffix keeps two
    runs created in the same second distinguishable.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{ts}_{suffix}"


def make_run_dir(runs_root: Path, run_id: str) -> Path:
    """Create and return the per-run directory under ``runs_root``."""
    d = runs_root / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
