"""Writes the agent resource table in `docs/reports/README.md` ("Agent resource use") from the runs
root, so the table is a function of the runs and is regenerated, never assembled by hand:

    python -m evaluation.table [--write]

One row per task and arm. The run is the latest with `evaluation/evaluation.json` (scored); without
one, the latest with a session (`not scored`); without a session, `not run`. Verdict columns
come from `evaluation.json`; token and tool-call columns from the session files through
the Claude Code or Codex readers in `transcript`, cut at `launch.json` `ended` where set; sessions are
found and read by the same functions the scorer uses. A Codex session records no per-response usage, so
its responses, cost and peak context cells are empty (the cost weights are also Anthropic's). Task directories starting with `_` are smoke tasks and left out.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .prepare import REPO_ROOT, RUNS_ROOT_ENV_VAR
from .settings import path_setting
from .scorer import _ended, _sessions
from .transcript import accesses, codex_accesses, codex_usage, usage

README = REPO_ROOT / "docs" / "reports" / "README.md"
BEGIN, END = "<!-- agent-resource-table:begin -->", "<!-- agent-resource-table:end -->"
ARMS = ("plain", "skill")
# Listed only for tasks that have such a run: it is a pilot arm, not run on every task.
OPTIONAL_ARMS = ("builder-skill",)
# Token cost relative to uncached input (Anthropic's price ratios), for the wait share.
WEIGHTS = {"input_tokens": 1, "cache_creation_input_tokens": 1.25, "cache_read_input_tokens": 0.1, "output_tokens": 5}
HEADER = ("| Task | Arm | Run | `end_to_end_success` | Human messages | Executions (failed) | Unapproved departures "
          "| Tool calls | Responses (excl. wait) | Output | Cost | Cost excl. wait | Peak context | Wall time |")


def _count(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.1f}K" if n >= 1e3 else str(n)


def _pick(task_dir: Path, arm: str) -> tuple[Path | None, str]:
    runs = sorted(task_dir.glob(f"{arm}-*"), reverse=True)
    for run in runs:
        if (run / "evaluation" / "evaluation.json").is_file():
            return run, "scored"
    for run in runs:
        if (run / "launch.json").is_file() and _launched(run)[2]:
            return run, "not scored"
    return None, "not run (prepared)" if runs else "not run"


def _launched(run: Path) -> tuple[dict, str, list[Path]]:
    """The run's launch record, harness, and session files."""
    launch = json.loads((run / "launch.json").read_text())
    harness, sessions, _ = _sessions(launch, Path(launch["workspace"]))
    return launch, harness, sessions


def _row(task: str, arm: str, run: Path | None, state: str) -> str:
    if run is None:
        return f"| {task} | {arm} | {state} |" + " |" * 11
    launch, harness, sessions = _launched(run)
    codex, ended = harness == "codex", _ended(launch)
    u = (codex_usage if codex else usage)(sessions, ended) or {}
    if state == "scored":
        e = json.loads((run / "evaluation" / "evaluation.json").read_text())
        success = {True: "yes", False: "no", None: "not recorded"}[e.get("end_to_end_success")]
        executions = f"{e['executions']} ({e['failed_executions']})" if e.get("executions") is not None else ""
        departures = (e.get("route_departures") or {}).get("unapproved", "not judged")
        verdicts = [success, e.get("human_interventions", ""), executions, departures]
    else:
        verdicts = [state, "", "", "not judged"]
    total = sum(u.get(k, 0) * w for k, w in WEIGHTS.items())
    wait = sum(u.get(f"wait_{k}", 0) * w for k, w in WEIGHTS.items())
    wall = u.get("wall_seconds")
    responses = "" if codex else f"{u.get('responses', 0)} ({u.get('responses', 0) - u.get('wait_responses', 0)})"
    costs = ["", ""] if codex else [_count(total), _count(total - wait)]
    cells = [len((codex_accesses if codex else accesses)(sessions, ended)), responses,
             _count(u.get("output_tokens", 0)), *costs, "" if codex else _count(u.get("peak_context", 0)),
             f"{wall // 3600}h{wall % 3600 // 60:02d}" if wall is not None else ""]
    return "| " + " | ".join(str(c) for c in [task, arm, f"`{run.name}`", *verdicts, *cells]) + " |"


def table(runs_root: Path) -> str:
    rows = [HEADER, "|" + "---|" * HEADER.count(" | ") + "---|"]
    for task_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        rows += [_row(task_dir.name, arm, *_pick(task_dir, arm)) for arm in ARMS]
        for arm in OPTIONAL_ARMS:
            run, state = _pick(task_dir, arm)
            if run is not None:
                rows.append(_row(task_dir.name, arm, run, state))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help=f"replace the table between the markers in {README.name}")
    args = parser.parse_args()
    runs_root = path_setting(RUNS_ROOT_ENV_VAR, 2)
    if runs_root is None:
        raise SystemExit(f"{RUNS_ROOT_ENV_VAR} is not set and local preferences section 2 is unanswered: "
                         "run python -m evaluation.setup")
    text = table(runs_root)
    if not args.write:
        print(text)
        return
    readme = README.read_text()
    if readme.count(BEGIN) != 1 or readme.count(END) != 1:
        raise SystemExit(f"{README} needs exactly one {BEGIN} and one {END}")
    head, rest = readme.split(BEGIN)
    README.write_text(f"{head}{BEGIN}\n{text}\n{END}{rest.split(END, 1)[1]}")


if __name__ == "__main__":
    main()
