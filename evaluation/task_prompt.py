"""Renders a task prompt from a dataset record (docs/dataset-schema.md) and the fixed template, the way
CORE-Bench fills `benchmark_prompts.json` from each task's `task_prompt`. Run by `agent-evaluation-profile`
author mode, step 5:

    python -m evaluation.task_prompt <source_id> [--tasks <id>,<id>] [--out <bundle task dir>/prompt.md]

`--out` also writes `tasks.json` beside the prompt; `evaluation.prepare` copies it into the workspace.

The Compute paragraph is not part of it: `evaluation.prepare --compute` appends it at launch.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from string import Template

from pydantic import BaseModel, ConfigDict

from .cache import PLACEHOLDER, REPO_ROOT, local_path, portable_path

DATASET_DIR = REPO_ROOT / "dataset" / "tasks"
TEMPLATE = REPO_ROOT / "skills" / "agent-evaluation-profile" / "assets" / "task_prompt.md"
# docs/local-preferences.md, section 5: the shared uv cache in the cache directory.
UV_CACHE = f"{PLACEHOLDER}/uv"
# How the prompt names each tasks.json field; only the fields some selected task has are named.
FIELDS = {"id": "`id`", "task_prompt": "`task_prompt` (what to run)", "results": "`results` (the values to report)",
          "outputs": "`outputs` (files to write to OUTPUT_DIR)",
          "log_lines": "`log_lines` (a line prefix its logs must contain)",
          "inputs": "`inputs` (repository files to fetch yourself)", "platform": "`platform` (overrides the one below)",
          "hints": "`hints`"}


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str  # `.cache/...` in the cache directory, otherwise relative to this repository's root
    holds: str | None = None


class Policy(BaseModel):
    """Source-level answers, given once for every task of the source. A field left out has not been answered."""
    model_config = ConfigDict(extra="forbid")
    compute: str | None = None  # default for evaluation.prepare --compute
    platform: str | None = None
    inputs: dict[str, Input] | None = None  # {} when the agent fetches every input itself
    forbidden: str | None = None
    wall_clock_cap: str | None = None


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    selector_id: str | None = None  # record-local selection key when controlled conditions share one target id
    task_prompt: str  # the entry instruction, as in CORE-Bench and SocSci-Repro-Bench
    results: list[str]  # names the evaluator prints; the values are in the gold file
    outputs: dict[str, str] | None = None  # files ./reproduce.sh writes to OUTPUT_DIR -> what each holds
    platform: str | None = None  # overrides the source's platform for this task
    log_lines: dict[str, str] | None = None  # line prefix -> what follows it
    hints: list[str] | None = None  # help fixed before the run (RExBench hints)
    inputs: list[str] | None = None  # repository-relative input files the agent fetches itself
    input_path_overrides: dict[str, str] | None = None  # named policy inputs whose path differs for this task
    forbidden_override: str | None = None  # condition-specific access restriction for a selected task


class Source(BaseModel):
    model_config = ConfigDict(extra="allow")
    source_id: str
    repo: dict
    support_level: str
    policy: Policy = Policy()
    tasks: list[Task]


def load(source_id: str) -> Source:
    return Source.model_validate(json.loads((DATASET_DIR / f"{source_id}.json").read_text()))


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\\[ \t]*\n", " ", text)).strip()


def undocumented_commands(source: Source, tasks: list[Task]) -> list[str]:
    """Backticked commands in a `task_prompt` that the task directory's docs do not give whole; only a
    `$VAR`/`${VAR}`/`$(VAR)` may stand for another variable. An abbreviated command drops flags that define the route.
    Advisory: a task may cover a combination the docs omit, and then the agent starts from the documented flags."""
    clone = local_path(source.repo["path"])
    directory = clone.resolve() / source.repo.get("directory", "")
    if not directory.is_dir():
        raise SystemExit(f"{source.source_id}: {directory} is missing; clone {source.repo.get('url')} into {clone} "
                         f"and check out {source.repo['commit']}: the task_prompt commands are checked against its docs")
    docs = " ".join(_flat(p.read_text(errors="ignore")) for p in sorted(directory.rglob("*"))
                    if p.is_file() and (p.suffix in (".md", ".yaml", ".yml", ".sh") or p.name == "Makefile")
                    and p.stat().st_size < 2_000_000)
    var = r"\$\{\w+\}|\$\(\w+\)|\$\w+"

    def documented(command: str) -> bool:
        parts = re.split(var, _flat(command))
        pattern = re.compile(r"\S+".join(re.escape(part) for part in parts))
        anchor = max(parts, key=len)  # a plain substring search finds where to try the pattern
        at = docs.find(anchor)
        while at != -1:
            if pattern.search(docs, max(0, at - 4000), at + len(anchor) + 4000):
                return True
            at = docs.find(anchor, at + 1)
        return False

    # A quoted sequence `a; b` is checked line by line: each line must be documented.
    return [f"{t.id}: `{c}`" for t in tasks for c in re.findall(r"`([^`]+)`", t.task_prompt)
            if not all(documented(line) for line in c.split(";") if line.strip())]


def render(source: Source, task_ids: list[str] | None = None) -> tuple[str, str]:
    """The prompt and the agent-visible `tasks.json` (the selected tasks' records, in run order)."""
    tasks = [t for t in source.tasks if task_ids is None or (t.selector_id or t.id) in task_ids]
    if task_ids is not None and len(tasks) != len(task_ids):
        selected = {t.selector_id or t.id for t in tasks}
        raise SystemExit(f"{source.source_id} has no task {sorted(set(task_ids) - selected)}")
    policy = source.policy
    missing = [f"policy.{f}" for f in ("platform", "inputs", "forbidden") if getattr(policy, f) is None]
    missing += [f"{t.id}.outputs" for t in tasks if not t.outputs]
    missing += [f"{t.id}.outputs.{name} (what it holds)" for t in tasks for name, holds in (t.outputs or {}).items() if not holds]
    if missing:
        raise SystemExit(f"{source.source_id}: the dataset record has no answer for {', '.join(missing)}; "
                         f"add it in dataset/tasks/{source.source_id}.json, never in the prompt")

    authored = [policy.platform or ""] + [f"{t.task_prompt} {t.platform or ''}" for t in tasks]
    unresolved = sorted({m for text in authored for m in re.findall(r"<[A-Za-z][^<>\n]*>", text)})
    if unresolved:
        raise SystemExit(f"{source.source_id}: task_prompt or platform holds unresolved {', '.join(unresolved)}; "
                         f"decide it in dataset/tasks/{source.source_id}.json, never in the prompt")

    undocumented = undocumented_commands(source, tasks)
    if undocumented:  # a warning: READMEs omit model x backend x scenario combinations, so the check cannot be a gate
        warnings.warn(f"{source.source_id}: task_prompt quotes a command its documentation does not give whole: "
                      f"{'; '.join(undocumented)}", stacklevel=2)

    overrides = {name: path for task in tasks for name, path in (task.input_path_overrides or {}).items()}
    unknown = sorted(set(overrides) - set(policy.inputs or {}))
    if unknown:
        raise SystemExit(f"{source.source_id}: task input_path_overrides names unknown policy input(s) {', '.join(unknown)}")
    paths = {name: portable_path(overrides.get(name, i.path)) for name, i in (policy.inputs or {}).items()}
    # With several inputs, all in the cache directory, that directory is stated once.
    root = PLACEHOLDER if len(paths) > 1 and all(p.startswith(PLACEHOLDER) for p in paths.values()) else None
    uv_cache = UV_CACHE
    if root:
        paths = {name: p.replace(root, "$CACHE_ROOT", 1) for name, p in paths.items()}
        uv_cache = uv_cache.replace(root, "$CACHE_ROOT", 1)
    python = source.model_extra.get("language") == "Python"  # the uv cache is named only where uv is used
    uv_cache = f"\nShared package cache: UV_CACHE_DIR={uv_cache} (use no other cache location).\n" if python else ""
    if policy.inputs:
        inputs = "Inputs, read-only, do not modify:\n" + (f"CACHE_ROOT={root}\n" if root else "") + "\n".join(
            f"{name}={p}" for name, p in paths.items())
        held = [f"{name} holds {i.holds}." for name, i in policy.inputs.items() if i.holds]
        inputs += ("\n" + " ".join(held)) if held else ""
    else:
        inputs = "Inputs: none provided." + (" Fetch each target's `inputs` yourself." if any(t.inputs for t in tasks) else "")
    records = [t.model_dump(exclude_none=True, exclude={"selector_id", "input_path_overrides", "forbidden_override"}) for t in tasks]
    example = json.dumps({t.id: {r: "<value>" for r in t.results} for t in tasks[:2]})
    time_limit = f"Time limit: {policy.wall_clock_cap} of wall clock.\n" if policy.wall_clock_cap else ""
    forbidden = next((task.forbidden_override for task in tasks if task.forbidden_override), policy.forbidden)
    prompt = Template(TEMPLATE.read_text()).substitute(
        commit=source.repo["commit"], fields=", ".join(FIELDS[f] for f in FIELDS if any(f in r for r in records)),
        platform=policy.platform, inputs=inputs, uv_cache=uv_cache, cache_named=" and the package cache" if python else "",
        report_example=example, forbidden=forbidden, time_limit=time_limit).rstrip("\n") + "\n"
    # The per-target fields go to the agent as a file, as SocSci-Repro-Bench gives papers_tasks/<id>.json.
    return prompt, "[\n" + ",\n".join(json.dumps(r) for r in records) + "\n]\n"  # one entry per line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source_id")
    parser.add_argument("--tasks", help="comma-separated task ids (one env group); default every task of the source")
    parser.add_argument("--out", type=Path, help="write prompt.md here and tasks.json beside it, instead of printing")
    args = parser.parse_args()
    text, tasks = render(load(args.source_id), args.tasks.split(",") if args.tasks else None)
    if args.out:
        args.out.write_text(text)
        (args.out.parent / "tasks.json").write_text(tasks)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
