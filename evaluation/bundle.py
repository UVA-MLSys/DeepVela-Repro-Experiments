"""Reads a rubric bundle into an `EvaluationProfile` and one task's
`EvaluationTask`. Layout and digest recipe: `skills/agent-evaluation-profile/references/rubric-authoring.md`.

The phase list and its rubric-template membership live only in the bundle's own `bundle.yaml`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .cache import INPUT_ROOT, PLACEHOLDER, PLACEHOLDERS, fill_text, placeholder_values
from .schema import EvaluationProfile, EvaluationTask, PhaseRubric

BUNDLE_ROOT_ENV_VAR = "EVALUATION_BUNDLE_DIR"


class EvaluationBundleError(ValueError):
    """The bundle cannot be read into a usable profile or task: a missing or malformed file, a
    phase naming a rubric template no file provides, or a rubric file declaring a different
    rubric_id/phase_id than the one it is listed under."""


def resolve_bundle_root() -> Path:
    """The deployment's bundle root, from `EVALUATION_BUNDLE_DIR`. A relative value resolves from the
    working directory, the repository root when the skill runs the command. There is no default: the
    answer is the deployment's, recorded in `docs/local-preferences.md`."""
    value = os.environ.get(BUNDLE_ROOT_ENV_VAR)
    if not value:
        raise EvaluationBundleError(f"{BUNDLE_ROOT_ENV_VAR} is not set: take it from "
                                    "docs/local-preferences.md, section 1")
    return Path(value).expanduser().resolve()


def check_id(kind: str, value: str) -> None:
    """A profile or task ID is `/`-separated names of letters, digits, `.`, `_` and `-`, so it names a folder under
    the bundle and runs roots and stays safe inside a shell command."""
    if not all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]*", part) for part in value.split("/")):
        raise EvaluationBundleError(f"{kind} {value!r} must be '/'-separated names of letters, digits, '.', '_', '-'")


def profile_dir(profile_id: str) -> Path:
    """The folder holding `bundle.yaml`: the bundle root itself, or `<bundle root>/<profile_id>` when the
    root holds several profiles."""
    check_id("profile_id", profile_id)
    root = resolve_bundle_root()
    return root if (root / "bundle.yaml").is_file() else root / profile_id


def task_dir(profile: Path, task_id: str) -> Path:
    """A task's folder: `<profile>/<task_id>`, or `<profile>/tasks/<task_id>` in bundles laid out that way."""
    check_id("task_id", task_id)
    nested = profile / "tasks" / task_id
    return nested if nested.is_dir() else profile / task_id


def load_yaml(path: Path, *, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationBundleError(f"{what} does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise EvaluationBundleError(f"{what} at {path} is not valid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationBundleError(f"{what} at {path} must be a YAML mapping")
    return payload


def bundle_digest(profile: Path, task_id: str) -> str:
    """sha256 of the `MANIFEST.sha256` lines that cover one task: `bundle.yaml`, `checkers/`, `rubrics/`
    and the task's folder. Another task's files are outside the scope, so adding or editing a task
    never changes the digest of a run against this one. Raises if a file in scope is missing from the
    manifest, listed but absent, or no longer matches its listed hash, so an edit that did not
    regenerate the manifest (rubric-authoring.md) cannot return the old digest."""
    manifest = profile / "MANIFEST.sha256"
    if not manifest.is_file():
        raise EvaluationBundleError(f"{manifest} does not exist -- run the digest step in rubric-authoring.md")
    task = task_dir(profile, task_id)
    task_prefix = task.relative_to(profile).as_posix() + "/"
    prefixes = ("checkers/", "rubrics/", task_prefix)
    in_scope = lambda rel: rel == "bundle.yaml" or rel.startswith(prefixes)
    canonical = f"tasks/{task_id}/"  # the digest names task files this way in every layout, so moving a task keeps it
    lines: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, _, relative = line.partition("  ")
        relative = relative.removeprefix("./")
        if in_scope(relative):
            if task_prefix != canonical and relative.startswith(task_prefix):
                line = f"{digest}  ./{canonical}{relative.removeprefix(task_prefix)}"
            lines[relative] = line
    listed = {rel: line.partition("  ")[0] for rel, line in lines.items()}
    scoped = [profile / "bundle.yaml", profile / "checkers", profile / "rubrics", task]
    on_disk = {p.relative_to(profile).as_posix() for s in scoped
               for p in ([s] if s.is_file() else s.rglob("*") if s.is_dir() else []) if p.is_file()}
    unlisted, missing = sorted(on_disk - listed.keys()), sorted(listed.keys() - on_disk)
    changed = sorted(r for r in on_disk & listed.keys() if hashlib.sha256((profile / r).read_bytes()).hexdigest() != listed[r])
    if unlisted or missing or changed:
        raise EvaluationBundleError(f"{manifest} does not match task {task_id!r}: unlisted {unlisted}, missing {missing}, changed {changed}")
    ordered = sorted(lines, key=lambda r: canonical + r.removeprefix(task_prefix) if r.startswith(task_prefix) else r)
    return hashlib.sha256(("\n".join(lines[r] for r in ordered) + "\n").encode()).hexdigest()


def load_profile(profile_id: str) -> EvaluationProfile:
    folder = profile_dir(profile_id)
    bundle_yaml = load_yaml(folder / "bundle.yaml", what=f"bundle.yaml for {profile_id!r}")
    if bundle_yaml.get("profile_id") != profile_id:
        raise EvaluationBundleError(f"{folder}/bundle.yaml declares profile_id {bundle_yaml.get('profile_id')!r}, expected {profile_id!r}")
    phases = bundle_yaml.get("phases")
    if not isinstance(phases, list) or not phases:
        raise EvaluationBundleError(f"{folder}/bundle.yaml is missing a non-empty 'phases' list")
    try:
        return EvaluationProfile(
            profile_id=profile_id,
            profile_version=bundle_yaml.get("profile_version"),
            phases=[{"id": p.get("id"), "label": p.get("label"), "rubric_templates": p.get("templates")} for p in phases],
        )
    except ValidationError as exc:
        raise EvaluationBundleError(f"{folder}/bundle.yaml does not match EvaluationProfile: {exc}") from exc


def _load_target_rubrics(profile: EvaluationProfile, task_dir: Path, manifest: dict[str, Any]) -> list[PhaseRubric]:
    """A multi-target task carries its own rubric files, one per `<rubric_id>.yaml` under the task's
    `rubrics/`, each naming a profile phase and a target (null for a shared rubric). A target's shared
    rubrics are the target-less rubrics in the `prerequisites` closure of its own rubrics; the
    manifest's `shared_rubrics` must list exactly those."""
    phase_ids = {p.id for p in profile.phases}
    rubrics = []
    for path in sorted((task_dir / "rubrics").glob("*.yaml")):
        rubric = PhaseRubric.model_validate(load_yaml(path, what=f"rubric file {path.name}"))
        if rubric.rubric_id != path.stem or rubric.phase_id not in phase_ids:
            raise EvaluationBundleError(f"{path} declares {rubric.rubric_id!r}/{rubric.phase_id!r}: the id must match the file name and the phase must be one of {sorted(phase_ids)}")
        if rubric.target is not None and rubric.target not in manifest["targets"]:
            raise EvaluationBundleError(f"{path} names target {rubric.target!r}, not one of {sorted(manifest['targets'])}")
        rubrics.append(rubric)
    by_id = {r.rubric_id: r for r in rubrics}
    for target, spec in manifest["targets"].items():
        closure = rubric_closure(rubrics, target)
        quality_id = spec.get("quality_rubric_id")
        quality = by_id.get(quality_id)
        if quality is None or quality.target != target:
            owner = None if quality is None else quality.target
            raise EvaluationBundleError(
                f"target {target!r}: quality_rubric_id {quality_id!r} belongs to {owner!r}, not this target"
            )
        unknown = sorted({q for r in closure for q in r.prerequisites} - by_id.keys())
        foreign = sorted((r.rubric_id, r.target) for r in closure if r.target not in (None, target))
        shared = sorted(r.rubric_id for r in closure if r.target is None)
        if unknown or foreign or shared != sorted(spec.get("shared_rubrics", [])):
            raise EvaluationBundleError(
                f"target {target!r}: shared_rubrics {sorted(spec.get('shared_rubrics', []))} differ "
                f"from the prerequisites closure {shared}; unknown prerequisites {unknown}; "
                f"foreign target rubrics {foreign}"
            )
    return rubrics


def rubric_closure(rubrics: list[PhaseRubric], target: str) -> list[PhaseRubric]:
    """The target's own rubrics plus every rubric they depend on, transitively, through `prerequisites`."""
    by_id = {r.rubric_id: r for r in rubrics}
    wanted = {r.rubric_id for r in rubrics if r.target == target}
    stack = list(wanted)
    while stack:
        for q in by_id[stack.pop()].prerequisites:
            if q in by_id and q not in wanted:
                wanted.add(q)
                stack.append(q)
    return [r for r in rubrics if r.rubric_id in wanted]


def fill_cache_root(manifest: dict[str, Any], cache_root: str) -> dict[str, Any]:
    """The manifest with its placeholders replaced: literally in paths, escaped in the `forbidden_access` regexes."""
    values = placeholder_values(cache_root, manifest.get("input_set_sha256"))
    escaped = {placeholder: re.escape(path) for placeholder, path in values.items()}
    def fill(value: Any, replacements: dict[str, str]) -> Any:
        if isinstance(value, str):
            return fill_text(value, replacements)
        if isinstance(value, list):
            return [fill(v, replacements) for v in value]
        if isinstance(value, dict):
            return {k: fill(v, replacements) for k, v in value.items()}
        return value
    return {k: fill(v, escaped if k == "forbidden_access" else values) for k, v in manifest.items()}


def load_task(profile: EvaluationProfile, task_id: str, cache_root: str | None = None) -> EvaluationTask:
    """Assemble an `EvaluationTask` from the task's `manifest.yaml` plus each phase's rubric
    files. A template promoted to the profile-wide `rubrics/` wins over the task's own copy.
    `cache_root` fills a portable manifest; a run's is the one its `launch.json` records."""
    folder = profile_dir(profile.profile_id)
    task_folder = task_dir(folder, task_id)
    manifest_path = task_folder / "manifest.yaml"
    manifest = load_yaml(manifest_path, what=f"task manifest for {task_id!r}")
    if cache_root is not None:
        manifest = fill_cache_root(manifest, cache_root)
    if any(p in yaml.safe_dump(manifest, width=10**9) for p in PLACEHOLDERS):
        raise EvaluationBundleError(f"{manifest_path} holds an unfilled placeholder: pass the cache directory the run was "
                                    f"prepared with; {INPUT_ROOT} also needs input_set_sha256")
    if manifest.get("profile_id") != profile.profile_id or manifest.get("profile_version") != profile.profile_version:
        raise EvaluationBundleError(f"{manifest_path} names profile {manifest.get('profile_id')!r} v{manifest.get('profile_version')}, expected {profile.profile_id!r} v{profile.profile_version}")

    phase_rubrics: list[PhaseRubric] = []
    if manifest.get("targets"):
        phase_rubrics = _load_target_rubrics(profile, task_folder, manifest)
    for phase in [] if phase_rubrics else profile.phases:
        for template_id in phase.rubric_templates:
            promoted = folder / "rubrics" / f"{template_id}.yaml"
            rubric_path = promoted if promoted.is_file() else task_folder / "rubrics" / f"{template_id}.yaml"
            rubric = load_yaml(rubric_path, what=f"rubric {template_id!r} for phase {phase.id!r}")
            if rubric.get("rubric_id") != template_id or rubric.get("phase_id") != phase.id:
                raise EvaluationBundleError(f"{rubric_path} declares {rubric.get('rubric_id')!r}/{rubric.get('phase_id')!r}, expected {template_id!r}/{phase.id!r}")
            phase_rubrics.append(PhaseRubric.model_validate(rubric))

    manifest_fields = {k: v for k, v in manifest.items() if k not in {"profile_id", "profile_version"}}
    try:
        return EvaluationTask.model_validate({
            **manifest_fields,
            "evaluation_profile_id": profile.profile_id,
            "evaluation_profile_version": profile.profile_version,
            "task_family": manifest.get("task_family", profile.profile_id),
            "phases": [r.model_dump() for r in phase_rubrics],
        })
    except ValidationError as exc:
        raise EvaluationBundleError(f"{manifest_path} does not match EvaluationTask: {exc}") from exc


def check_task_manifest(profile_id: str, task_id: str) -> None:
    """The task's manifest.yaml names this profile and task, a dataset record, and that record's commit, which the
    task's prompt also states."""
    from .task_prompt import DATASET_DIR, load as load_record  # needs pydantic models the scorer does not
    folder = task_dir(profile_dir(profile_id), task_id)
    manifest = load_yaml(folder / "manifest.yaml", what=f"task manifest for {task_id!r}")
    for key, expected in (("task_id", task_id), ("profile_id", profile_id)):
        if manifest.get(key) != expected:
            raise EvaluationBundleError(f"{folder}/manifest.yaml names {key} {manifest.get(key)!r}, expected {expected!r}")
    source_id = manifest.get("source_id")
    if not source_id or not (DATASET_DIR / f"{source_id}.json").is_file():
        raise EvaluationBundleError(f"{folder}/manifest.yaml names source_id {source_id!r}, which has no dataset record")
    commit = load_record(source_id).repo["commit"]
    if manifest.get("source_revision") != commit:
        raise EvaluationBundleError(f"{folder}/manifest.yaml names source_revision {manifest.get('source_revision')!r}; the record pins {commit}")
    prompt = folder / "prompt.md"
    if not prompt.is_file() or commit not in prompt.read_text():
        raise EvaluationBundleError(f"{prompt} does not state commit {commit}: render it again from the record")
    text = prompt.read_text()
    if PLACEHOLDER in text or INPUT_ROOT in text:
        check_cache_inputs(manifest, text, folder)
    if INPUT_ROOT in text:
        check_input_set(manifest, source_id, task_id, folder)


def check_input_set(manifest: dict[str, Any], source_id: str, task_id: str, folder: Path) -> None:
    """The manifest pins the set the source's input spec gives this task, the last part of its id."""
    from .warmup import load_spec
    selector = task_id.rsplit("/", 1)[-1]
    expected = [s["manifest_sha256"] for s in load_spec(source_id)["sets"].values() if selector in s["tasks"]]
    if expected != [manifest.get("input_set_sha256")]:
        raise EvaluationBundleError(f"{folder}/manifest.yaml names input_set_sha256 {manifest.get('input_set_sha256')!r}; "
                                    f"the spec of {source_id} gives task {selector!r} {expected}")


def check_cache_inputs(manifest: dict[str, Any], prompt: str, folder: Path) -> None:
    """In a portable task, the manifest's cache paths agree with the inputs the prompt gives, so the access rule
    and the input rubrics cover the paths the agent is told to use."""
    found = re.findall(r"^\w+=(?:\$(CACHE_ROOT|INPUT_ROOT)|\{\{(CACHE_ROOT|INPUT_ROOT)\}\})(/\S+)$", prompt, re.M)
    inputs = ["{{" + (variable or placeholder) + "}}" + path for variable, placeholder, path in found]
    within = lambda path, roots: any(path == r or path.startswith(r + "/") for r in roots)
    allowed = (manifest.get("input_reuse_contract") or {}).get("allowed_cache_roots", [])
    undeclared = [p for p in inputs if not within(p, allowed)]
    unused = [r for r in allowed if r.startswith("{{") and not any(within(p, [r]) or within(r, [p]) for p in inputs)]
    stray = [p for p in (manifest.get("artifact_paths") or {}).values() if p.startswith("{{") and not within(p, inputs)]
    if undeclared or unused or stray:
        raise EvaluationBundleError(f"{folder}: the prompt's inputs and the manifest differ: inputs outside "
                                    f"allowed_cache_roots {undeclared}; allowed_cache_roots the prompt does not give "
                                    f"{unused}; artifact_paths outside the inputs {stray}")


def seal(profile_id: str, task_id: str) -> str:
    """Check the task's manifest, regenerate MANIFEST.sha256 for the whole profile (the recipe in
    references/rubric-authoring.md), and return the task's digest."""
    check_task_manifest(profile_id, task_id)
    folder = profile_dir(profile_id)
    skipped = {"MANIFEST.sha256", "README.md"}
    lines = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.name not in skipped
                       and p.relative_to(folder).parts[0] not in ("runs", "audit")):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.relative_to(folder).as_posix()}\n")
    (folder / "MANIFEST.sha256").write_text("".join(lines))
    return bundle_digest(folder, task_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a task's manifest, write MANIFEST.sha256, print the task's digest.")
    parser.add_argument("action", choices=["seal", "check"])
    parser.add_argument("profile_id")
    parser.add_argument("task_id")
    args = parser.parse_args()
    if args.action == "seal":
        print(seal(args.profile_id, args.task_id))
    else:
        check_task_manifest(args.profile_id, args.task_id)
        print(bundle_digest(profile_dir(args.profile_id), args.task_id))


if __name__ == "__main__":
    main()
