"""The deployment's cache directory: raw inputs, input sets, the package cache and benchmark source clones.

Rendered prompts and bundle manifests hold placeholders in place of its paths, so a bundle and its digest are the
same on every machine; `evaluation.prepare` fills them in and records the directory in `launch.json`, which the
scorer reads. `{{PACKAGE_CACHE}}` is the shared uv cache. A task with an input spec (dataset/inputs/) names its
inputs under `{{INPUT_ROOT}}`, the input set its manifest pins; any other task names `.cache/...` record paths
under `{{CACHE_ROOT}}`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT_ENV_VAR = "REPRODUCTION_CACHE_DIR"
PLACEHOLDER = "{{CACHE_ROOT}}"
INPUT_ROOT = "{{INPUT_ROOT}}"
PACKAGE_CACHE = "{{PACKAGE_CACHE}}"
PLACEHOLDERS = (PLACEHOLDER, INPUT_ROOT, PACKAGE_CACHE)
SET_ID_LENGTH = 16
RECORD_PREFIX = ".cache"


def resolve_cache_root() -> Path:
    """`REPRODUCTION_CACHE_DIR`, or this repository's `.cache`. The path is limited to characters the access
    monitor can read back from a transcript."""
    value = os.environ.get(CACHE_ROOT_ENV_VAR)
    root = (Path(value).expanduser() if value else REPO_ROOT / RECORD_PREFIX).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_./+@~-]+", str(root)):
        raise SystemExit(f"cache directory {root} may hold only letters, digits and _ . / + @ ~ -")
    return root


def input_set_dir(cache_root: Path, manifest_sha256: str) -> Path:
    """Where `evaluation.warmup` builds the input set whose manifest has this sha256."""
    return cache_root / "inputs" / manifest_sha256[:SET_ID_LENGTH]


def placeholder_values(cache_root: str, input_set_sha256: str | None) -> dict[str, str]:
    """Each placeholder's path for a task; INPUT_ROOT only when the task pins an input set."""
    values = {PLACEHOLDER: cache_root, PACKAGE_CACHE: str(Path(cache_root) / "uv")}
    if input_set_sha256:
        values[INPUT_ROOT] = str(input_set_dir(Path(cache_root), input_set_sha256))
    return values


def fill_text(text: str, values: dict[str, str]) -> str:
    for placeholder, path in values.items():
        text = text.replace(placeholder, path)
    return text


def local_path(record_path: str) -> Path:
    """A task record's path on this machine: `.cache/...` is under the cache directory, any other path is
    relative to this repository's root."""
    parts = Path(record_path).parts
    if parts[:1] == (RECORD_PREFIX,):
        return resolve_cache_root().joinpath(*parts[1:])
    return REPO_ROOT / record_path


def portable_path(record_path: str) -> str:
    """A task record's path as a rendered prompt names it."""
    parts = Path(record_path).parts
    if parts[:1] == (RECORD_PREFIX,):
        return "/".join((PLACEHOLDER, *parts[1:]))
    return str(REPO_ROOT / record_path)
