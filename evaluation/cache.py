"""The deployment's cache directory: raw inputs, the package cache and benchmark source clones.

Task records name cache paths as `.cache/...`. Rendered prompts and bundle manifests hold PLACEHOLDER in place of
the directory, so a bundle and its digest are the same on every machine; `evaluation.prepare` fills it in and
records the value in `launch.json`, which the scorer reads.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT_ENV_VAR = "REPRODUCTION_CACHE_DIR"
PLACEHOLDER = "{{CACHE_ROOT}}"
RECORD_PREFIX = ".cache"


def resolve_cache_root() -> Path:
    """`REPRODUCTION_CACHE_DIR`, or this repository's `.cache`. The path is limited to characters the access
    monitor can read back from a transcript."""
    value = os.environ.get(CACHE_ROOT_ENV_VAR)
    root = (Path(value).expanduser() if value else REPO_ROOT / RECORD_PREFIX).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_./+@~-]+", str(root)):
        raise SystemExit(f"cache directory {root} may hold only letters, digits and _ . / + @ ~ -")
    return root


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
