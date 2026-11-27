"""Reports what this machine has and which deployment answers are missing, for agent-evaluation-profile setup mode.

    python -m evaluation.setup           # report only, changes nothing
    python -m evaluation.setup --init    # also create docs/local-preferences.local.md from the template, if absent
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .cache import input_set_dir, resolve_cache_root
from .settings import LOCAL, REPO_ROOT, TEMPLATE, preferences_file, unanswered


def shown(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def run(*command: str) -> str | None:
    """The command's first output lines, or None when it is absent or fails."""
    if shutil.which(command[0]) is None:
        return None
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    output = (done.stdout or done.stderr).strip()
    return output if done.returncode == 0 and output else None


def report() -> list[str]:
    lines = [f"python {platform.python_version()} on {platform.system()} {platform.machine()}"]
    gpus = run("nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader")
    rocm = None if gpus else run("rocm-smi", "--showproductname")
    lines.append("GPUs: " + (gpus.replace("\n", "; ") if gpus else rocm.splitlines()[0] if rocm else "none found (nvidia-smi, rocm-smi)"))
    slurm = run("sinfo", "--version")
    lines.append(f"scheduler: {slurm}" if slurm else "scheduler: no Slurm (sinfo) on PATH")
    for tool, flag in (("git", "--version"), ("uv", "--version"), ("claude", "--version"), ("codex", "--version"),
                       ("apptainer", "--version"), ("docker", "--version")):
        found = run(tool, flag)
        lines.append(f"{tool}: {found.splitlines()[0] if found else 'not found'}")
    cache = resolve_cache_root()
    existing = next(p for p in (cache, *cache.parents) if p.exists())
    lines.append(f"cache directory: {cache} ({shutil.disk_usage(existing).free / 1e9:.0f} GB free)")
    for spec_path in sorted((REPO_ROOT / "dataset" / "inputs").rglob("*.yaml")):
        spec = yaml.safe_load(spec_path.read_text())
        size = sum(f["size"] for f in spec["files"].values()) / 1e9
        built = [name for name, s in spec["sets"].items() if input_set_dir(cache, s["manifest_sha256"]).is_dir()]
        lines.append(f"inputs {spec['source_id']}: {len(built)} of {len(spec['sets'])} sets present, "
                     f"{size:.1f} GB to download if none is stored (python -m evaluation.warmup {spec['source_id']})")
    source = preferences_file()
    missing = unanswered()
    lines.append(f"answers: {shown(source)}, {len(missing)} unanswered")
    lines += [f"  - {item}" for item in missing]
    return lines


def init() -> str:
    if LOCAL.exists():
        return f"{shown(LOCAL)} exists; left unchanged"
    shutil.copyfile(TEMPLATE, LOCAL)
    return f"created {shown(LOCAL)} from {shown(TEMPLATE)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init", action="store_true", help="create the local preferences file if it is absent")
    args = parser.parse_args()
    if args.init:
        print(init())
    print("\n".join(report()))


if __name__ == "__main__":
    sys.exit(main())
