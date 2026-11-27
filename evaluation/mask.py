"""Keeps a task's masked source reproducible from its pinned commit.

A task folder holds `source.tar.gz`, the masked source a run unpacks as `./repo`. Beside it, `source-files.txt` lists
the files taken from the pinned commit (`manifest.yaml` `source_revision`), and `source-mask.patch` turns them into the
archive's contents: edits and deletions only. Symlinks are kept as links and compared by target. Both files hold masked content, so, like the rest of the bundle, a
reproducing agent must not read them.

    python -m evaluation.mask write <task dir> --source <clone of the benchmark repository>
    python -m evaluation.mask check <task dir> [--source <clone>]

`write` derives the two files from the commit and the archive. `check` rebuilds the source from the commit, the
list and the patch, and fails unless every file, its bytes and its executable bit equal the archive's. `--source`
defaults to the clone the task's dataset record names.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml

FILES, PATCH, ARCHIVE = "source-files.txt", "source-mask.patch", "source.tar.gz"


def git(*args: str, cwd: Path | None = None) -> bytes:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_AUTHOR_NAME="mask",
                       GIT_AUTHOR_EMAIL="mask@example.invalid", GIT_COMMITTER_NAME="mask",
                       GIT_COMMITTER_EMAIL="mask@example.invalid")
    if cwd is not None:  # never use a repository above `cwd`
        environment["GIT_CEILING_DIRECTORIES"] = str(cwd.parent)
    done = subprocess.run(["git", *args], cwd=cwd, env=environment, capture_output=True)
    if done.returncode:
        raise SystemExit(f"git {' '.join(args[:3])} failed: {done.stderr.decode(errors='replace')[-500:]}")
    return done.stdout


def files_of(directory: Path) -> dict[str, tuple[bytes, bool | str]]:
    """Each file's bytes and executable bit; a symlink as its target, not followed."""
    entries = {}
    for p in sorted(directory.rglob("*")):
        if ".git" in p.relative_to(directory).parts:
            continue
        if p.is_symlink():
            entries[p.relative_to(directory).as_posix()] = (os.readlink(p).encode(), "symlink")
        elif p.is_file():
            entries[p.relative_to(directory).as_posix()] = (p.read_bytes(), bool(p.stat().st_mode & 0o111))
    return entries


def extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive) as handle:
        handle.extractall(destination, filter="data")


def export(source: Path, commit: str, names: list[str], destination: Path) -> None:
    """The commit's files `names`, written into `destination`."""
    data = git("-C", str(source), "archive", commit, "--", *names)
    with tarfile.open(fileobj=io.BytesIO(data)) as handle:
        handle.extractall(destination, filter="data")
    if missing := sorted(set(names) - set(files_of(destination))):
        raise SystemExit(f"commit {commit} lacks {missing[:3]}")


def commit_of(task: Path) -> str:
    return yaml.safe_load((task / "manifest.yaml").read_text())["source_revision"]


def default_source(task: Path) -> Path:
    from .cache import local_path
    from .task_prompt import load
    source_id = yaml.safe_load((task / "manifest.yaml").read_text())["source_id"]
    return local_path(load(source_id).repo["path"])


def base_names(source: Path, commit: str, masked: dict[str, tuple[bytes, bool]], others: list[Path]) -> list[str]:
    """The commit files the masked tree was taken from: its own files, and files another condition of the same
    benchmark keeps under the same top folders, which this condition deleted."""
    tracked = set(git("-C", str(source), "ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0")) - {""}
    names = set(masked)
    if extra := sorted(names - tracked):
        raise SystemExit(f"the archive holds files the commit does not: {extra[:3]}")
    for other in others:
        with tarfile.open(other) as handle:
            names |= {m.name.removeprefix("./") for m in handle.getmembers() if m.isfile() or m.issym()} & tracked
    return sorted(names)


def write(task: Path, source: Path, others: list[Path]) -> None:
    commit = commit_of(task)
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        extract(task / ARCHIVE, work / "masked")
        masked = files_of(work / "masked")
        names = base_names(source, commit, masked, others)
        repo = work / "repo"
        export(source, commit, names, repo)
        git("init", "-q", cwd=repo)
        git("add", "-A", cwd=repo)
        git("commit", "-qm", "base", cwd=repo)
        for path in list(repo.iterdir()):
            if path.name != ".git":
                shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
        shutil.copytree(work / "masked", repo, symlinks=True, dirs_exist_ok=True)
        git("add", "-A", cwd=repo)
        patch = git("diff", "--cached", "--no-renames", "--full-index", cwd=repo)
    (task / FILES).write_text("".join(f"{name}\n" for name in names))
    (task / PATCH).write_bytes(patch)


def rebuild(task: Path, source: Path, destination: Path) -> None:
    export(source, commit_of(task), (task / FILES).read_text().splitlines(), destination)
    if (task / PATCH).stat().st_size:
        git("apply", "--whitespace=nowarn", str((task / PATCH).resolve()), cwd=destination)


def check(task: Path, source: Path) -> list[str]:
    """Why the rebuilt source differs from the archive; empty when it equals it."""
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        (work / "built").mkdir()
        rebuild(task, source, work / "built")
        extract(task / ARCHIVE, work / "archive")
        built, archived = files_of(work / "built"), files_of(work / "archive")
    differ = sorted(n for n in built.keys() | archived.keys() if built.get(n) != archived.get(n))
    return [f"{len(differ)} files differ from {ARCHIVE}, such as {differ[:3]}"] if differ else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["write", "check"])
    parser.add_argument("task", type=Path, help="a task folder of a bundle")
    parser.add_argument("--source", type=Path, help="a clone of the benchmark repository holding the pinned commit")
    parser.add_argument("--other", type=Path, action="append", default=[],
                        help="write: another condition's source.tar.gz, whose files this condition may have deleted")
    args = parser.parse_args()
    source = args.source or default_source(args.task)
    if args.action == "write":
        write(args.task, source, args.other)
    if problems := check(args.task, source):
        raise SystemExit(f"{args.task}: {'; '.join(problems)}")
    print(f"{args.task}: rebuilt from {commit_of(args.task)[:12]} and {PATCH}; equals {ARCHIVE}")


if __name__ == "__main__":
    main()
