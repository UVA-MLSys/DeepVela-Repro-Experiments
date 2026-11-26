"""Gets a task's raw inputs once and builds verified input sets from them (dataset/inputs/<source_id>.yaml).

    python -m evaluation.warmup <source_id> [--tasks c0,c2]    # report only, changes nothing
        --build        build the missing sets from verified files already in the store
        --from DIR     first import missing items from DIR, laid out as the spec names them
        --download     first download the items still missing

In the cache directory (evaluation/cache.py), `store/` holds the verified downloads, `inputs/<set id>/` one input set
each, and `uv/` the package cache. A set's id is the start of the sha256 of its manifest, one
`<sha256> <x or -> <path>` line per file, which the spec pins. Nothing existing is overwritten: an item or set that
fails verification stops the command. Warmup builds no environment and preprocesses nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import yaml

from .cache import REPO_ROOT, resolve_cache_root

SPEC_DIR = REPO_ROOT / "dataset" / "inputs"
SET_ID_LENGTH = 16


class WarmupError(SystemExit):
    """An item or set that cannot be had or does not verify; the message says which and what to do."""


def load_spec(source_id: str) -> dict:
    path = SPEC_DIR / f"{source_id}.yaml"
    if not path.is_file():
        raise WarmupError(f"no input spec {path}")
    return yaml.safe_load(path.read_text())


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_text(entries: dict[str, tuple[str, bool]]) -> str:
    return "".join(f"{digest} {'x' if executable else '-'} {name}\n" for name, (digest, executable) in sorted(entries.items()))


def set_id(manifest_sha256: str) -> str:
    return manifest_sha256[:SET_ID_LENGTH]


def set_dir(cache: Path, manifest_sha256: str) -> Path:
    return cache / "inputs" / set_id(manifest_sha256)


def is_bytecode_cache(path: Path) -> bool:
    """A `__pycache__` directory holding only `.pyc` files, or one of those files."""
    if path.is_symlink():
        return False
    if path.name == "__pycache__" and path.is_dir():
        return all(p.suffix == ".pyc" and p.is_file() and not p.is_symlink() for p in path.iterdir())
    return path.parent.name == "__pycache__" and path.suffix == ".pyc" and path.is_file()


def verify_set(directory: Path, manifest_sha256: str) -> list[str]:
    """Why `directory` is not the set `manifest_sha256` pins; empty when it is. Any added, removed or changed file,
    changed executable bit, symlink or empty directory fails. Python's bytecode cache is not part of a set: `.pyc`
    files in `__pycache__/`, which importing a set's code writes, are ignored."""
    if not directory.is_dir() or directory.is_symlink():
        return [f"{directory} is not a directory"]
    entries, problems = {}, []
    for path in sorted(directory.rglob("*")):
        name = path.relative_to(directory).as_posix()
        if is_bytecode_cache(path):
            continue
        if path.is_symlink():
            problems.append(f"symlink {name}")
        elif path.is_file():
            entries[name] = (file_hash(path), bool(path.stat().st_mode & 0o111))
        elif not any(path.iterdir()):
            problems.append(f"empty directory {name}")
    found = hashlib.sha256(manifest_text(entries).encode()).hexdigest()
    if found != manifest_sha256:
        recorded = directory.parents[1] / "store" / "manifests" / set_id(manifest_sha256)
        if recorded.is_file():
            old, new = set(recorded.read_text().splitlines()), set(manifest_text(entries).splitlines())
            names = sorted({line.split(" ", 2)[2] for line in old ^ new})
            problems.append(f"{len(names)} files differ from the set's manifest, such as {names[:3]}")
        else:
            problems.append(f"its manifest sha256 is {found}, not {manifest_sha256}")
    return problems


@contextlib.contextmanager
def scratch(cache: Path):
    """A temporary directory on the cache's filesystem, so a finished item or set is moved into place in one step."""
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=cache / "tmp"))
    try:
        yield directory
    finally:
        if directory.exists():
            for path in [directory, *directory.rglob("*")]:
                if path.is_dir() and not path.is_symlink():
                    path.chmod(path.stat().st_mode | 0o700)
            shutil.rmtree(directory)


# Store: single files, named by their sha256

def stored_file(cache: Path, item: dict) -> Path:
    return cache / "store" / "files" / item["sha256"]


def check_file(path: Path, name: str, item: dict) -> None:
    if path.stat().st_size != item["size"]:
        raise WarmupError(f"{name}: {path} has {path.stat().st_size} bytes, the spec says {item['size']}")
    if (found := file_hash(path)) != item["sha256"]:
        raise WarmupError(f"{name}: {path} has sha256 {found}, the spec pins {item['sha256']}")
    if "publisher_md5" in item and (found := file_hash(path, "md5")) != item["publisher_md5"]:
        raise WarmupError(f"{name}: {path} has md5 {found}, the publisher lists {item['publisher_md5']}")


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "reproduction-warmup"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, 1 << 20)


def add_file(cache: Path, name: str, item: dict, source: Path | None) -> str:
    """Copies `source`, or downloads the spec's URL, into the store once it verifies."""
    target = stored_file(cache, item)
    with scratch(cache) as work:
        partial = work / "item"
        if source is not None:
            shutil.copyfile(source, partial)
        else:
            try:
                download(item["url"], partial)
            except OSError as exc:
                raise WarmupError(f"{name}: download of {item['url']} failed: {exc}") from exc
        check_file(partial, name, item)
        partial.chmod(0o444)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(partial, target)
    return f"imported from {source}" if source else f"downloaded from {item['url']}"


# Store: repositories, as bare clones of one commit

def git(*args: str, cwd: Path | None = None) -> bytes:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    if cwd is not None:  # never discover a repository above `cwd`: the cache may lie inside one
        environment["GIT_CEILING_DIRECTORIES"] = str(cwd.parent)
    done = subprocess.run(["git", *args], cwd=cwd, env=environment, capture_output=True)
    if done.returncode:
        raise WarmupError(f"git {' '.join(args)} failed: {done.stderr.decode(errors='replace')[-600:]}")
    return done.stdout


def stored_repository(cache: Path, name: str, item: dict) -> Path:
    return cache / "store" / "repos" / f"{name}-{item['commit']}.git"


def check_repository(path: Path, name: str, item: dict) -> None:
    found = git("--git-dir", str(path), "rev-parse", f"{item['commit']}^{{tree}}").decode().strip()
    if found != item["tree"]:
        raise WarmupError(f"{name}: commit {item['commit']} in {path} has tree {found}, the spec pins {item['tree']}")


def add_repository(cache: Path, name: str, item: dict, source: Path | None) -> str:
    origin = source.resolve().as_uri() if source is not None else item["url"]
    target = stored_repository(cache, name, item)
    with scratch(cache) as work:
        bare = work / "repo.git"
        git("init", "--quiet", "--bare", str(bare))
        git("--git-dir", str(bare), "fetch", "--quiet", "--depth", "1", origin, item["commit"])
        check_repository(bare, name, item)
        git("--git-dir", str(bare), "update-ref", "refs/heads/pinned", item["commit"])
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(bare, target)
    return f"imported from {source}" if source else f"fetched from {item['url']}"


def export_commit(bare: Path, commit: str, destination: Path) -> set[str]:
    """Writes the commit's files, without history, into `destination` and returns the executable ones. Each file is
    checked against its blob id, so an export that a git setting or attribute rewrote is refused."""
    listing = git("--git-dir", str(bare), "ls-tree", "-r", "-z", commit).decode().split("\0")
    tree = {name: (mode, blob) for entry in filter(None, listing)
            for (mode, _, blob), name in [(entry.split("\t", 1)[0].split(), entry.split("\t", 1)[1])]}
    if odd := sorted(n for n, (mode, _) in tree.items() if mode not in ("100644", "100755")):
        raise WarmupError(f"{commit} holds symlinks or submodules, which input sets do not support: {odd[:3]}")
    with tarfile.open(fileobj=io.BytesIO(git("--git-dir", str(bare), "archive", commit))) as archive:
        archive.extractall(destination, filter="data")
    for name, (_, blob) in tree.items():
        path = destination / name
        data = path.read_bytes() if path.is_file() else None
        if data is None or hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != blob:
            raise WarmupError(f"{commit}: exported {name} differs from the commit")
    return {name for name, (mode, _) in tree.items() if mode == "100755"}


def apply_mask(parent: Path, directory: str, patch: Path) -> None:
    """Applies a study mask patch to `parent/directory`. The patches compare the unmasked clone `<directory>` with a
    masked copy `<directory>-<condition>`: hunks under `.git/` are skipped (sets have no history), the copy's name is
    rewritten to `directory`, and files the patch reports only as binary may only be deletions."""
    body = re.sub(rb"(?m)^(diff --git a/.*? b/|\+\+\+ b/)" + re.escape(directory.encode()) + rb"-[A-Za-z0-9]+/",
                  rb"\g<1>" + directory.encode() + b"/", patch.read_bytes())
    text, deleted = [], []
    for part in filter(None, re.split(rb"(?m)^(?=diff --git )", body)):
        name = part.split(b"\n", 1)[0].rsplit(b" b/", 1)[-1].decode()
        if "/.git/" in name:
            continue
        if not re.search(rb"(?m)^Binary files ", part):
            text.append(part)
        elif re.search(rb"(?m)^Binary files .* and /dev/null differ$", part):
            deleted.append(name)
        else:
            raise WarmupError(f"{patch}: {name} is a binary change without its content")
    for name in deleted:  # the study copy also deleted untracked files, which an export never has
        target = (parent / name).resolve()
        if not target.is_relative_to((parent / directory).resolve()):
            raise WarmupError(f"{patch}: {name} lies outside {directory}")
        target.unlink(missing_ok=True)
    if text:
        (parent.parent / "mask.patch").write_bytes(b"".join(text))
        git("apply", "--whitespace=nowarn", str(parent.parent / "mask.patch"), cwd=parent)


def check_mask_hashes(directory: Path, listing: Path) -> None:
    want = {name.removeprefix("./"): digest for digest, name in
            (line.split("  ", 1) for line in listing.read_text().splitlines())}
    have = {p.relative_to(directory).as_posix(): file_hash(p) for p in directory.rglob("*") if p.is_file()}
    differ = sorted(n for n in want.keys() | have.keys() if want.get(n) != have.get(n))
    if differ:
        raise WarmupError(f"masked {directory.name} differs from {listing} in {len(differ)} files, such as {differ[:3]}")


# Input sets

def build_set(cache: Path, spec: dict, name: str) -> str:
    """Assembles set `name` from the store in a temporary directory and moves it into place if its manifest is the
    pinned one."""
    chosen = spec["sets"][name]
    with scratch(cache) as work:
        root = work / "set"
        root.mkdir()
        executables: set[str] = set()
        for relative, item in spec["files"].items():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            check_file(stored_file(cache, item), relative, item)
            shutil.copyfile(stored_file(cache, item), root / relative)
        for directory, item in spec.get("repositories", {}).items():
            bare = stored_repository(cache, directory, item)
            check_repository(bare, directory, item)
            executables |= {f"{directory}/{n}" for n in export_commit(bare, item["commit"], root / directory)}
            if mask := chosen.get("masks", {}).get(directory):
                masks = REPO_ROOT / mask
                apply_mask(root, directory, masks / "dle-mask.patch")
                check_mask_hashes(root / directory, masks / "dle-files.sha256")
        for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            if not any(path.iterdir()):
                path.rmdir()
        entries = {p.relative_to(root).as_posix(): (file_hash(p), p.relative_to(root).as_posix() in executables)
                   for p in root.rglob("*") if p.is_file()}
        manifest = manifest_text(entries)
        found = hashlib.sha256(manifest.encode()).hexdigest()
        pinned = chosen["manifest_sha256"]
        if found != pinned:
            raise WarmupError(f"set {name}: built manifest sha256 {found}, the spec pins {pinned}")
        target = set_dir(cache, pinned)
        record = cache / "store" / "manifests" / set_id(pinned)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(manifest)
        for path in root.rglob("*"):  # no write bits, against accidental edits; verification is what detects them
            path.chmod(0o555 if path.is_dir() or path.relative_to(root).as_posix() in executables else 0o444)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(root, target)
        target.chmod(0o555)  # after the move, which needs the directory writable
    return str(target)


def select_sets(spec: dict, tasks: list[str] | None) -> list[str]:
    if tasks is None:
        return list(spec["sets"])
    unknown = sorted(set(tasks) - {t for s in spec["sets"].values() for t in s["tasks"]})
    if unknown:
        raise WarmupError(f"the spec has no input set for task(s) {', '.join(unknown)}")
    return [name for name, s in spec["sets"].items() if set(s["tasks"]) & set(tasks)]


def items(cache: Path, spec: dict):
    """(name, item, stored path, check, add) for every file and repository."""
    for name, item in spec["files"].items():
        yield name, item, stored_file(cache, item), check_file, add_file
    for name, item in spec.get("repositories", {}).items():
        yield name, item, stored_repository(cache, name, item), check_repository, add_repository


def warmup(spec: dict, cache: Path, tasks: list[str] | None = None, build: bool = False,
           source: Path | None = None, allow_download: bool = False, report=print) -> bool:
    """Reports each needed set, and with `build`, `source` or `allow_download` gets what is missing and builds it.
    Returns whether every needed set is ready. A set that is ready needs nothing from the store or the network."""
    build = build or source is not None or allow_download
    wanted = select_sets(spec, tasks)
    missing_sets = []
    for name in wanted:
        chosen = spec["sets"][name]
        directory = set_dir(cache, chosen["manifest_sha256"])
        label = f"set {name} (tasks {', '.join(chosen['tasks'])}) {directory}"
        if not directory.exists():
            missing_sets.append(name)
            report(f"missing  {label}")
        elif problems := verify_set(directory, chosen["manifest_sha256"]):
            raise WarmupError(f"{label} does not verify: {'; '.join(problems)}. It is left as it is; "
                              "move it away to have it rebuilt")
        else:
            report(f"ready    {label}")
    if not missing_sets:
        return True
    to_download = 0
    for name, item, stored, check, add in items(cache, spec):
        if stored.exists():
            check(stored, name, item)
            report(f"ready    {name} in the store")
            continue
        local = source / name if source is not None and (source / name).exists() else None
        if build and (local is not None or allow_download):
            report(f"got      {name}: {add(cache, name, item, local)}")
            continue
        to_download += item.get("size", 0)
        size = f"{item['size']:,} bytes" if "size" in item else "one commit, size not listed"
        report(f"missing  {name} ({size})")
    if not build:
        report(f"to download: {to_download:,} bytes of files, plus any missing repository. Nothing was changed.")
        return False
    if any(not stored.exists() for _, _, stored, _, _ in items(cache, spec)):
        raise WarmupError("items are still missing: import them with --from DIR, or allow --download")
    (cache / "uv").mkdir(parents=True, exist_ok=True)
    for name in missing_sets:
        report(f"built    set {name}: {build_set(cache, spec, name)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source_id")
    parser.add_argument("--tasks", help="comma-separated task ids; default every set of the spec")
    parser.add_argument("--build", action="store_true", help="build missing sets from the store")
    parser.add_argument("--from", dest="source", type=Path, help="import missing items from this directory")
    parser.add_argument("--download", action="store_true", help="download the items that are still missing")
    args = parser.parse_args()
    if args.source is not None and not args.source.is_dir():
        raise WarmupError(f"--from {args.source} is not a directory")
    cache = resolve_cache_root()
    print(f"cache directory: {cache}")
    ready = warmup(load_spec(args.source_id), cache, args.tasks.split(",") if args.tasks else None,
                   args.build, args.source, args.download)
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
