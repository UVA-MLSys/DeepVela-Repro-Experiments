"""evaluation.warmup stores verified inputs once and builds input sets that match the spec's manifests."""

import hashlib
import stat
import subprocess

import pytest

from evaluation import warmup as w

MODEL, DATA = b"model bytes\n", b'{"data": []}\n'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(directory, *args) -> str:
    return subprocess.run(["git", "-C", str(directory), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def expected(entries: dict[str, tuple[bytes, bool]]) -> str:
    return sha(w.manifest_text({n: (sha(d), x) for n, (d, x) in entries.items()}).encode())


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A source folder laid out as the spec names it, a spec pinning it, and a cache inside a git repository."""
    source = tmp_path / "source"
    code = source / "Code"
    (code / "sub").mkdir(parents=True)
    (source / "model").mkdir()
    (source / "model/model.bin").write_bytes(MODEL)
    (source / "data.json").write_bytes(DATA)
    (code / "run.sh").write_bytes(b"echo secret\n")
    (code / "run.sh").chmod(0o755)
    (code / "sub/doc.md").write_bytes(b"answer: 42\nkeep\n")
    (code / "sub/blob.bin").write_bytes(b"\0\1\2")
    git(code, "init", "-q")
    git(code, "add", "-A")
    git(code, "commit", "-qm", "c")

    # A mask as the study wrote it: a `.git` hunk, a renamed destination, a binary deletion, and an untracked one.
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "dle-mask.patch").write_text(
        "diff --git a/Code/.git/HEAD b/Code/.git/HEAD\ndeleted file mode 100644\n--- a/Code/.git/HEAD\n+++ /dev/null\n"
        "@@ -1 +0,0 @@\n-ref: refs/heads/main\n"
        "diff --git a/Code/sub/doc.md b/Code-c2/sub/doc.md\n--- a/Code/sub/doc.md\n+++ b/Code-c2/sub/doc.md\n"
        "@@ -1,2 +1,2 @@\n-answer: 42\n+answer: <masked>\n keep\n"
        "diff --git a/Code/sub/blob.bin b/Code/sub/blob.bin\nBinary files a/Code/sub/blob.bin and /dev/null differ\n"
        "diff --git a/Code/x.pyc b/Code/x.pyc\nBinary files a/Code/x.pyc and /dev/null differ\n")
    masked = {"run.sh": b"echo secret\n", "sub/doc.md": b"answer: <masked>\nkeep\n"}
    (masks / "dle-files.sha256").write_text("".join(f"{sha(d)}  ./{n}\n" for n, d in masked.items()))

    files = {"model/model.bin": (MODEL, False), "data.json": (DATA, False)}
    base = files | {"Code/run.sh": (b"echo secret\n", True), "Code/sub/doc.md": (b"answer: 42\nkeep\n", False),
                    "Code/sub/blob.bin": (b"\0\1\2", False)}
    c2 = files | {"Code/run.sh": (masked["run.sh"], True), "Code/sub/doc.md": (masked["sub/doc.md"], False)}
    spec = {
        "files": {n: {"url": (source / n).as_uri(), "size": len(d), "sha256": sha(d),
                      "publisher_md5": hashlib.md5(d).hexdigest()} for n, (d, _) in files.items()},
        "repositories": {"Code": {"url": code.as_uri(), "commit": git(code, "rev-parse", "HEAD"),
                                  "tree": git(code, "rev-parse", "HEAD^{tree}")}},
        "sets": {"base": {"tasks": ["c0", "c1"], "manifest_sha256": expected(base)},
                 "c2": {"tasks": ["c2"], "masks": {"Code": str(masks)}, "manifest_sha256": expected(c2)}},
    }
    outer = tmp_path / "checkout"  # the cache usually lies inside the repository, which `git apply` must not see
    outer.mkdir()
    git(outer, "init", "-q")
    yield spec, source, outer / ".cache"
    for path in tmp_path.rglob("*"):  # let pytest remove the read-only sets
        if path.is_dir() and not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o700)


def run(world, **options):
    spec, _, cache = world
    log = []
    return w.warmup(spec, cache, report=log.append, **options), "\n".join(log)


def sets_of(world):
    spec, _, cache = world
    return {name: w.set_dir(cache, s["manifest_sha256"]) for name, s in spec["sets"].items()}


def test_check_changes_nothing(world):
    ready, log = run(world)
    assert not ready and "Nothing was changed" in log and not world[2].exists()


def test_import_builds_read_only_sets_without_history(world):
    ready, _ = run(world, source=world[1])
    sets = sets_of(world)
    assert ready and (sets["c2"] / "Code/sub/doc.md").read_bytes() == b"answer: <masked>\nkeep\n"
    assert not (sets["c2"] / "Code/sub/blob.bin").exists() and not (sets["base"] / "Code/.git").exists()
    for name, directory in sets.items():
        assert directory.name == world[0]["sets"][name]["manifest_sha256"][:16]
        assert not any(p.stat().st_mode & stat.S_IWUSR for p in [directory, *directory.rglob("*")])
        assert (directory / "Code/run.sh").stat().st_mode & stat.S_IXUSR
        assert not (directory / "data.json").stat().st_mode & stat.S_IXUSR
    assert not any((world[2] / "tmp").iterdir())


def test_repeat_needs_neither_the_source_nor_the_network(world, monkeypatch):
    run(world, source=world[1])
    monkeypatch.setattr(w, "download", lambda *a: pytest.fail("downloaded"))
    monkeypatch.setattr(w, "git", lambda *a, **k: pytest.fail("ran git"))
    ready, log = run(world, build=True, allow_download=True)
    assert ready and log.count("ready    set") == 2 and "built" not in log


def test_download_only_when_allowed_and_only_the_needed_set(world):
    with pytest.raises(w.WarmupError, match="still missing"):
        run(world, build=True)
    ready, log = run(world, allow_download=True, tasks=["c2"])
    assert ready and log.count("downloaded from") == 2 and log.count("fetched from") == 1
    assert sets_of(world)["c2"].is_dir() and not sets_of(world)["base"].exists()


def test_wrong_download_is_not_stored(world):
    spec, source, cache = world
    (source / "model/model.bin").write_bytes(b"model byte")  # an interrupted transfer
    with pytest.raises(w.WarmupError, match="has 10 bytes"):
        run(world, allow_download=True)
    assert not w.stored_file(cache, spec["files"]["model/model.bin"]).exists() and not (cache / "inputs").exists()


def test_corrupt_source_and_corrupt_store_are_refused_and_left_alone(world):
    spec, source, cache = world
    (source / "model/model.bin").write_bytes(MODEL.upper())
    with pytest.raises(w.WarmupError, match="has sha256"):
        run(world, source=source)
    stored = w.stored_file(cache, spec["files"]["model/model.bin"])
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes(MODEL.upper())
    (source / "model/model.bin").write_bytes(MODEL)
    with pytest.raises(w.WarmupError, match="has sha256"):
        run(world, source=source)
    assert stored.read_bytes() == MODEL.upper()


def test_python_bytecode_cache_is_not_a_change(world):
    run(world, source=world[1])
    directory = sets_of(world)["base"]
    (directory / "Code/sub").chmod(0o755)
    (directory / "Code/sub/__pycache__").mkdir()
    (directory / "Code/sub/__pycache__/doc.cpython-310.pyc").write_bytes(b"bytecode")
    ready, log = run(world, build=True)
    assert ready and "built" not in log


@pytest.mark.parametrize("added", ["Code/sub/__pycache__/notes.txt", "Code/sub/stray.pyc"])
def test_other_files_in_or_like_the_bytecode_cache_are_changes(world, added):
    run(world, source=world[1])
    directory = sets_of(world)["base"]
    (directory / "Code/sub").chmod(0o755)
    (directory / added).parent.mkdir(exist_ok=True)
    (directory / added).write_text("x")
    assert w.verify_set(directory, world[0]["sets"]["base"]["manifest_sha256"])


@pytest.mark.parametrize("tamper", ["change", "add", "remove", "mode"])
def test_changed_set_is_reported_and_not_rebuilt(world, tamper):
    run(world, source=world[1])
    directory = sets_of(world)["base"]
    for path in [directory, directory / "Code", directory / "Code/sub"]:
        path.chmod(0o755)
    target = directory / "Code/sub/doc.md"
    if tamper == "change":
        target.chmod(0o644)
        target.write_text("other\n")
    elif tamper == "add":
        (directory / "Code/sub/new.py").write_text("x")
    elif tamper == "remove":
        target.unlink()
    else:
        target.chmod(0o755)
    with pytest.raises(w.WarmupError, match="does not verify.*files differ"):
        run(world, source=world[1])
    assert w.verify_set(directory, world[0]["sets"]["base"]["manifest_sha256"])


def test_build_that_misses_the_pinned_manifest_is_not_moved_into_place(world):
    spec, source, cache = world
    spec["sets"]["base"]["manifest_sha256"] = "0" * 64
    with pytest.raises(w.WarmupError, match="the spec pins"):
        run(world, source=source)
    assert not (cache / "inputs").exists() and not any((cache / "tmp").iterdir())


def test_mask_that_does_not_give_its_hash_list_is_refused(world):
    spec, source, cache = world
    listing = w.Path(spec["sets"]["c2"]["masks"]["Code"]) / "dle-files.sha256"
    listing.write_text(listing.read_text() + f"{'0' * 64}  ./sub/other.md\n")
    with pytest.raises(w.WarmupError, match="differs from"):
        run(world, source=source, tasks=["c2"])
    assert not sets_of(world)["c2"].exists()


def test_repository_with_another_tree_is_refused(world):
    spec, source, cache = world
    spec["repositories"]["Code"]["tree"] = "0" * 40
    with pytest.raises(w.WarmupError, match="has tree"):
        run(world, source=source)


def test_unknown_task_is_refused(world):
    with pytest.raises(w.WarmupError, match="c9"):
        run(world, tasks=["c9"])
