"""evaluation.mask records a masked source as a file list and a patch on the pinned commit, and checks it rebuilds."""

import io
import subprocess
import tarfile

import pytest

from evaluation import mask


def git(directory, *args):
    return subprocess.run(["git", "-C", str(directory), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def archive(path, files):
    with tarfile.open(path, "w:gz") as handle:
        for name, (data, kind) in files.items():
            entry = tarfile.TarInfo(name)
            if kind == "link":
                entry.type, entry.linkname = tarfile.SYMTYPE, data.decode()
            else:
                entry.size, entry.mode = len(data), 0o755 if kind == "exec" else 0o644
            handle.addfile(entry, None if kind == "link" else io.BytesIO(data))


@pytest.fixture
def task(tmp_path):
    source = tmp_path / "source"
    (source / "bench").mkdir(parents=True)
    (source / "bench/README.md").write_text("score: 0.86\nusage\n")
    (source / "bench/run.sh").write_text("run\n")
    (source / "bench/run.sh").chmod(0o755)
    (source / "bench/secret.py").write_text("x = 1\n")
    (source / "bench/a manual.pdf").write_bytes(b"%PDF\0binary")
    (source / "other.txt").write_text("not used\n")
    (source / "conf").symlink_to("bench/run.sh")
    git(source, "init", "-q")
    git(source, "add", "-A")
    git(source, "commit", "-qm", "c")
    folder = tmp_path / "task"
    folder.mkdir()
    (folder / "manifest.yaml").write_text(f"source_revision: {git(source, 'rev-parse', 'HEAD')}\n")
    full = {"bench/README.md": (b"score: 0.86\nusage\n", "file"), "bench/run.sh": (b"run\n", "exec"),
            "bench/secret.py": (b"x = 1\n", "file"), "bench/a manual.pdf": (b"%PDF\0binary", "file"), "conf": (b"bench/run.sh", "link")}
    archive(tmp_path / "c0.tar.gz", full)
    masked = dict(full, **{"bench/README.md": (b"score: <masked>\nusage\n", "file")})
    del masked["bench/secret.py"]
    archive(folder / "source.tar.gz", masked)
    return folder, source, tmp_path / "c0.tar.gz"


def test_write_records_edits_and_deletions_and_check_rebuilds_the_archive(task):
    folder, source, c0 = task
    mask.write(folder, source, [c0])
    assert (folder / "source-files.txt").read_text().splitlines() == ["bench/README.md", "bench/a manual.pdf", "bench/run.sh", "bench/secret.py", "conf"]
    patch = (folder / "source-mask.patch").read_text()
    assert "+score: <masked>" in patch and "deleted file mode 100644" in patch and "other.txt" not in patch
    assert mask.check(folder, source) == []


def test_check_fails_when_the_archive_and_the_patch_disagree(task):
    folder, source, c0 = task
    mask.write(folder, source, [c0])
    patch = folder / "source-mask.patch"
    patch.write_text(patch.read_text().replace("+score: <masked>", "+score: <hidden>"))
    assert mask.check(folder, source) == ["1 files differ from source.tar.gz, such as ['bench/README.md']"]


def test_an_executable_bit_or_symlink_change_is_a_difference(task, tmp_path):
    folder, source, c0 = task
    mask.write(folder, source, [c0])
    archive(folder / "source.tar.gz", {"bench/README.md": (b"score: <masked>\nusage\n", "file"),
                                       "bench/a manual.pdf": (b"%PDF\0binary", "file"), "bench/run.sh": (b"run\n", "file"),
                                       "conf": (b"bench/README.md", "link")})
    assert "2 files differ" in mask.check(folder, source)[0]
