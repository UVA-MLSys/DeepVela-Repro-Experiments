"""evaluation.prepare builds each arm's workspace from a sealed profile bundle."""

import io
import json
import tarfile
import time

import pytest

from evaluation import prepare as prepare_module
from evaluation.bundle import EvaluationBundleError, seal
from evaluation.prepare import prepare
from evaluation.task_prompt import load

COMMIT = load("mlperf-language-bert").repo["commit"]


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    profile = tmp_path / "bundles/p"
    task = profile / "tasks/t"
    task.mkdir(parents=True)
    (profile / "bundle.yaml").write_text("profile_id: p\nprofile_version: 1\n")
    (task / "manifest.yaml").write_text(
        f"task_id: t\nprofile_id: p\nsource_id: mlperf-language-bert\nsource_revision: {COMMIT}\n")
    (task / "prompt.md").write_text(f"Reproduce the synthetic task at commit {COMMIT}.\n")
    (task / "tasks.json").write_text("[]\n")
    (task / "masked.txt").write_text("No reference value stated.\n")
    with tarfile.open(task / "source.tar.gz", "w:gz") as archive:
        payload = b"synthetic pinned source\n"
        entry = tarfile.TarInfo("README.md")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    monkeypatch.setenv("EVALUATION_BUNDLE_DIR", str(tmp_path / "bundles"))
    monkeypatch.setenv("REPRODUCTION_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(prepare_module, "USER_CODEX_HOME", tmp_path / "no-codex")
    seal("p", "t")
    return task


def test_skill_arm_gets_each_harness_package(bundle):
    root = prepare("p", "t", "skill", compute="local")
    ws = root / "ws"
    claude, codex = ws / ".claude/skills/reproduce", ws / ".agents/skills/reproduce"
    assert (claude / "SKILL.md").read_text() == (prepare_module.SKILLS / "claude/reproduce/SKILL.md").read_text()
    assert (codex / "SKILL.md").read_text() == (prepare_module.SKILLS / "codex/reproduce/SKILL.md").read_text()
    assert json.loads((ws / ".claude/settings.json").read_text())["claudeMdExcludes"]
    assert (ws / "prompt.md").read_text().startswith("/reproduce\nReproduce the synthetic task")
    assert (ws / "repo/README.md").is_file()
    launch = json.loads((root / "launch.json").read_text())
    assert launch["profile_id"] == "p" and launch["skill"] == "reproduce" and launch["ended"] is None


def test_builder_arm_installs_both_skills_and_names_the_handoff(bundle):
    ws = prepare("p", "t", "builder-skill", compute="local", autonomy=True) / "ws"
    for harness in (".claude/skills", ".agents/skills"):
        assert {p.name for p in (ws / harness).iterdir()} == {"builder-reproduce", "reproduce"}
    text = (ws / "prompt.md").read_text()
    assert text.startswith("/builder-reproduce\n") and "No human will reply" in text
    assert text.rstrip().endswith(prepare_module.BUILDER_INSTRUCTION)


def test_plain_arm_has_no_skill(bundle):
    ws = prepare("p", "t", "plain", compute="local") / "ws"
    assert not (ws / ".claude/skills").exists() and not (ws / ".agents/skills").exists()
    assert (ws / "prompt.md").read_text().startswith("Reproduce the synthetic task")


def test_unanswered_compute_wording_refuses_launch(bundle):
    with pytest.raises(SystemExit, match=r"\{\{account\}\}"):
        prepare("p", "t", "skill", compute="slurm")


def test_bundle_edited_after_sealing_refuses_launch(bundle):
    (bundle / "prompt.md").write_text("A different task.\n")
    with pytest.raises(EvaluationBundleError, match="does not match"):
        prepare("p", "t", "skill", compute="local")


def test_overlapping_roots_refuse_launch(bundle, monkeypatch):
    monkeypatch.setenv("REPRODUCTION_RUNS_DIR", str(bundle.parent / "runs"))
    with pytest.raises(SystemExit, match="overlap"):
        prepare("p", "t", "skill", compute="local")


def test_manifest_must_match_the_record(bundle):
    (bundle / "manifest.yaml").write_text(
        "task_id: t\nprofile_id: p\nsource_id: mlperf-language-bert\nsource_revision: 0000\n")
    with pytest.raises(EvaluationBundleError, match="record pins"):
        seal("p", "t")


def test_unsafe_ids_are_refused(bundle):
    with pytest.raises(EvaluationBundleError, match="task_id"):
        prepare("p", "../t", "skill", compute="local")
    with pytest.raises(EvaluationBundleError, match="profile_id"):
        prepare("../p", "t", "skill", compute="local")


def test_codex_command_is_interactive_unless_autonomous_and_quoted(bundle, monkeypatch, tmp_path):
    monkeypatch.setenv("REPRODUCTION_RUNS_DIR", str(tmp_path / "runs with space"))
    interactive = json.loads((prepare("p", "t", "skill", compute="local", model="m1") / "launch.json").read_text())
    autonomous = json.loads((prepare("p", "t", "plain", compute="local", autonomy=True) / "launch.json").read_text())
    assert " exec " not in interactive["codex_command"] and interactive["model"] == "m1"
    assert " exec " in autonomous["codex_command"] and "'" in autonomous["codex_command"]


def test_codex_home_holds_only_the_study_model_not_the_operators_config(bundle, monkeypatch, tmp_path):
    home = tmp_path / "operator-codex"
    home.mkdir()
    (home / "config.toml").write_text('[mcp_servers.secret]\ncommand = "x"\n')
    monkeypatch.setattr(prepare_module, "USER_CODEX_HOME", home)
    root = prepare("p", "t", "plain", compute="local", model="m1", effort="medium")
    config = (root / "codex-home/config.toml").read_text()
    assert 'model = "m1"' in config and 'model_reasoning_effort = "medium"' in config
    assert "secret" not in config and "enabled = false" in config


def test_prior_artifacts_are_staged_read_only_per_trial_and_arm(bundle, tmp_path):
    prior = tmp_path / "earlier/ws"
    for name in ("reproduce.sh", "report.json", "3d-unet/out/accuracy.txt", "3d-unet/out/mlperf_log_accuracy.json",
                 "3d-unet/work/preprocessed/case.pkl", "3d-unet/my-env/pyvenv.cfg", "3d-unet/my-env/lib/mod.py"):
        (prior / name).parent.mkdir(parents=True, exist_ok=True)
        (prior / name).write_text("x\n")
    manifest = (bundle / "manifest.yaml").read_text() + "targets:\n  3d-unet: {}\nprior_artifacts:\n  trials:\n"
    (bundle / "manifest.yaml").write_text(
        manifest + f"    2:\n      plain: {prior}\n    3:\n      plain:\n        v4: {prior}\n        v5: {prior}\n")
    seal("p", "t")
    with pytest.raises(SystemExit, match="--trial is required"):
        prepare("p", "t", "plain", compute="local")
    time.sleep(1.1)  # run folders are named by the second

    root = prepare("p", "t", "plain", compute="local", trial=2)
    snapshot = root / "ws/prior-artifacts"
    assert (snapshot / "reproduce.sh").is_file()
    assert (snapshot / "3d-unet/out/accuracy.txt").is_file()
    assert not (snapshot / "3d-unet/out/mlperf_log_accuracy.json").exists()
    assert not (snapshot / "3d-unet/work/preprocessed/case.pkl").exists()
    assert not (snapshot / "3d-unet/my-env").exists()
    assert snapshot.stat().st_mode & 0o222 == 0
    launch = json.loads((root / "launch.json").read_text())
    assert launch["prior_artifacts"]["source_workspace"] == str(prior.resolve())
    assert launch["trial"] == 2
    assert "same arm's prior campaign" in (root / "ws/prompt.md").read_text()

    time.sleep(1.1)  # run folders are named by the second
    root = prepare("p", "t", "plain", compute="local", trial=3)
    for label in ("v4", "v5"):
        assert (root / "ws/prior-artifacts" / label / "reproduce.sh").is_file()
        assert (root / "ws/prior-artifacts" / label / "provenance.json").is_file()
    assert "./prior-artifacts/v4, ./prior-artifacts/v5" in (root / "ws/prompt.md").read_text()
    assert set(json.loads((root / "launch.json").read_text())["prior_artifacts"]["sources"]) == {"v4", "v5"}
