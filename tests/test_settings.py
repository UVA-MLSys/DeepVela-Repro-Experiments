"""Deployment answers come from an environment variable, else the ignored local preferences file."""

import pytest

from evaluation import settings, setup
from evaluation.bundle import EvaluationBundleError, resolve_bundle_root

TEXT = settings.TEMPLATE.read_text()


@pytest.fixture
def local(tmp_path, monkeypatch):
    path = tmp_path / "local-preferences.local.md"
    monkeypatch.setattr(settings, "LOCAL", path)
    monkeypatch.setattr(setup, "LOCAL", path)
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(setup, "REPO_ROOT", tmp_path)
    for variable in ("EVALUATION_BUNDLE_DIR", "REPRODUCTION_RUNS_DIR", "EVALUATION_JUDGE", "REPRODUCTION_CACHE_DIR"):
        monkeypatch.delenv(variable, raising=False)
    return path


def answered(text: str, section: str, value: str) -> str:
    head, rest = text.split(f"## {section}", 1)
    return head + f"## {section}" + rest.replace("Answer: unanswered", f"Answer: {value}", 1)


def test_the_template_is_unanswered_and_lists_every_item(local):
    assert settings.preferences_file() == settings.TEMPLATE
    assert all(settings.answer(n) is None for n in (1, 2, 5)) and settings.answer(4, "Judge") is None
    items = settings.unanswered()
    assert "1. Bundle root" in items and "4. Study: Judge for claims a rubric checks by reading the transcript" in items
    assert any("Scheduler" in i and i.endswith(": Account") for i in items)


def test_local_answers_are_read_and_the_environment_overrides_them(local, monkeypatch, tmp_path):
    text = answered(TEXT, "1. Bundle root", "`../bundles`")
    text = answered(text, "4. Study", "x")  # the first item of section 4, not the judge
    local.write_text(text.replace("favors their output, so recommend the other vendor. Answer: unanswered",
                                  "favors their output, so recommend the other vendor. Answer: codex"))
    assert settings.preferences_file() == local
    assert resolve_bundle_root() == (tmp_path / "../bundles").resolve()  # relative to the repository root
    assert settings.answer(4, "Judge") == "codex" and settings.answer(4, "Arms") == "x"
    assert "1. Bundle root" not in settings.unanswered()
    monkeypatch.setenv("EVALUATION_BUNDLE_DIR", str(tmp_path / "elsewhere"))
    assert resolve_bundle_root() == tmp_path / "elsewhere"


def test_an_unanswered_root_is_refused(local):
    with pytest.raises(EvaluationBundleError, match="run python -m evaluation.setup"):
        resolve_bundle_root()


def test_init_copies_the_template_once_and_never_overwrites(local):
    assert "created" in setup.init() and local.read_text() == TEXT
    local.write_text("my answers\n")
    assert "left unchanged" in setup.init() and local.read_text() == "my answers\n"


def test_report_runs_without_changing_anything(local):
    lines = setup.report()
    assert lines[0].startswith("python ") and any(l.startswith("answers: ") for l in lines) and not local.exists()
