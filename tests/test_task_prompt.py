"""Task prompts are rendered from the dataset record, never written by hand (evaluation/task_prompt.py)."""

import json

import pytest

from evaluation import task_prompt
from evaluation.cache import INPUT_ROOT, PACKAGE_CACHE, PLACEHOLDER
from evaluation.task_prompt import DATASET_DIR, REPO_ROOT, load, render


@pytest.fixture(autouse=True)
def no_source_clones(monkeypatch):
    # The documented-command check reads the benchmark clones under .cache/sources, absent in a fresh checkout.
    monkeypatch.setattr(task_prompt, "undocumented_commands", lambda source, tasks: [])


def test_every_answered_source_renders_every_task_with_its_record():
    rendered = 0
    for path in sorted(DATASET_DIR.rglob("*.json")):
        if path.name.endswith("-prior-artifacts.json"):  # run maps of a controlled study, not task records
            continue
        source = load(path.relative_to(DATASET_DIR).with_suffix("").as_posix())
        if source.policy.forbidden is None:  # unanswered source: covered by the refusal test
            continue
        text, tasks = render(source)
        assert "$" not in text.replace("$(", "").replace("$CACHE_ROOT", "").replace("$INPUT_ROOT", "")  # every slot filled
        if task_prompt.input_set_paths(source.source_id):  # inputs are paths inside the task's input set
            assert f"INPUT_ROOT={INPUT_ROOT}\n" in text and "CACHE_ROOT" not in text
        elif len(source.policy.inputs or {}) > 1:  # a shared path prefix is stated once, as CACHE_ROOT
            assert f"CACHE_ROOT={PLACEHOLDER}\n" in text
        if source.model_extra.get("language") == "Python":
            assert f"UV_CACHE_DIR={PACKAGE_CACHE} " in text
        assert str(REPO_ROOT / ".cache") not in text  # a rendered prompt names no machine's cache directory
        assert json.loads(tasks) == [t.model_dump(exclude_none=True, exclude={"selector_id", "input_path_overrides", "forbidden_override"})
                                     for t in source.tasks]
        assert not any(t.task_prompt in text for t in source.tasks)  # per-target fields only in tasks.json
        rendered += 1
    assert rendered == 18  # every shipped record whose policy is answered


def test_every_condition_of_a_source_with_input_sets_names_its_inputs_alike():
    source = load("mlperf-c0-c6/bert-squad")
    blocks = {render(source, [f"c{i}"])[0].split("Inputs, read-only")[1].split("Work for each")[0] for i in range(7)}
    assert len(blocks) == 1 and "$INPUT_ROOT/DeepLearningExamples\n" in blocks.pop()


def test_input_path_outside_the_input_spec_is_refused():
    source = load("mlperf-c0-c6/bert-squad")
    source.policy.inputs["SQUAD_DIR"].path = "data-c4"
    with pytest.raises(SystemExit, match="not in its input spec"):
        render(source, ["c0"])


def test_unanswered_field_stops_rendering_and_names_it():
    source = load("mlperf-language-bert")
    source.policy.forbidden = None
    with pytest.raises(SystemExit, match="policy.forbidden"):
        render(source)
