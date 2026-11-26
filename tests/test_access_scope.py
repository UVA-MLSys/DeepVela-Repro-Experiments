"""The scorer flags what the task prompt's access rule forbids inside the framework repository, the runs root and
the cache directory. The workspace, the package cache, the task's inputs, and everything outside the guarded roots
pass."""

import json

from pathlib import Path

from evaluation.prepare import access_scope
from evaluation.scorer import outside_scope

SCOPE = {"guarded": ["/p/repo", "/p/runs"],
         "allowed": ["/p/runs/t/skill-1/ws", "/p/repo/.cache"]}


def flagged(command: str) -> list[str]:
    return outside_scope(json.dumps({"command": command}), SCOPE)


def test_named_paths_and_outside_roots_pass():
    assert flagged("CACHE_ROOT=/p/repo/.cache; ls /p/repo/.cache/imagenet/x.JPEG /usr/lib /tmp/s") == []
    assert flagged("cd /p/runs/t/skill-1/ws/repo && UV_CACHE_DIR=/p/repo/.cache/uv uv sync") == []


def test_reports_and_other_runs_are_flagged_cache_is_not():
    assert flagged("cat /p/repo/docs/reports/README.md") == ["/p/repo/docs/reports/README.md"]
    assert flagged("cat /p/runs/t/plain-1/ws/report.json") == ["/p/runs/t/plain-1/ws/report.json"]
    assert flagged("ls /p/repo/.cache/sources /p/repo/.cache/huggingface") == []
    assert flagged("cat /p/repo/docs/reports/notes.md") == ["/p/repo/docs/reports/notes.md"]


def test_search_pattern_is_not_an_access_but_a_read_is():
    from evaluation.scorer import without_search_patterns
    import re
    forbidden = r"inference_results|github\.com/mlcommons/inference|runs\.csv"
    hit = lambda text: bool(re.search(forbidden, without_search_patterns(text)))
    assert not hit("if rg -n 'inference_results|github.com/mlcommons/inference' reproduce.sh; then exit 1; fi")
    assert not hit("grep -rn -e 'runs.csv' src/")
    assert hit("cat runs.csv")
    assert hit("rg -n 'x' inference_results/a.txt")
    assert hit("rg -n 'runs.csv' f; curl https://github.com/mlcommons/inference")
    assert hit("rg -n 'runs.csv' f; cat runs.csv")


def cache_scope(cache: str) -> dict:
    return access_scope("s", Path("/p/runs/t/skill-1/ws"), Path("/p/runs"), Path(cache),
                        [f"{cache}/bert-squad/data-c4", f"{cache}/bert-squad/model"])


def test_only_declared_inputs_and_the_package_cache_pass():
    for cache in ("/p/repo/.cache", "/elsewhere/cache"):  # inside the repository, and outside every other root
        scope = cache_scope(cache)
        hits = lambda command: outside_scope(json.dumps({"command": command}), scope)
        assert hits(f"cat {cache}/bert-squad/data-c4/dev-v1.1.json; ls {cache}/uv {cache}/bert-squad/model") == []
        for other in ("bert-squad/data/evaluate-v1.1.py", "bert-squad/DeepLearningExamples", "bert-backup-2/a.xlsx",
                      "sources/inference", "bert-squad/data-c4-copy"):
            assert hits(f"cat {cache}/{other}") == [f"{cache}/{other}"]
        assert hits(f"find {cache} -name 'evaluate*'") == [cache]


def test_cache_root_variable_is_expanded_and_its_assignment_passes():
    scope = cache_scope("/c")
    hits = lambda tool_input: outside_scope(json.dumps(tool_input), scope)
    for command in ("CACHE_ROOT=/c", "export CACHE_ROOT=\"/c\"; echo ok", "CACHE_ROOT=${CACHE_ROOT:-/c}",
                    "CACHE_ROOT=\"${CACHE_ROOT:-/c/}\"\nUV_CACHE_DIR=$CACHE_ROOT/uv"):
        assert hits({"command": command}) == [], command
    assert hits({"command": "cat $CACHE_ROOT/bert-squad/data/x ${CACHE_ROOT}/Papers/y"}) == [
        "/c/bert-squad/data/x", "/c/Papers/y"]
    assert hits({"command": "CACHE_ROOT=/c/Papers"}) == ["/c/Papers"]
    assert hits({"command": "ls /c"}) == ["/c"]


def test_a_run_on_an_input_set_may_touch_that_set_only():
    scope = access_scope("s", Path("/p/runs/t/skill-1/ws"), Path("/p/runs"), Path("/c"),
                         ["/c/inputs/aaaa/data"], Path("/c/inputs/aaaa"))
    hits = lambda command: outside_scope(json.dumps({"command": command}), scope)
    assert hits("INPUT_ROOT=/c/inputs/aaaa; find $INPUT_ROOT/ -type f; ls /c/inputs/aaaa /c/uv ${INPUT_ROOT}/model") == []
    for other in ("/c", "/c/inputs", "/c/inputs/bbbb/data", "/c/store/files/x", "/c/inputs/aaaa-copy"):
        assert hits(f"ls {other}") == [other]
    assert hits("cat $INPUT_ROOT/../bbbb/x") != []
