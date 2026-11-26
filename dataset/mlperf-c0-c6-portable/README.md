# MLPerf C0–C6 portable bundle: BERT v1 C0, C2 and C4

**This folder contains reference answers.** Do not make it readable by a benchmark agent.

A new version of three tasks from the [C0–C6 study](../mlperf-c0-c6/README.md) whose prompts and manifests hold `{{CACHE_ROOT}}` in place of a machine's cache directory. The other conditions, BERT v2 and 3D U-Net are not included. No benchmark has been run with this bundle, so do not pool its results with the study's without showing they are comparable.

## Differences from the study bundle

- `{{CACHE_ROOT}}` replaces the original cache directory.
- The `Platform:` line states only the hardware. `evaluation.prepare` appends the scheduling and sharing instructions.
- C4 gives `SQUAD_DIR` as `bert-squad/data-c4`, as its `masked.txt` and the study manifest state. The study's v1 and v2 C4 prompts give `bert-squad/data`, which also holds `evaluate-v1.1.py`.
- In C2 and C4, the pattern for the unmasked DeepLearningExamples clone no longer ends in `$`, which a tool call's JSON text does not end with.
- Task digests differ from the study's.

`tasks.json`, `masked.txt`, the source archive, the rubrics and the DeepLearningExamples patch files are copies.

## Use

The cache directory must hold the inputs each prompt names, under `bert-squad/`. They are not included, and no script builds the masked copies (`DeepLearningExamples-c2`, `DeepLearningExamples-c4`, `data-c4`).

1. Copy this folder outside the repository, the runs root and the cache directory.
2. Set `EVALUATION_BUNDLE_DIR` to the copy, `REPRODUCTION_RUNS_DIR` to the runs root, and `REPRODUCTION_CACHE_DIR` to the cache directory (default: `.cache` in the repository root).
3. Run `python -m evaluation.prepare reproduce bert-squad-v1/<condition> --arm <arm> --compute local`.

## Access rule

Within the cache directory, a run is allowed its task's inputs and `uv/`. Keep only inputs, packages and source clones there. The [monitor's limits](../../skills/agent-evaluation-profile/references/rubric-authoring.md#cold-and-warm-cache-conditions) apply: review flagged calls.
