# MLPerf C0–C6 portable bundle: BERT v1 and 3D U-Net v1

**This folder contains reference answers.** Do not make it readable by a benchmark agent.

A new version of the seven BERT v1 and seven 3D U-Net v1 tasks of the [C0–C6 study](../mlperf-c0-c6/README.md). Prompts and manifests name no machine's paths. The v2 studies, which stage earlier runs' artifacts, are not included. No benchmark has been run with this bundle, so do not pool its results with the study's without showing they are comparable.

## Differences from the study bundle

- Inputs come from a verified input set, `{{INPUT_ROOT}}`, and each manifest pins it with `input_set_sha256`. All prompts of a benchmark name the inputs alike.
- BERT: C0, C1, C3, C5 and C6 share one set; C2 and C4, which also mask DeepLearningExamples, have their own. 3D U-Net: all conditions share one set, because its conditions mask only the source archive.
- 3D U-Net's set holds the 42 stored cases and the `.ptc` model; it leaves out `3dunet_kits19_pytorch_checkpoint.pth`, which the study's model folder also held.
- The BERT sets hold DeepLearningExamples without `.git/`, and leave out SQuAD's `evaluate-v1.1.py`, which the study's data folder also held. The study's v1 and v2 C4 prompts gave that folder, although the C4 mask and manifest name a folder without the evaluator.
- `{{PACKAGE_CACHE}}` replaces the package cache path.
- The `Platform:` line states only the hardware. `evaluation.prepare` appends the scheduling and sharing instructions.
- C2 and C4 forbid "unmodified copies of the DeepLearningExamples input" and drop the manifest pattern that named the unmasked clone's path. The access rule flags other input sets and the downloads.
- Task digests differ from the study's.

`masked.txt`, the source archives, the rubrics and the DeepLearningExamples patch files are copies, and `tasks.json` holds the same records.

## Use

Set `REPRODUCTION_CACHE_DIR` if the cache is not `.cache` in the repository root, for both commands.

1. Build the input sets: `python -m evaluation.warmup mlperf-c0-c6/bert-squad --download` (about 1.35 GB of files and one commit of DeepLearningExamples; each set takes a copy of the model) and `python -m evaluation.warmup mlperf-c0-c6/3d-unet-kits19 --download` (about 4.1 GB). `--from <dir>` imports files you already have.
2. Copy this folder outside the repository, the runs root and the cache directory.
3. Set `EVALUATION_BUNDLE_DIR` to the copy and `REPRODUCTION_RUNS_DIR` to the runs root.
4. Run `python -m evaluation.prepare reproduce <bert-squad-v1 or 3d-unet-kits19-v1>/<condition> --arm <arm> --compute local`.

`prepare` verifies the set before a launch. Freezing the run verifies it again, and the scorer refuses a run whose set changed. Python's `__pycache__/*.pyc` files are not counted as changes. Runs share sets, so a change found after one run may come from another that used the set at the same time.

## Access rule

Within the cache directory, a run is allowed its input set and `uv/`. The [monitor's limits](../../skills/agent-evaluation-profile/references/rubric-authoring.md#cold-and-warm-cache-conditions) apply: review flagged calls.
