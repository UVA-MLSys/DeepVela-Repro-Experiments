# MLPerf C0–C6 masking study

**This folder contains reference answers** (rubric targets, published values in some masked sources, and agent reports). Do not make it readable by a benchmark agent.

It holds the task bundles of the study, 28 tasks, and a subset of the outputs of its earlier runs. The bundles name no machine's paths. They are a new version of the bundles those runs used, which are in git at commit `c7b91b9` and earlier.

## Contents

- [Studies](#studies)
- [Layout](#layout)
- [Running a task](#running-a-task)
- [Differences from the bundles of the earlier runs](#differences-from-the-bundles-of-the-earlier-runs)
- [Run outputs](#run-outputs)

## Studies

Each version has seven masking conditions (C0–C6), three workflows (`plain`, `skill`, `builder-skill`) and three trials: 63 runs. In v2, each run also receives the files of the v1 run with the same condition, workflow and trial.

| Here | Prior artifacts | Name in run records and analysis scripts |
|---|---|---|
| `3d-unet-kits19-v1` | no | `3d-unet-kits19-v4` |
| `3d-unet-kits19-v2` | yes, from v1 | `3d-unet-kits19-v5` |
| `bert-squad-v1` | no | `bert-squad-v1` |
| `bert-squad-v2` | yes, from v1 | `bert-squad-v2` |

`dataset/paper_runs.json`, `dataset/tasks/mlperf-c0-c6/` and the scripts under `docs/MLSys/` still use the names in the last column. `3d-unet-kits19-v1` and `-v2` there refer to older studies that are not in this folder.

## Layout

```text
bundle.yaml, MANIFEST.sha256
<version>/<condition>/        prompt.md, tasks.json, masked.txt, manifest.yaml, source.tar.gz, rubrics/
                              BERT C2 and C4 also: dle-mask.patch, dle-files.sha256 (the DeepLearningExamples mask)
run_outputs/<version>/
  index.json                  one entry per run: condition, workflow, trial, reported result, source run folder
  <condition>/<workflow>-trial-<n>/
                              reproduce.sh, report.json, evaluation.json, checks.json, plan.md
```

The profile id is `reproduce`; a task id is `<version>/<condition>`, for example `bert-squad-v1/c0`.

## Running a task

1. Answer the deployment questions: `python -m evaluation.setup --init` ([local preferences](../../docs/local-preferences.md)).
2. Build the input sets: `python -m evaluation.warmup mlperf-c0-c6/bert-squad --download` (about 1.35 GB of files and one commit of DeepLearningExamples; each set takes a copy of the model) and `python -m evaluation.warmup mlperf-c0-c6/3d-unet-kits19 --download` (about 4.1 GB). `--from <dir>` imports files you already have.
3. Copy this folder outside the repository, the runs root and the cache directory, and give the copy as the bundle root. `evaluation.prepare` rejects a bundle root that overlaps any of them.
4. Run `python -m evaluation.prepare reproduce <bert-squad-v1 or 3d-unet-kits19-v1>/<condition> --arm <arm> --compute local`.

For v2, launch each v1 run with `--trial <n>` and let it finish and be frozen first. Then `prepare … <benchmark>-v2/<condition> --arm <arm> --trial <n>` copies that run's retained files read-only into `ws/prior-artifacts/`. It accepts only a frozen run launched against this bundle's v1 task; when several match, name one with `--prior-run <run dir>`.

`prepare` verifies the input set before a launch. Freezing the run verifies it again, and the scorer refuses a run whose set changed. Python's `__pycache__/*.pyc` files are not counted as changes. Runs share sets, so a change found after one run may come from another that used the set at the same time.

Within the cache directory, a run is allowed its input set and `uv/`. The [monitor's limits](../../skills/agent-evaluation-profile/references/rubric-authoring.md#cold-and-warm-cache-conditions) apply: review flagged calls.

No benchmark has been run with these bundles, so do not pool new results with those in `run_outputs/` without showing they are comparable.

## Differences from the bundles of the earlier runs

The rubrics, `masked.txt`, source archives and DeepLearningExamples patch files are unchanged, and `tasks.json` holds the same records. Prompts and manifests changed:

- Inputs come from a verified input set, `{{INPUT_ROOT}}`, and each manifest pins it with `input_set_sha256`. All prompts of a benchmark name the inputs alike.
- BERT: C0, C1, C3, C5 and C6 share one set; C2 and C4, which also mask DeepLearningExamples, have their own. 3D U-Net: all conditions share one set, because its conditions mask only the source archive.
- 3D U-Net's set holds the 42 stored cases and the `.ptc` model; it leaves out `3dunet_kits19_pytorch_checkpoint.pth`, which the earlier runs' model folder also held.
- The BERT sets hold DeepLearningExamples without `.git/`, and leave out SQuAD's `evaluate-v1.1.py`, which the earlier runs' data folder also held. The earlier v1 and v2 C4 prompts gave that folder, although the C4 mask and manifest name a folder without the evaluator.
- `{{PACKAGE_CACHE}}` replaces the package cache path.
- The `Platform:` line states only the hardware. `evaluation.prepare` appends the scheduling and sharing instructions.
- C2 and C4 forbid "unmodified copies of the DeepLearningExamples input" and drop the manifest pattern that named the unmasked clone's path. The access rule flags other input sets and the downloads.
- A v2 task names its v1 task (`prior_artifacts.source_task`) instead of the earlier run folders, and stages your own v1 run.
- Task digests differ from those the earlier runs recorded.

## Run outputs

`run_outputs/` keeps a subset of each earlier run: the delivered script, report, scoring files and plan. `plain` runs have no `plan.md`. Session logs, execution logs, evaluator logs and the rest of each `prior-artifacts/` folder are not included, so the runs may not be re-scorable from these files alone.

- 3D U-Net task ids were renamed (v4 to v1, v5 to v2) in their scoring outputs. Absolute paths to the original run folders keep the original names.
- `reproduce.sh`, `plan.md` and `report.json` are unchanged from the runs.
- `evaluation.json` records the digest of the bundle the run used, which is in git at commit `c7b91b9`, not the digest of the bundles in this folder.
