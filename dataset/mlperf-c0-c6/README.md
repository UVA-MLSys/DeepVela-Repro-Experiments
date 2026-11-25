# MLPerf C0–C6 masking study: before and after prior artifacts

**This folder contains reference answers** (rubric targets, published values in some masked sources, and agent reports). Do not make it readable by a benchmark agent.

## Contents

- [Studies](#studies)
- [Layout](#layout)
- [Scoring a new run](#scoring-a-new-run)
- [Differences from the source files](#differences-from-the-source-files)

## Studies

Each version has seven masking conditions (C0–C6), three workflows (`plain`, `skill`, `builder-skill`) and three trials: 63 runs. In v2, each run also received the files of the v1 run with the same condition, workflow and trial.

| Here | Prior artifacts | Name in run records and analysis scripts |
|---|---|---|
| `3d-unet-kits19-v1` | no | `3d-unet-kits19-v4` |
| `3d-unet-kits19-v2` | yes, from v1 | `3d-unet-kits19-v5` |
| `bert-squad-v1` | no | `bert-squad-v1` |
| `bert-squad-v2` | yes, from v1 | `bert-squad-v2` |

`dataset/paper_runs.json`, `dataset/tasks/mlperf-c0-c6/` and the scripts under `docs/MLSys/` still use the names in the last column. `3d-unet-kits19-v1` and `-v2` there refer to older studies that are not in this folder.

## Layout

```text
bundle.yaml, checkers/, MANIFEST.sha256
<version>/<condition>/        prompt.md, tasks.json, masked.txt, manifest.yaml, source.tar.gz, rubrics/
run_outputs/<version>/
  index.json                  one entry per run: condition, workflow, trial, reported result, source run folder
  <condition>/<workflow>-trial-<n>/
                              reproduce.sh, report.json, evaluation.json, checks.json, plan.md
```

`plain` runs have no `plan.md`.

## Scoring a new run

Set `EVALUATION_BUNDLE_DIR` to the absolute path of this folder. The profile id is `reproduce`; a task id is `<version>/<condition>`, for example `bert-squad-v1/c0`. All 28 tasks load with `evaluation.bundle.load_task` and match `MANIFEST.sha256`. A full prepare-run-score cycle has not been run from this copy.

Running a task also needs the inputs named in its manifest (dataset, model checkpoint), which are not included. Some manifests contain absolute paths from the original machine. The v2 manifests list the original v1 run folders as prior-artifact sources.

## Differences from the source files

- 3D U-Net task ids are renamed (v4 to v1, v5 to v2) in manifests and scoring outputs. Absolute paths to original run folders keep the original names. Bundle digests recorded in old evaluations therefore do not match this copy. `MANIFEST.sha256` was regenerated.
- `source.tar.gz`, `reproduce.sh`, `plan.md` and `report.json` are unchanged.
- Run outputs are a subset. Session logs, execution logs, evaluator logs and the rest of each `prior-artifacts/` folder are not included, so the old runs may not be re-scorable from these files alone.
