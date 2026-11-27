Reproduce the targets in ./tasks.json from ./repo, the source tree of commit 3fbc329939999c13d0a7b5e67fb2092287e06047 without Git metadata. Each entry has `id`, `task_prompt` (what to run), `results` (the values to report), `outputs` (files to write to OUTPUT_DIR), `log_lines` (a line prefix its logs must contain), `hints`.

Platform: one NVIDIA A100 80GB GPU per run.

Inputs, read-only, do not modify:
INPUT_ROOT={{INPUT_ROOT}}
KITS19_RAW_DIR=$INPUT_ROOT/data
MODEL_DIR=$INPUT_ROOT/model
KITS19_RAW_DIR holds the raw imaging and segmentation of the cases in meta/inference_cases.json, except case_00400, the documented duplicate of case_00185. MODEL_DIR holds 3dunet_kits19_pytorch.ptc.

Shared package cache: UV_CACHE_DIR={{PACKAGE_CACHE}} (use no other cache location).

Work for each target in ./<target id>/ with OUTPUT_DIR=./<target id>/out, and save every log (build, run, evaluator) as a
*.log file there. Only files are scored.

Deliverable: ./reproduce.sh --target <target id>. It must work when copied alone beside a fresh copy of the repository,
with the inputs above, OUTPUT_DIR set and nothing built, run once per target in the order of ./tasks.json, all from one
directory, and write the target's `outputs` to OUTPUT_DIR.

Then write ./report.json as {target id: {result name: value exactly as printed}}, for example {"3d-unet": {"mean": "<value>", "kidney": "<value>", "tumor": "<value>"}}.

Do not access the public mlcommons/inference repository or mirrors of it, the MLPerf Inference results repositories (mlcommons/inference_results_*), published MLPerf submissions, or the MLCommons results tables. Outside this directory, use only the inputs and the package cache above, system software and temporary
directories.
Time limit: 1 hour of wall clock.
