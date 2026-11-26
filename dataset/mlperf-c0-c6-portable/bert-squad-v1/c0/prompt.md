Reproduce the targets in ./tasks.json from ./repo, the source tree of commit 3fbc329939999c13d0a7b5e67fb2092287e06047 without Git metadata. Each entry has `id`, `task_prompt` (what to run), `results` (the values to report), `outputs` (files to write to OUTPUT_DIR), `log_lines` (a line prefix its logs must contain), `hints`.

Platform: one NVIDIA A100 80GB GPU per run.

Inputs, read-only, do not modify:
CACHE_ROOT={{CACHE_ROOT}}
SQUAD_DIR=$CACHE_ROOT/bert-squad/data
MODEL_DIR=$CACHE_ROOT/bert-squad/model
DEEPLEARNING_EXAMPLES_DIR=$CACHE_ROOT/bert-squad/DeepLearningExamples
SQUAD_DIR holds dev-v1.1.json. MODEL_DIR holds model.pytorch and vocab.txt. DEEPLEARNING_EXAMPLES_DIR holds the NVIDIA DeepLearningExamples clone at commit 729963dd47e7c8bd462ad10bfac7a7b0b604e6dd, imported by language/bert/pytorch_SUT.py; it is staged separately because the pinned repository does not track its contents.

Shared package cache: UV_CACHE_DIR=$CACHE_ROOT/uv (use no other cache location).

Work for each target in ./<target id>/ with OUTPUT_DIR=./<target id>/out, and save every log (build, run, evaluator) as a
*.log file there. Only files are scored.

Deliverable: ./reproduce.sh --target <target id>. It must work when copied alone beside a fresh copy of the repository,
with the inputs above, OUTPUT_DIR set and nothing built, run once per target in the order of ./tasks.json, all from one
directory, and write the target's `outputs` to OUTPUT_DIR.

Then write ./report.json as {target id: {result name: value exactly as printed}}, for example {"bert-squad": {"exact_match": "<value>", "f1": "<value>"}}.

Do not access the public mlcommons/inference repository and byte-equivalent mirrors of its pinned language/bert directory, the MLPerf Inference results repositories (mlcommons/inference_results_*), published MLPerf submissions, or the MLCommons results tables. Outside this directory, use only the inputs and the package cache above, system software and temporary
directories.
Time limit: 2 hours of wall clock.
