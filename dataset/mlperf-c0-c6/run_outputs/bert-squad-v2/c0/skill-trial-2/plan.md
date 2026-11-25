# Reproduction plan

## Workload

- Target `bert-squad`: build an isolated Python 3.11 environment; stage the pinned BERT source and supplied DeepLearningExamples commit; preprocess the SQuAD v1.1 validation set; run fp32 BERT-Large with PyTorch in LoadGen Offline AccuracyOnly mode; evaluate the LoadGen accuracy log with `accuracy-squad.py`.
- Inputs: 10,570 validation questions producing 10,833 QSL features, cached `model.pytorch`, cached `vocab.txt`, and the supplied DeepLearningExamples tree at commit `729963dd47e7c8bd462ad10bfac7a7b0b604e6dd`.
- Required outputs: `mlperf_log_accuracy.json`, `mlperf_log_detail.txt`, and evaluator stdout in `accuracy.txt`; all build, preprocessing, run, and evaluator logs remain in `bert-squad/out`.

## Measured costs and resources

- Environment build: approximately 30 seconds, CPU/filesystem only, using the mandated shared uv cache.
- Full preprocessing: 59.7 seconds wall time, one CPU core, 872,472 KiB maximum resident memory; generated 10,833 features.
- Eight-feature end-to-end preflight: 25.8 seconds wall time, 2,326,204 KiB maximum host resident memory; successful LoadGen run and repository evaluation.
- GPU check immediately before preflight: NVIDIA A100-SXM4-80GB with 79,323 MiB free of 81,920 MiB. The run stage uses the local GPU and no other run from this session overlaps it.

## Schedule and expected time

Run `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from build`. This recreates the environment and preprocessing cache, checks current GPU memory immediately before launch, runs all 10,833 features, and invokes the repository evaluator.

The measured setup cost is under two minutes. Model startup dominates the bounded preflight, so its per-feature extrapolation is not useful; the read-only prior snapshot from the same arm measured approximately 19 minutes for the identical full inference under shared-GPU contention. Allow 25 minutes for the full run plus setup, safely below the two-hour limit.

Reserve 20 minutes for one repair if needed, result extraction, `report.json`, required-artifact checks, and final standalone-script inspection.

## Re-plan triggers

- After the first several percent of the accuracy log is written, compare its growth rate with the 25-minute run projection.
- If the projection exceeds the 100-minute execution budget, record the observed rate and revise the schedule before continuing.
- If a failure occurs, repair it through `reproduce.sh --from <stage>` so downstream outputs and markers are invalidated by the script itself.

## Re-plan during full run

- Trigger: at 81 seconds of inference, the accuracy log contained 582 of 10,833 records (5.4%).
- Revised projection: the observed rate projects approximately 25 minutes for inference and about 27 minutes including the clean build and preprocessing stages.
- Schedule decision: continue the single serialized GPU run unchanged. The projection remains well below the 100-minute execution budget and retains the 20-minute reporting/repair reserve.

## Completion

- The clean build-through-evaluation invocation completed successfully in 25:08.47 wall time.
- LoadGen reported no warnings or errors; the accuracy log contains 10,833 unique QSL records spanning indices 0 through 10,832.
- Repository evaluator result: exact match `83.68968779564806`, F1 `90.87487229720105`.
