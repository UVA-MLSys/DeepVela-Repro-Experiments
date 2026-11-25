# Reproduction plan

## Workload

One target, `bert-squad`, with four dependency-ordered stages implemented by the delivered
`reproduce.sh`: `setup -> preprocess -> run -> evaluate`. Setup copies the supplied source,
creates the Python environment from the mandated cache, and builds LoadGen. Preprocess creates
the fixed 10,833-feature cache from the complete SQuAD v1.1 validation set. Run executes the
PyTorch FP32 AccuracyOnly/Offline target on the local A100. Evaluate invokes the supplied
`accuracy-squad.py`/`evaluate_v1.1.py` route and writes `accuracy.txt`.

## Preflight measurements (2026-11-23)

- Setup: 49.57 seconds wall time; GPU unused.
- Preprocess: 62.50 seconds for the full dataset; 10,833 features from 10,570 questions;
  GPU unused.
- Accelerator preflight: eight features completed in a 0.359-second LoadGen interval. The whole
  stage, including model construction and checkpoint loading, took 12.54 seconds.
- Evaluator preflight: 8.47 seconds and printed `{"exact_match": 75.0, "f1": 75.0}`. This is
  bounded-run evidence only, not the reportable baseline.
- Resources immediately before the bounded GPU run: NVIDIA A100-SXM4-80GB, 81,920 MiB total
  and 79,323 MiB free. The host exposes 8 logical CPUs and about 1.9 TiB available RAM.

Evidence: `bert-squad/out/{build.log,preprocess.log,gpu-check.log,run.log,evaluator.log}` and
`bert-squad/out/mlperf_log_detail.txt` (`effective_accuracy_sample_count=8`).

## Schedule and projection

There is one target, so the full replay runs setup and preprocessing on CPU, followed by exactly
one local GPU run, followed by CPU evaluation. No second ready target exists whose CPU work could
overlap the accelerator stage. GPU work is serialized as required.

The current 8-feature interval extrapolates below 10 minutes, but a prior clean replay of the same
arm measured a conservative 0.141 seconds/feature, projecting 25.5 minutes for 10,833 features.
Using that slower rate plus setup, preprocessing, evaluation, and checks gives a conservative
**29-minute** total. Reserve **10 minutes** after execution for `report.json`, output validation,
and the standalone clean-copy replay check.

## Re-plan triggers

After the first few percent of samples, compare elapsed rate with 0.141 seconds/feature. If the
projection exceeds 100 minutes (the wall limit less the 10-minute reserve and time already used),
record a re-plan here before changing execution. Any failure returns the target to the earliest
affected stage using `./reproduce.sh --from <stage>`; the wrapper clears its own downstream
markers and outputs.

## Completion

The clean `--from setup` replay completed successfully in 1,513.67 seconds (25.23 minutes),
within the 29-minute projection. LoadGen produced all 10,833 feature records, and both the primary
evaluation and an independent evaluator replay printed `exact_match=83.68968779564806` and
`f1=90.87487229720105`. Required output and log checks passed; no re-plan was needed.
