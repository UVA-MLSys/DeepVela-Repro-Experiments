# Reproduction plan

## Workload

One target, `bert-squad`, with four dependency-ordered stages implemented by the delivered
`reproduce.sh`: `setup -> preprocess -> run -> evaluate`. Setup copies the supplied source,
creates the Python environment from the mandated cache, and builds LoadGen. Preprocess creates
the fixed 10,833-feature cache from the complete SQuAD v1.1 validation set. Run executes the
PyTorch FP32 AccuracyOnly/Offline target on the local A100. Evaluate invokes the supplied
`accuracy-squad.py`/`evaluate_v1.1.py` route and writes `accuracy.txt`.

## Preflight measurements (2026-11-22)

- Setup: approximately 40 seconds wall time; 8 CPU cores available; GPU unused.
- Preprocess: approximately 60 seconds for the full dataset; 10,833 features from 10,570
  questions; GPU unused.
- Accelerator preflight: eight features completed in 1.130 seconds of LoadGen time, or about
  0.141 seconds/feature. The whole run stage, including model construction/checkpoint loading,
  took approximately 16.5 seconds for those eight features.
- Evaluator preflight: approximately 5 seconds for the bounded output and printed
  `{"exact_match": 75.0, "f1": 75.0}`. This value is only bounded-run evidence and is not the
  reportable baseline.
- Resources observed immediately before the bounded GPU run: NVIDIA A100-SXM4-80GB, 81,920 MiB
  total and about 72,738 MiB free; host had 8 logical CPUs and about 1.9 TiB available RAM.

Evidence: `bert-squad/out/{build.log,preprocess.log,gpu-check.log,run.log,evaluator.log}` and
`bert-squad/out/mlperf_log_detail.txt` (`effective_accuracy_sample_count=8`, logger/power interval
1.130 seconds).

## Schedule and projection

There is one target, so setup and preprocessing run first on CPU, followed by exactly one local
GPU run, followed by CPU evaluation. There is no second ready target whose CPU work can overlap
the accelerator stage. The accelerator stage is serialized as required.

At the measured 0.141 seconds/feature, 10,833 features project to about 1,530 seconds (25.5
minutes). Adding model loading, setup, preprocessing, full post-processing/evaluation, and file
checks gives a conservative total projection of **29 minutes**. The two-hour limit therefore
leaves more than 80 minutes; reserve **10 minutes** after execution for `report.json`, artifact
checks, and the required standalone clean-copy replay check.

## Re-plan triggers

After the first few percent of samples, compare elapsed rate with 0.141 seconds/feature. If the
projection exceeds 100 minutes (the wall limit less the 10-minute reserve and already consumed
setup time), record a re-plan here before changing execution. Any failure returns the target to
the earliest affected stage using `./reproduce.sh --from <stage>`; downstream markers and outputs
are cleared by the wrapper itself.
