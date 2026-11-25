# BERT-SQuAD reproduction plan

## Workload

One target, `bert-squad`: build a target-local Python environment and LoadGen,
preprocess all 10,570 SQuAD v1.1 questions into 10,833 feature windows, run the
BERT-Large fp32 PyTorch CUDA SUT in Offline AccuracyOnly mode, then run the
repository-owned evaluator.

## Measured costs and resources

- Current clean two-example end-to-end preflight: about 2 minutes, including
  environment build, LoadGen compilation, full preprocessing, model load, CUDA
  inference, and evaluator.
- Current preprocessing produced 10,833 feature windows.
- The permitted prior trial measured 36.5 seconds for a 100-sample run with the
  environment and features ready, including fixed model-load/evaluator overhead,
  and about 6.1 windows/s during its full run.
- Host: 8 visible CPU cores and about 1.9 TiB available RAM.
- GPU: one shared NVIDIA A100-SXM4-80GB. The current preflight snapshot showed
  75,540 MiB free immediately before inference, enough for the validated run.

## Schedule and projection

Run `./reproduce.sh --target bert-squad --from build` once with the requested
target output directory. Build and preprocessing are CPU-only and precede the
sole GPU stage. No other GPU work is launched.

The measured full-run rate projects about 30 minutes for inference. Treating the
entire 36.5-second 100-sample measurement as linear gives a conservative upper
projection of about 66 minutes, plus about 2 minutes to rebuild and preprocess.
Both fit the 2-hour wall-clock limit. Reserve 15 minutes for evaluator completion,
`report.json`, required-artifact checks, and a standalone-script inspection.

## Re-plan at 8.8% completion

The fresh accuracy log contained 957 of 10,833 feature-window results at
16:01:58, about 86 seconds after the GPU stage started. The observed rate of
roughly 11 windows/s projects approximately 16--18 minutes for inference despite
other visible GPU processes. This is faster than the original 30--66 minute
envelope, so stage order and parallelism remain unchanged and the 15-minute
reporting reserve is retained.

## Re-plan after standalone inspection

The first full run succeeded with an absolute `OUTPUT_DIR`, but inspection found
that a relative `OUTPUT_DIR` would be reinterpreted after the benchmark changes
directories. Canonicalize the created output directory in the delivered script,
then repeat the full clean-build run with `OUTPUT_DIR=./bert-squad/out`. The first
run took about 21 minutes including evaluation, so the repeat remains well inside
the two-hour limit and the reporting reserve.
