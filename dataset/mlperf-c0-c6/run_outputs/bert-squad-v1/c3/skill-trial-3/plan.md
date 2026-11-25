# BERT-SQuAD reproduction plan

## Workload

One target, `bert-squad`: build a target-local Python environment and LoadGen,
preprocess all 10,570 SQuAD v1.1 questions into 10,833 feature windows, run the
BERT-Large fp32 PyTorch CUDA SUT in Offline AccuracyOnly mode, then run the
repository-owned evaluator.

## Measured costs and resources

- Clean two-sample end-to-end preflight: 2m11s, including environment build,
  LoadGen compilation, full preprocessing, model load, CUDA inference, and evaluator.
- 100-sample run with environment and features ready: 36.5s wall time, 2.33 GB
  peak host RSS. This includes model load and evaluator overhead.
- Full preprocessing produced 10,833 feature windows.
- Host: 8 visible CPU cores and about 1.9 TiB available RAM.
- GPU: one shared NVIDIA A100-SXM4-80GB. The successful checks observed 70.7 GB
  free immediately before inference, sufficient for the roughly 8-10 GB footprint.

## Schedule and projection

Run `./reproduce.sh --target bert-squad --from build` once. Build and preprocessing
are CPU-only and precede the sole GPU stage. No other GPU work is launched.

The 100-sample measurement projects approximately 30-35 minutes when fixed model
load/evaluator overhead is separated from inference. Treating all 36.5s as linear
is a deliberately conservative upper projection of about 66 minutes for inference,
plus about 2 minutes to rebuild/preprocess. Both fit the 2-hour wall-clock limit.
Reserve 15 minutes for evaluator completion, `report.json`, artifact checks, and a
standalone-script inspection.

## Re-plan at 5.7% completion

The full run completed 622 of 10,833 feature windows in 102 seconds of the GPU
stage, about 6.1 samples/s. Six unrelated GPU processes were concurrently visible,
but this measured rate still projects approximately 29-30 minutes for inference.
The projection matches the primary plan, so stage order and parallelism remain
unchanged and the 15-minute reporting reserve is retained.
