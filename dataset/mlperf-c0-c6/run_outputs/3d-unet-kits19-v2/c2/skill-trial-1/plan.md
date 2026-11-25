# 3d-unet reproduction plan

## Workload

One target: all 43 KiTS19 inference cases in `meta/inference_cases.json`, using the
fp32 TorchScript model, PyTorch backend, Offline scenario, accuracy mode, and the
repository's `accuracy_kits.py` evaluator. Stages are clean environment/build,
CPU preprocessing, one local-GPU LoadGen run, and CPU evaluation.

## Measured preflight costs

- Clean Python environment plus LoadGen build: approximately 75--85 seconds.
- One representative case (`case_00000`) preprocessing: approximately 20 seconds;
  output shape `(1, 192, 384, 384)`.
- GPU inference: 50 sub-volumes in 7.24311 seconds on the A100.
- One-case postprocessing/evaluation: under 10 seconds.
- Bounded evaluator result: mean 0.90521, kidney 0.9700, tumor 0.8404.
- GPU free memory immediately before the bounded launch: 67,487 MiB.

## Resources and schedule

The process sees 8 CPU cores, about 2 TiB host memory, and one shared NVIDIA A100
80GB. Preprocessing uses eight processes. GPU work is serialized and begins only
after a fresh `nvidia-smi` check; the script requires at least 12,000 MiB free.

The full target runs sequentially because inference consumes the completed
preprocessing set. Scaling the bounded preprocessing conservatively gives under
15 minutes, while prior same-arm measured artifacts record about 3.5 minutes for
all 43 GPU cases despite variable shapes. Allow 2 minutes for the build, 15 minutes
for preprocessing, 10 minutes for inference, and 3 minutes for evaluation: an
expected total under 30 minutes. A 10-minute reserve remains for reporting and
artifact validation inside the one-hour limit.

The LoadGen detail log may report the repository's known configuration warning
from loading both `mlperf.conf` and `user.conf`. These calls are part of the
documented `run.py` route and do not prevent accuracy mode or the repository-owned
evaluator from completing, so the full run retains them.

## Re-plan trigger

After the first full-run cases, compare observed inference time with the 10-minute
allowance. If projection threatens the reporting reserve, record the new projection
here before changing the schedule.

## Re-plan after full-run progress

All 43 GPU cases completed in roughly four minutes under shared-GPU contention.
Observed per-case times ranged from about 1.1 to 15.7 seconds. This is comfortably
inside the 10-minute inference allowance, so the schedule is unchanged and the
10-minute reporting reserve remains intact. CPU evaluation follows immediately.
