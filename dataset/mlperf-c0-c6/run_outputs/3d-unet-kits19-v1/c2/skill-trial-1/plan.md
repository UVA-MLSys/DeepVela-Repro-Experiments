# 3d-unet reproduction plan

## Workload

One target: the 43 KiTS19 inference cases in `meta/inference_cases.json`, using the
fp32 TorchScript model, PyTorch backend, Offline scenario, accuracy mode, and the
repository's `accuracy_kits.py` evaluator. Stages are clean environment/build,
CPU preprocessing, one local-GPU LoadGen run, and CPU evaluation.

## Preflight measurements

- Clean environment and LoadGen build: about 75 seconds.
- One representative case (`case_00000`) preprocessing: about 21 seconds; its
  output shape was `(1, 192, 384, 384)`.
- GPU inference: 50 sub-volumes in 5.05368 seconds on the A100; GPU memory free
  immediately before launch was 81,037 MiB.
- One-case postprocessing/evaluation: under 10 seconds.
- Bounded end-to-end evaluator result: mean 0.90521, kidney 0.9700, tumor 0.8404.

## Resources and schedule

The machine exposes 192 CPU cores, ample host storage and one shared A100 80GB.
Preprocessing uses eight processes to limit contention. GPU work is serialized
and starts only after a fresh `nvidia-smi` memory check; the script requires at
least 12,000 MiB free. No other GPU work is launched by this reproduction.

The full run is sequential within the target because inference depends on all
preprocessed volumes. Based on the bounded rate, allow 3-8 minutes for parallel
preprocessing, 5-15 minutes for inference (case shapes vary), and 1-3 minutes for
evaluation, plus the measured 75-second build. Expected total is under 30 minutes.
A 10-minute reserve is held for reporting and artifact verification within the
one-hour limit.

The LoadGen detail log's preflight error says that loading both the repository
`mlperf.conf` and `user.conf` is not valid for an official submission. Both calls
are made by the repository's documented `run.py` path and do not prevent accuracy
mode or the official evaluator from completing, so the full run retains them.

## Re-plan after full-run progress

All 43 GPU cases completed in about 3.5 minutes under shared-GPU contention.
Observed case times ranged from 0.97 seconds (8 sub-volumes) to 11.62 seconds
(144 sub-volumes). This is faster than the original 5-15 minute inference range,
so no scheduling change is needed and the completion projection is now well under
15 minutes total remaining. CPU evaluation follows immediately; the reporting
reserve remains intact.
