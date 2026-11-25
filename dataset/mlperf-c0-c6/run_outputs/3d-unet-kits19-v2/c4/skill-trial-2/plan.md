# Full reproduction plan

## Workload

One target, `3d-unet`, runs three ordered stages. The environment stage creates
a Python 3.10 virtual environment from the required shared uv cache and builds
the repository's LoadGen. The CPU-only preprocessing stage prepares and verifies
all 43 entries in `meta/inference_cases.json`, with `case_00400` linked to its
documented duplicate `case_00185`. The run stage checks current GPU memory, runs
PyTorch/TorchScript fp32 Offline accuracy inference, then invokes the repository
evaluator. The accelerator stage depends on preprocessing, and only one GPU run
is launched.

## Measured costs

- Fresh one-case end-to-end preflight: 5:03 wall time under shared CPU
  contention, 60.3 s user time, 18.6 s system time, and 3,060,944 KiB peak RSS.
- Cached package install plus local LoadGen compilation completed successfully;
  LoadGen compilation itself took 54.6 s.
- One-case preprocessing produced shape `(1, 192, 384, 384)` and passed the
  repository checksum manifest.
- One-case inference processed 50 sliding-window sub-volumes in 5.31 s.
- GPU launch check measured 81,037 MiB free of 81,920 MiB.
- The one-case official evaluator completed and printed mean, kidney, and tumor
  Dice values.

The bounded input reduced both the case manifest and checksum manifest to the
same `case_00000`, so no repository input check was bypassed.

## Resources and schedule

The active execution context exposes 8 logical CPUs, about 2 TiB host RAM, one
shared NVIDIA A100 80GB, and 5.2 TiB free workspace storage. The wrapper uses
only the fixed shared uv cache with `UV_EXCLUDE_NEWER=2026-11-10`. CPU stages run
first; the single GPU stage begins only after `nvidia-smi` confirms at least 16
GiB free. There is no independent second target whose CPU work can overlap the
accelerator stage. Stage success markers make the run resumable without
repeating completed work.

## Expected time and reserve

The measured inference rate gives a lower bound near 4 minutes for 43 cases,
with additional variation from case dimensions. Shared CPU contention dominated
the preflight, so full preprocessing and evaluation are budgeted more
conservatively. The projected full run is 25--35 minutes. Ten minutes of the
one-hour limit is reserved for report generation, output validation, and a
fresh-copy deliverable check.

## Re-plan after initial full-run progress

The full environment and preprocessing stages finished in roughly 3 minutes,
well ahead of the conservative estimate. During inference the accuracy log
reached 1.29 GB and grew to 1.36 GB in 3.2 seconds while the GPU remained at
100% utilization with about 6.9 GiB allocated. The prior measured full log was
about 2.7 GB, so inference projects to finish within several more minutes.
Stage order and parallelism remain unchanged; the overall projection is now
15--25 minutes, retaining the 10-minute closing reserve.

## Completion

The delivered no-stage script completed from nothing built in 6:31.89. All 43
cases ran, the official evaluator completed, and the required outputs were
written. Peak host RSS was 11,454,852 KiB. The measured evaluator line was
`Accuracy: mean = 0.86168, kidney = 0.9347, tumor = 0.7887`.
The copied wrapper also completed its environment stage from nothing built
beside a fresh repository copy, including LoadGen compilation and CUDA
availability validation.
