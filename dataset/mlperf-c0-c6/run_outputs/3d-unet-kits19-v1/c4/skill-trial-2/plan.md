# Full reproduction plan

## Workload

One target, `3d-unet`, runs in three ordered stages: build a Python 3.10
environment and the bundled LoadGen; preprocess and checksum all 43 entries in
`meta/inference_cases.json` (including the documented `case_00400` duplicate);
then run PyTorch/TorchScript fp32 Offline accuracy inference and the repository
evaluator. The run stage depends on the complete preprocessed dataset and uses
the A100; the other stages are CPU-only.

## Measured costs

- Cached environment installation plus local LoadGen compilation: about 55 s.
- One-case preprocessing and repository checksum verification: about 15 s with
  16 processes.
- One-case inference: 50 sliding-window sub-volumes in 7.05 s on the A100.
- One-case postprocessing/evaluation: a few seconds.
- Preflight peak availability at the GPU check: 60,677 MiB free of 81,920 MiB.

The preflight used `case_00000`; both the case manifest and checksum manifest
were reduced to that same case. Its preprocessing MD5 matched the repository
manifest and its end-to-end evaluator result was printed successfully.

## Resources and schedule

The machine provides 192 logical CPUs, about 1.5 TiB host RAM, one shared A100
80GB, and ample workspace storage. Package resolution is offline against the
required shared uv cache. Preprocessing and evaluator multiprocessing use 16
workers. GPU work is single-run only and starts after an immediate `nvidia-smi`
check requires at least 16 GiB free.

There is only one target, so its CPU stages run first and its GPU stage follows;
there is no independent second target whose CPU work can overlap inference.
Success markers make completed stages resumable without repeating them.

Re-plan after initial full-run progress: the accuracy log grew from 1.29 GB to
2.67 GB in about 77 s with the GPU at 100% utilization and 6.8 GiB allocated.
This projects the accelerator stage near the original five-minute lower bound,
so stage order and parallelism remain unchanged.

## Expected time and reserve

The measured one-case GPU rate gives a lower-bound full inference projection of
about 303 s (43 x 7.05 s). Allowing for varying case dimensions, shared-GPU
contention, full preprocessing, and evaluation, the planned full run is 15--20
minutes. Ten minutes of the one-hour limit is reserved for report generation,
artifact validation, and a clean-copy deliverable check.

## Completion

The full run completed all 43 cases, the evaluator completed over 43 prediction
rows plus its aggregate mean row, and all required outputs were non-empty. A
copied `reproduce.sh` also completed its environment stage from nothing built
beside a fresh repository copy, including local LoadGen compilation and CUDA
availability validation.
