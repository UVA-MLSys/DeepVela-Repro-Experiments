# 3D-UNet KiTS19 reproduction plan

## Workload

Target `3d-unet` runs four ordered stages: environment/setup, preprocessing of all 43 cases in
`meta/inference_cases.json`, fp32 PyTorch Offline accuracy inference on the local A100, and the
repository Dice evaluator. Setup links the read-only cache inputs into target-local work storage;
`case_00400` links to the documented duplicate `case_00185`.

## Preflight measurements

A bounded end-to-end run covered `case_00000`, `case_00185`, and duplicate `case_00400`.

- Setup resolved the pinned environment from the mandated uv cache. Repository LoadGen 6.0.17
  could not compile because system Python lacks `Python.h`; cached wheel 6.0.17 is used instead.
- Preprocessing: 26.53 seconds wall time, 2,560,076 KiB peak host RSS. All repository checksums
  matched exactly.
- GPU stage: 38.93 seconds wall time for 300 sliding-window subvolumes, 5,974,968 KiB peak host
  RSS. The A100 had 67,487 MiB free but was at 100% utilization from other sessions.
- Evaluation: 16.12 seconds wall time.

## Resources and schedule

The host exposes 192 logical CPUs, about 2 TiB RAM, and one shared NVIDIA A100 80GB. Preprocessing
uses eight processes. GPU work runs one invocation at a time and starts only after a fresh
`nvidia-smi` check reports at least 16,000 MiB free. There is one target, so no other local GPU
work is launched. CPU setup/preprocessing precedes inference; no independent target is available
to overlap.

The three-case measurements project about 6.3 minutes for preprocessing, 9.3 minutes for GPU
inference under the measured contention, and 3.9 minutes for evaluation. Allowing setup and
nonlinear volume sizes, the full projection is 30 minutes. Ten minutes are reserved for artifact
checks and `report.json`, well inside the one-hour wall-clock limit.

## Repairs established by preflight

The target-local benchmark copy applies three compatibility repairs without changing the fixed
backend, precision, data, scenario, model, preprocessing, or metric: SciPy's relocated Gaussian
API, the evaluator's omitted documented mean row, and the required model-parameter device log.
Failed preflight attempts and their logs remain under `3d-unet/preflight/out`.
