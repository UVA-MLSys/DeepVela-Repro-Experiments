# Full reproduction plan

Updated: 2026-11-19 16:02 EST, after successful bounded preflight.

## Workload and dependencies

The single `3d-unet` target runs four serial stages through the delivered `reproduce.sh`:

1. `environment` (CPU): Python 3.11 venv, pinned packages from the required uv cache, bundled
   LoadGen build, and target-local application instrumentation.
2. `data` (CPU/filesystem): stage the 43 fixed cases, represent documented duplicate
   `case_00400` with `case_00185`, preprocess, and retain checksum verification.
3. `gpu` (one local A100): check free memory, load the fp32 model, run Offline AccuracyOnly.
4. `evaluate` (CPU/filesystem): decode the LoadGen log and compute Dice with the repository evaluator.

## Preflight measurements

The bounded run used `case_00005`, retained its published checksum, and completed end to end.

| Stage | Measured evidence |
|---|---|
| environment | about 3m20s under concurrent system builds; 8 compile jobs maximum |
| data | about 24s; 8 workers; output shape `(1,256,448,448)`; checksum matched |
| gpu | about 23s stage; 13.44s inference over 108 subvolumes; `cuda:0` |
| evaluate | about 10s; benchmark evaluator produced a valid one-case metric line |

The A100 reported 81,920 MiB total, 9,115 MiB used, and 72,039 MiB free immediately before the
preflight. The wrapper requires 12,000 MiB free and aborts rather than oversubscribing. Same-arm
prior evidence over the identical source (apart from the same device-log instrumentation) completed
all 43 inference cases, with observed per-case times roughly 1.49–7.55 seconds and matching evaluator
outputs; it is estimation evidence only, not substituted measurement.

## Schedule and projection

Stages run serially because each consumes the preceding stage’s files. The GPU stage runs alone.
CPU stages use at most 8 workers, leaving host capacity to other sessions. There is no second target
whose CPU work could overlap this target’s GPU stage.

Conservative projection:

- environment: 4 minutes under current compilation contention;
- 43-case preprocessing and checksum verification: 5 minutes;
- GPU inference and LoadGen logging: 8 minutes;
- evaluator/postprocessing: 8 minutes;
- artifact validation and `report.json`: 2 minutes.

Projected total is 27 minutes. A 10-minute reserve is held before the one-hour limit. The launch is
driven by the current 13.44-second bounded inference and same-arm full-case range, leaving about 23
minutes beyond projection plus reserve.

## Re-plan triggers

After the first several full-run cases, compare observed rates with the eight-minute GPU projection.
If projected completion exceeds 50 minutes from launch, update this file and preserve completed stage
markers. A failure resumes with `./reproduce.sh --target 3d-unet --from <stage>` so only that stage and
its dependents are cleared by the wrapper. At the 10-minute reserve boundary, start no new work and
write the report from measured artifacts available.
