# Full reproduction plan

Updated: 2026-11-19 02:22 EST, after successful bounded preflight.

## Workload and dependencies

One target, `3d-unet`, runs four serial dependency stages through the delivered `reproduce.sh`:

1. `environment` (CPU): create a Python 3.11 venv, install pinned packages from the required uv
   cache, build the supplied LoadGen, copy and locally patch the application.
2. `data` (CPU and filesystem): stage the 43 fixed KiTS19 cases, duplicate `case_00185` logically as
   `case_00400`, and preprocess with the repository checksum verification enabled.
3. `gpu` (one local A100): check free memory, load the fp32 model, and run Offline AccuracyOnly.
4. `evaluate` (CPU and filesystem): decode the LoadGen accuracy log and compute the three Dice
   metrics with the repository evaluator.

## Preflight measurements

The bounded run used `case_00005`, the largest compressed imaging input in the inference manifest
(313,477,305 bytes). It retained the corresponding published checksum and completed end to end.

| Stage | Measured wall time | Resource evidence |
|---|---:|---|
| environment | about 40 s | package install plus local C++ LoadGen build, at most 8 compile jobs |
| data | 22.0 s | 8 CPU workers; one output pickle was 256,901,381 bytes; checksum matched |
| gpu | 25.9 s stage, 7.87 s model inference | A100 had 81,037 MiB free at launch; 108 subvolumes; device `cuda:0` |
| evaluate | 43.4 s | 8 CPU workers; one-case accuracy JSON was 102,760,497 bytes |

Host resources observed before planning: one NVIDIA A100-SXM4-80GB, 2.0 TiB RAM with roughly
1.8 TiB available, and 4.3 TiB free on the workspace filesystem. The wrapper requires at least
12,000 MiB free GPU memory immediately before launch and aborts rather than oversubscribing it.

## Schedule and projection

There is one target, so dependency stages run in order. The GPU stage runs alone. CPU environment,
preprocessing, and evaluation use at most 8 workers, leaving substantial host capacity for other
sessions. No other ready target exists to overlap with GPU work.

Conservative projection:

- environment: 1 minute;
- 43-case preprocessing: 43 times the measured largest-case duration divided by 8-way parallelism,
  plus I/O and checksum allowance: 4 minutes;
- GPU inference: 43 times the measured largest-case 7.87 seconds plus model/load/log allowance:
  8 minutes;
- evaluation: scale the largest 103 MB accuracy result across 43 cases with 8 workers and allow
  filesystem overhead: 8 minutes;
- final validation and `report.json`: 2 minutes.

Projected total is 23 minutes. A 10-minute reserve is held before the one-hour limit. The decision
to launch is driven by the measured worst-case input inference rate and still leaves approximately
27 minutes beyond projection plus reserve.

## Re-plan triggers

After the first several full-run cases, compare per-case inference time with 7.87 seconds. If the
projected completion exceeds 50 minutes from launch, preserve the existing run, update this file,
and reduce only CPU parallelism if GPU contention or filesystem saturation is the cause. Any stage
failure returns to that stage using `--from`; completed earlier stages remain marked and are not
repeated. At the 10-minute reserve boundary, start no new work and write the required report from
the measured artifacts available.
