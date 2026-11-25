# Full reproduction plan

Recorded 2026-11-19 after the successful bounded end-to-end preflight.

## Workload and dependencies

Target `3d-unet` has four stages in `reproduce.sh`: `env` builds a Python 3.11 environment and the
supplied LoadGen; `data` constructs the 43-entry read-only raw-data view and preprocesses it; `gpu`
loads all preprocessed cases and runs the FP32 TorchScript model in Offline accuracy mode on the
local A100; `evaluate` runs `accuracy_kits.py` on that run's LoadGen accuracy log. Each stage reads
the previous stage's files and writes a success marker only after all commands succeed.

## Preflight measurements

The bounded workload was the first manifest entry (`case_00000`), with the inference manifest and
published checksum manifest reduced consistently to that entry.

| Stage | Measured wall time | Resource evidence |
|---|---:|---|
| Environment/build | 70 s | Python packages resolved offline from required uv cache; supplied LoadGen built locally |
| Preprocessing | 23 s | 8 CPU processes; output pickle 141,558,019 bytes; repository MD5 matched |
| GPU | 30 s total | A100 free memory 78,276 MiB before run; model work 4.29982 s for 50 windows; required device line was `cuda:0` |
| Evaluation | 6 s | 8 CPU processes; evaluator completed and printed all three metrics |

The repaired preflight reported no LoadGen warnings or errors and measured `mean=0.90521`,
`kidney=0.9700`, `tumor=0.8404` for its one case.

## Available resources

Immediately before planning: one shared NVIDIA A100-SXM4-80GB had 81,037 MiB free; the host had
1.8 TiB available RAM; the workspace filesystem had 4.7 TiB free. The script requires at least
20,000 MiB free immediately before its GPU stage and records `nvidia-smi`. It uses eight CPU
processes for preprocessing/evaluation and one serialized GPU run.

## Schedule and projection

Run the delivered command exactly once from a clean script-owned build state:
`OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet`. The script itself clears only its own
stage outputs and rebuilds from `env`; the repository and read-only inputs remain untouched.

There is one target, so there is no other ready CPU target to overlap with its GPU stage. Using the
one-case costs conservatively gives: environment 2 min; 43-case preprocessing 4 min (eight-way);
GPU 12 min (startup plus variable window counts at roughly the measured per-window rate); evaluator
3 min (eight-way plus multi-gigabyte JSON/NIfTI I/O). Expected total is 21 min. Hold a 10 min
reserve for report extraction and artifact validation, leaving 29 min contingency within the
one-hour limit.

## Re-plan triggers

After the first two full-run samples, project GPU duration from printed per-case times. Re-plan if
the total projection exceeds 50 min, if free GPU memory falls below 20,000 MiB before launch, or if
any stage fails. No new work starts once the 10-minute reserve is reached; measured results and
failure evidence then go into `report.json`.
