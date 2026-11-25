# BERT-SQuAD reproduction plan

Recorded 2026-11-23 after successful bounded preflight.

## Workload and dependencies

The sole target, `bert-squad`, has three ordered stages in delivered `reproduce.sh`: `build`
creates a private source copy, cached Python 3.11 environment, and local LoadGen extension;
`preprocess` converts the complete fixed validation set into 10,833 BERT features; `run` checks
GPU memory, runs FP32 PyTorch LoadGen AccuracyOnly in default Offline mode, then invokes the
repository evaluator. The clean full replay is:

`OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from build`

`--from build` removes and reconstructs only wrapper-owned work and markers. With one target,
there is no independent CPU work to overlap with its GPU stage and no accelerator concurrency.

## Measured costs

| Stage | Preflight measurement | Resource evidence |
|---|---:|---|
| Build | 44.60 s | 526,928 KiB peak host RSS; offline cached packages and local C++ build |
| Preprocess | 60.67 s | complete 10,833 features; 887,380 KiB peak host RSS; CPU-only |
| GPU setup + 16-feature run + evaluation | about 40 s | 76,967 MiB free before run; `cuda:0`; evaluator completed |
| Timed LoadGen inference | 1.485 s / 16 | 0.093 s per feature from `mlperf_log_detail.txt` |

The bounded run covered the only input kind with the same full feature cache, checkpoint, FP32
backend, Offline scenario, response format, decoder, and evaluator as the full run.

## Resources, schedule, projection, and reserve

The machine exposes 128 logical CPU cores, about 1.95 TiB available host memory, and one
NVIDIA A100-SXM4-80GB. The wrapper requires at least 10,000 MiB GPU memory free and records the
actual value immediately before inference. Linear projection is
`1.485 / 16 * 10833 = 1005 s` (16.8 minutes). Allow 3 minutes for build, preprocessing, model
setup, decoding, and evaluation, plus 25 minutes for shared-GPU contention: planned completion is
45 minutes. A final 15-minute reserve is held for `report.json`, artifact validation, and replay
checks, leaving substantial margin within the two-hour limit.

After several percent of the full pass, compare process elapsed time and GPU contention against
the 45-minute window. Do not alter unrelated processes. If the projection exceeds 105 minutes,
start no new run and preserve the remaining 15 minutes for measured reporting and validation.

## Re-plan 1 — shared GPU contention

After the first several percent, `nvidia-smi` showed this run's PID 791041 using 1,814 MiB beside
an unrelated BERT process (PID 786518, 1,832 MiB). The output accuracy log reached 11.2 MB after
about two minutes of this process, roughly 17% of its size projected from bounded output. Memory
is not limiting and the unrelated process must not be changed. Revise the expected inference
window to 20–30 minutes to allow shared compute; the 45-minute plan and 15-minute reserve remain
adequate, so the schedule stays at one benchmark GPU run.

## Completion against plan

The unrelated process ended without intervention near the end of inference. The full LoadGen pass
and evaluator completed successfully, printing exact match `83.68968779564806` and F1
`90.87487229720105`; no further run or schedule change was required.
