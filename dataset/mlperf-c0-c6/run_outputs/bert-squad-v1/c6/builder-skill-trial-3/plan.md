# BERT-SQuAD reproduction plan

Recorded 2026-11-23T00:35:54-05:00 after successful bounded preflight.

## Workload and dependencies

One target, `bert-squad`, runs three ordered stages through the delivered `reproduce.sh`:

1. `build` copies the pinned BERT and LoadGen sources into `bert-squad/work`, creates a Python
   3.11 environment from the mandated offline uv cache, and compiles LoadGen.
2. `preprocess` depends on that environment and converts all 10,570 validation questions into
   10,833 fixed BERT features.
3. `run` depends on the feature cache, checks GPU free memory, runs LoadGen AccuracyOnly in the
   default Offline scenario over all 10,833 features, then invokes the repository evaluator.

The full replay is `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from build`.
`--from build` clears and reconstructs only the wrapper's private work/state, so it starts with
nothing built. No other target exists, so there is no safe accelerator overlap to schedule.

## Measured costs

| Stage | Bounded measurement | Resource evidence |
|---|---:|---|
| Build | 39.19 s | peak RSS 527,400 KiB; cached package install plus local C++ build |
| Preprocess | 59.84 s for complete dataset | 10,833 features; peak RSS 886,772 KiB; CPU-only |
| GPU setup + 8-feature run | about 22 s wall-clock | 76,967 MiB free immediately before run; `cuda:0` |
| Timed LoadGen inference | 1.293 s for 8 features | 0.162 s/feature from `mlperf_log_detail.txt` |
| Bounded evaluator | about 7 s | official evaluator returned EM=75.0, F1=75.0 on its bounded denominator |

The 8-feature check samples the benchmark's only input kind and uses the same full feature cache,
checkpoint, FP32 PyTorch backend, Offline scenario, LoadGen response format, decoder, and metric
implementation as the full run.

## Available resources and schedule

The machine exposes 8 CPU cores, about 2,000,541 MiB available host memory, and one NVIDIA
A100-SXM4-80GB. Immediately before planning, the GPU had 76,967 MiB free of 81,920 MiB; another
session was using compute (100% instantaneous utilization), so no assumption of exclusive GPU
throughput is made. The wrapper requires at least 10,000 MiB free before launch and runs only one
GPU pass. CPU build and preprocessing precede it because there are no other targets whose CPU
work could overlap safely.

## Expected time and reserve

Linear inference projection is `1.2927 / 8 * 10833 = 1,750.6 s`, or 29.2 minutes. Build,
preprocessing, model setup, decoding, and evaluation add roughly 3 minutes. The planned completion
window is 45 minutes, allowing about 13 minutes for shared-GPU contention. A further 15-minute
reserve is held for `report.json`, artifact validation, and script replay checks. Even a 2x
inference slowdown projects near 62 minutes plus setup, within the two-hour wall-clock limit.

## Re-plan triggers

After the first several percent of samples, compare LoadGen/log progress and GPU process elapsed
time against the 45-minute window. If failures occur, group by the first causal log line and
return to preflight repair. At 105 minutes elapsed, start no new run and use the remaining
15-minute reserve to preserve measured outputs and report the status.

## Re-plan 1 — shared GPU contention

At 2026-11-23T00:39-05:00, after the full run launched, `nvidia-smi` showed two independent
BERT accuracy processes: this run's PID 143055 at about 1,814 MiB and another session's PID
129558 at about 1,832 MiB. GPU utilization remained 100% with 77,377 MiB free. No memory repair is
needed and the unrelated process must not be changed. The expected inference window is revised
from 29.2 minutes to 55–65 minutes to allow near-even compute sharing; setup has already completed,
so total completion remains below the two-hour limit and no schedule change other than continued
single-run execution is warranted.

## Completion against plan

The competing process ended early and the full LoadGen pass completed at approximately
2026-11-23T00:43-05:00, substantially earlier than the conservative projection. The evaluator
then printed exact match `83.68968779564806` and F1 `90.87487229720105`. No additional run or
schedule change was needed.
