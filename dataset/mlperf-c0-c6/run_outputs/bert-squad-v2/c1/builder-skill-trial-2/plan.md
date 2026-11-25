# Full reproduction plan

Planned 2026-11-23 after the bounded end-to-end preflight required by `baseline-reproduce`.

## Workload

One target, `bert-squad`, runs four ordered stages through the delivered `reproduce.sh`: environment build, deterministic SQuAD feature preprocessing, direct local A100 inference through LoadGen in Offline AccuracyOnly mode, and the pinned SQuAD evaluator. Each stage writes a success marker only after its commands and output checks pass. The full run starts with `./reproduce.sh --target bert-squad --from env`, which clears and recreates the wrapper's own staged environment and all downstream outputs.

## Measured costs

The bounded preflight used 8 LoadGen samples while covering the sole dataset/model/backend/scenario/evaluator route. Preprocessing still converted the complete input into 10,833 features.

| Stage | Preflight wall time | Peak host RSS | Accelerator |
|---|---:|---:|---|
| Environment | 43.26 s | 514 MiB | none |
| Preprocess | 60.22 s | 851 MiB | none |
| GPU inference (8 features) | 15.84 s | 2.22 GiB | A100, direct |
| Evaluate (8 questions) | 7.86 s | 829 MiB | none |

The bounded evaluator returned `{"exact_match": 75.0, "f1": 75.0}` and the GPU log contains `pytorch device: cuda:0`. These values are preflight evidence only, not reportable full-run results.

## Resources and schedule

This process sees 8 CPU cores, about 1.9 TiB available host memory, and one shared NVIDIA A100-SXM4-80GB. The wrapper requires at least 8 GiB free and records `nvidia-smi` immediately before inference. There is one target, so stages run in dependency order and no second GPU run is launched. No independent ready CPU work exists to overlap with inference.

The full run budget is conservatively 20 minutes: about 2 minutes for the measured rebuild/preprocessing/evaluator fixed costs and up to 18 minutes for 10,833 sequential feature inferences under shared-GPU contention. A further 10-minute reserve is held for output validation, clean-replay checks, and `report.json`. This projects completion within 30 minutes, leaving 90 minutes before the two-hour wall-clock limit.

## Re-plan triggers

After early full-run progress or any failure, compare elapsed rate with the 20-minute run budget. If projected run time exceeds 110 minutes total, stop launching new work and preserve measured artifacts/report state. Any repair restarts through the script's `--from` mechanism, never by deleting outputs manually.

## Re-plan 1 — early full-run rate

At 2026-11-23 13:53:03, the accuracy log contained 121 completed samples about 17 seconds after inference began, or approximately 7.1 samples/s while six GPU processes shared the A100. The revised inference projection is 25.4 minutes and the revised total projection is about 28 minutes. No schedule or implementation change is needed: there is only one target, GPU concurrency within this session remains one, and the projection remains well below the 110-minute reserve trigger.
