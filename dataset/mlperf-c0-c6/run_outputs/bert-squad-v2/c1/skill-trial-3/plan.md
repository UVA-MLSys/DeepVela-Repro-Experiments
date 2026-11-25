# BERT-SQuAD reproduction plan

## Workload

One target, `bert-squad`: create a Python 3.11 environment and build this source tree's LoadGen; stage the pinned BERT source and read-only inputs; preprocess the complete SQuAD v1.1 validation split into 10,833 features; run the BERT-Large fp32 PyTorch backend in LoadGen Offline AccuracyOnly mode on the local A100; and invoke the repository-owned `accuracy-squad.py`, which calls `evaluate_v1.1.py`.

## Preflight measurements

- Clean environment creation, package installation, and LoadGen compilation: 70 seconds, CPU only.
- Complete input preprocessing: 63 seconds, CPU only; 10,833 generated features.
- Bounded accelerator run: 17 seconds including construction and checkpoint loading for eight samples. The shared GPU had 73,954 MiB free but reported 100% utilization from concurrent work immediately before launch.
- Bounded evaluator plus orchestration overhead: about 8 seconds.
- Whole bounded run: 2 minutes 38 seconds, maximum host RSS 2,267,800 KiB.

## Resources and schedule

The host exposes 8 CPU cores, about 1.9 TiB available host memory, and one shared NVIDIA A100-SXM4-80GB. The single target's dependent stages run in order. The script checks GPU memory immediately before the accelerator stage and requires at least 8 GiB free. No other GPU work is started by this reproduction. There is no second target with ready CPU work to overlap with inference.

## Expected time and reserve

The bounded run's accelerator timing includes a large fixed model-load cost, so direct linear scaling is deliberately conservative: 17 seconds / 8 samples x 10,833 features is about 23 minutes. Adding the measured 2 minutes 13 seconds for clean build and preprocessing, evaluator time, filesystem work, and shared-GPU contention gives a 35-minute projection. A 20-minute reserve is held for evaluator validation, required artifact checks, `report.json`, and script verification. The projected total is within the two-hour limit.

## Launch

Run the unchanged delivered script through its clean reset path:

```bash
OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from environment
```

This removes its environment, private LoadGen build, working copy, markers, and downstream scored files through the script itself, while retaining the separately named preflight evidence under `bert-squad/out/preflight/`.

## Re-plan at 4.6% of full inference

The first 501 of 10,833 samples were written in about 70 seconds (approximately 7.2 samples/s) while the shared GPU remained at 100% utilization. This projects roughly 25 minutes for full inference. It remains within the 35-minute overall projection and retains the 20-minute closeout reserve, so the serial schedule is unchanged.

## Completion

The accepted clean run completed in 24 minutes 34 seconds with a maximum host RSS of 2,323,508 KiB. LoadGen wrote all 10,833 accuracy entries and reported no warnings or errors. The repository evaluator printed `exact_match` 83.68968779564806 and `f1` 90.87487229720105.
