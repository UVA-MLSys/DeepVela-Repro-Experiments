# Reproduction plan

## Workload and dependencies

Target `bert-squad` is one four-stage pipeline:

1. `environment`: stage the supplied source and pinned DeepLearningExamples, create a Python 3.11 uv environment from the mandated cache/cutoff, add the missing PyTorch SUT only to the private copy, and build local LoadGen.
2. `preprocess`: convert the complete SQuAD v1.1 validation set to a saved feature cache.
3. `run`: record current A100 memory/process state, load the fixed FP32 checkpoint, and run all feature windows through LoadGen Offline AccuracyOnly.
4. `evaluate`: decode the accuracy log with `accuracy-squad.py` and run `evaluate_v1.1.py` to write `accuracy.txt`.

Each stage has a success marker and depends on the preceding stage. The measured clean replay will use `./reproduce.sh --target bert-squad --from environment`, which clears and rebuilds only script-owned state and outputs.

## Preflight measurements

All measurements are from this workspace on 2026-11-23:

| Stage | Measured cost | Resource evidence |
|---|---:|---|
| environment | 77 s | uv cache; local LoadGen compile; CUDA imports successful |
| preprocess | 59.44 s | CPU-only; full dataset produced 10,833 feature windows |
| 16-window end-to-end | 29.92 s | A100 plus evaluator; finite EM/F1 and required device log |
| 256-window end-to-end | 45.17 s | A100 plus evaluator; finite EM/F1 |
| LoadGen inference, 256 windows | 23.79 s | 10.76 features/s from detail-log timestamps |

The bounded 256-window result was `exact_match=92.1875`, `f1=93.76302083333331`; it is correctness and sizing evidence only, not the reportable full result.

## Resources and schedule

The host exposes 128 logical CPUs, about 2 TiB RAM, and one shared NVIDIA A100 80GB. Immediately before both preflight GPU runs, 70,620 MiB was free while five unrelated processes were active and GPU utilization was 100%. Batch size 16 succeeded under that state, so it is retained to limit shared-memory pressure.

There is one target, so environment, preprocessing, GPU inference, and evaluation run serially. This reproduction starts no second GPU task while inference runs. The script checks free memory again immediately before the full GPU stage. No unrelated process will be modified or canceled.

## Projection, reserve, and re-plan trigger

At the current measured rate, full inference is `10833 / 10.76 = 1007 s` (16.8 minutes). Adding model startup, the measured 136 seconds for clean environment plus preprocessing, evaluation, and filesystem/contention variance gives a 25-minute central estimate and a 40-minute conservative projection. This fits the two-hour limit.

A 15-minute reserve is held for artifact verification, `report.json`, and one bounded repair if required. After initial full-run progress, the detail/run timestamps will be compared with 10.76 features/s. If the projection passes 105 minutes from full-run launch, no new work will begin; available measured outputs will be recorded per the task's no-human-response rule.
