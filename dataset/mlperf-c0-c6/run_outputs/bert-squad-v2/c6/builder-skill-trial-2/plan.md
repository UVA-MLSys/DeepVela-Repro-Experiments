# Reproduction plan

Written 2026-11-23 after a successful bounded preflight.

## Workload and dependencies

The single `bert-squad` target has three ordered `reproduce.sh` stages:

1. `environment`: verify immutable input hashes, copy the fresh source, stage the pinned DeepLearningExamples tree, create Python 3.11 environment, and build/install LoadGen.
2. `preprocess`: convert the fixed SQuAD v1.1 validation split to BERT features with sequence length 384, query length 64, and stride 128.
3. `run`: check free GPU memory, run fp32 PyTorch BERT-Large in Offline AccuracyOnly mode, evaluate with `accuracy-squad.py`/`evaluate_v1.1.py`, and copy scored outputs.

## Measured costs

The successful four-question preflight measured 50.7 s for a cold environment, 5.3 s for preprocessing, and 32.5 s for the GPU and evaluator stage. It printed `exact_match=75.0` and `f1=75.0`; these bounded values are not reportable results. Peak host RSS was 0.51 GiB during the build, 0.40 GiB during preprocessing, and 2.27 GiB during the run. Immediately before GPU work, `nvidia-smi` reported 77,395 MiB free.

The admissible local prior snapshot records a 100-question preflight at about 24 s GPU and 6 s evaluation and estimates roughly 10,833 full validation features. Its prior clean full run completed within a 15-minute plan. Combining that scaling evidence with today's cold-stage measurements gives a conservative 15-minute projection for the clean full replay.

## Resources, schedule, and reserve

This machine exposes 8 CPU cores, one NVIDIA A100-SXM4-80GB, and about 1.9 TiB currently available host memory. There is one target, so its stages run serially; no other GPU work from this session will overlap it. The full replay starts with the delivered script's `--from environment` path, which clears and recreates its own state. The two-hour limit leaves a 20-minute reserve for extraction, `report.json`, output validation, and a clean-copy deliverable check. If observed full-run progress projects beyond 100 minutes total, preserve measured artifacts and re-plan before the reserve.
