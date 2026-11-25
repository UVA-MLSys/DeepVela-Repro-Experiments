# Reproduction plan

## Workload

- Target `bert-squad`: build an isolated Python/LoadGen environment, preprocess the fixed SQuAD v1.1 validation set, run BERT-Large fp32 with the PyTorch backend in LoadGen Offline AccuracyOnly mode, then invoke `accuracy-squad.py`/`evaluate_v1.1.py`.
- Inputs are the fixed `dev-v1.1.json`, `model.pytorch`, `vocab.txt`, and DeepLearningExamples commit supplied under `CACHE_ROOT`.
- The full dataset produces 10,833 sequence features. The required outputs are LoadGen's accuracy/detail logs and the evaluator stdout.

## Measured costs and resources

- Clean environment build: about 44 seconds, CPU/filesystem only; LoadGen 6.0.17 builds locally.
- Full preprocessing: about 43 seconds, CPU only; produces 10,833 cached features.
- Eight-feature end-to-end preflight: about 21 seconds including model loading and evaluation.
- One full 32-feature batch end-to-end: about 12 seconds including fixed startup/evaluation overhead, with no LoadGen warnings or errors.
- GPU before the batch-32 preflight: NVIDIA A100-SXM4-80GB, 74,306 MiB free; another job held the GPU at 100% utilization. Batch 32 completed without OOM.

## Schedule and projection

- Run the single target serially because all GPU work must run one at a time.
- Launch the delivered script with `--from env` and no sample bound, forcing a clean build, preprocessing, full GPU pass, and official evaluation.
- There are 339 batches at size 32. Project 6–10 minutes for inference under observed contention plus about 2 minutes for clean setup/preprocessing/evaluation, well within the two-hour limit.
- Hold 15 minutes in reserve for report extraction, artifact validation, and a clean-copy portability check. No CPU work remains to overlap with the only GPU run.

## Full-run re-plan and completion

- Trigger: immediately before the full GPU stage, the A100 had 81,038 MiB free and 0% utilization, versus 74,306 MiB free and 100% utilization during the batch-32 preflight.
- Change: no schedule or batch-size change was needed; the projection improved from 6–10 minutes of inference to a few minutes, with the 15-minute reporting reserve unchanged.
- Result: the clean run rebuilt the environment and all 10,833 features, completed LoadGen with no warnings or errors, and the official evaluator printed exact match 83.68968779564806 and F1 90.87487229720105.
