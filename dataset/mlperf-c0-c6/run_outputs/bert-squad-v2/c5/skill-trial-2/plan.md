# Reproduction plan

## Workload

- Target `bert-squad`: build an isolated Python/LoadGen environment, preprocess the fixed SQuAD v1.1 validation set, run BERT-Large fp32 with the PyTorch backend in LoadGen Offline AccuracyOnly mode, then invoke `accuracy-squad.py`/`evaluate_v1.1.py`.
- Inputs are the supplied immutable `dev-v1.1.json`, `model.pytorch`, `vocab.txt`, and DeepLearningExamples commit under `CACHE_ROOT`.
- Preflight established that the full dataset produces 10,833 sequence features. Required outputs are LoadGen's accuracy/detail logs and evaluator stdout.

## Measured costs and resources

- Clean bounded preflight: about 115 seconds total from nothing built.
- Environment creation and local LoadGen build: about 60 seconds, CPU/filesystem only.
- Full preprocessing: about 50 seconds, CPU only; 10,833 cached features.
- Eight-feature CUDA inference and evaluation: a few seconds after model startup.
- GPU immediately before preflight inference: NVIDIA A100-SXM4-80GB, 75,666 MiB free and 100% utilization from shared work. Batch size 32 remains far below available memory and was validated by the provenance-checked prior trial.

## Schedule and projection

- Run the only target serially because all GPU work must run one at a time.
- Launch the delivered script with `--from env` and no sample bound, forcing a clean environment build, full preprocessing, full GPU pass, and official evaluation.
- There are 339 batches at size 32. The verified prior trial projected 6–10 minutes for inference under contention; adding about two minutes for clean setup/preprocessing/evaluation gives a projection below 12 minutes.
- Hold 15 minutes in reserve for report extraction, artifact validation, and portability checks. No independent CPU work remains to overlap with the single GPU run.

## Re-plan triggers

- Compare observed progress and GPU state at the beginning of the full run with this projection.
- If a failure occurs, repair through `reproduce.sh --from <stage>` so the wrapper itself clears invalidated outputs and markers.

## Full-run re-plan and completion

- Trigger: full GPU inference exceeded the prior 10-minute upper estimate under persistent shared 100% GPU utilization.
- Change: revised total projection to about 15 minutes. The schedule remained one uninterrupted GPU run because there is only one target and no remaining CPU work; the two-hour limit and 15-minute reserve remained safe.
- Result: the clean run rebuilt the environment, regenerated all 10,833 features, ran the model on `cuda:0`, completed LoadGen with no warnings or errors, and the official evaluator printed exact match 83.68968779564806 and F1 90.87487229720105.
