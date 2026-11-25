# Reproduction plan

Written after the successful bounded end-to-end preflight on 2026-11-23.

## Workload and dependencies

The single `bert-squad` target has three serialized stages owned by `reproduce.sh`: `env` verifies immutable hashes, builds the pinned offline environment and LoadGen, copies source, and instruments only the staged copy; `preprocess` transforms all 10,570 validation questions into 10,833 BERT features; `run` checks live GPU memory, runs every feature through the documented PyTorch Offline AccuracyOnly path, and invokes the checked-in evaluator once.

The canonical launch is `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from env`. It clears only script-owned artifacts and exercises the delivered wrapper from nothing built.

## Measured costs and resources

| Stage | Measurement |
|---|---|
| `env` | 41 seconds; 47 packages installed from the authorized offline cache and LoadGen 6.0.17 built locally (`bert-squad/build.log`) |
| `preprocess` | 58 seconds; 10,833 features and a 53,468,824-byte cache (`bert-squad/preprocess.log`) |
| bounded `run` | eight samples; about 1.662 seconds in LoadGen, approximately 208 ms/sample; model logged `cuda:0` and evaluator returned one valid result |
| GPU availability | 75,684 MiB free of 81,920 immediately before preflight (`bert-squad/gpu_memory.log`); wrapper requires at least 6,144 MiB |
| host | 8 CPU cores and about 1.9 TiB available host memory at planning time |

## Schedule, projection, reserve

Run `env`, `preprocess`, and `run` serially because there is one target and no independent ready CPU work to overlap with its GPU stage. The measured projection is `10,833 × 0.208 s ≈ 37.5 minutes`; allowing rebuild, preprocessing, model load, evaluator, and filesystem margin yields a 43-minute expectation, within the two-hour limit. The per-sample preflight rate drives the projection; the README QPS is irrelevant to AccuracyOnly.

Reserve 15 minutes for report transcription and artifact/replay checks. AccuracyOnly issues one Offline query and does not expose a safe intermediate fraction, so re-plan on completion or failure; no semantics-changing batching or parallelism adjustment is permitted.

## Live re-plan

At 12:30:49, after about 8.1 minutes of inference, the growing accuracy log contained 3,784 of 10,833 records (34.9%). The observed rate projects roughly 23.3 minutes for inference and about 26 minutes total through evaluation, earlier than the conservative 43-minute plan. Shared GPU utilization was 100%, but 70+ GiB remained free and the process was healthy. No schedule or semantic change is warranted; projected completion is approximately 12:48 with the full closeout reserve intact.

The canonical run and evaluator completed successfully at approximately 12:44. The evaluator printed exact match `83.68968779564806` and F1 `90.87487229720105`; closeout therefore proceeds with no further GPU work.
