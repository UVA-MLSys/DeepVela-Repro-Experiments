# Full reproduction plan

Updated: 2026-11-19 00:49 America/New_York

## Workload and dependencies

Target `3d-unet` has three serial stages in `reproduce.sh`: (1) create a pinned Python 3.11 environment and compile the supplied LoadGen; (2) construct a work-local 43-case raw-data view, preprocess it, and verify every published checksum; (3) check current A100 memory, run fp32 PyTorch Offline accuracy inference, and invoke `accuracy_kits.py`. The GPU stage consumes the data-stage output. There is one target and therefore no independent CPU target to overlap with its GPU work.

## Preflight measurements

- Environment: about 2 minutes wall time in the clean preflight, including a 35-second LoadGen compile and cold imports. Peak measured compiler RSS was 0.55 GB. Evidence: `3d-unet/preflight/build.log` and marker timestamps.
- Data: one representative inference case completed preprocessing and checksum verification in 24.11 seconds with 2.56 GB maximum RSS. Four preprocessing workers are configured; the one-case run exercised only one busy worker. Evidence: `3d-unet/preflight/preprocess.log`.
- GPU: free memory was 81,037 MiB immediately before the successful bounded run. The case had 50 subvolumes; total run time was 11.99 seconds and reported inference time was 3.026 seconds. Peak host RSS was 3.06 GB. Evidence: `3d-unet/preflight/run.log`.
- Evaluation: one case completed in 6.02 seconds with 2.25 GB maximum RSS and printed `Accuracy: mean = 0.90521, kidney = 0.9700, tumor = 0.8404`. Its LoadGen accuracy log was 55 MB. Evidence: `3d-unet/preflight/evaluator.log` and `3d-unet/preflight/out/accuracy.txt`.

## Available resources and schedule

This direct session exposes 8 CPU cores, roughly 2 TiB host RAM (about 1.8 TiB available at planning time), and one shared NVIDIA A100-SXM4-80GB. The script uses four CPU preprocessing/evaluator workers, leaving capacity for the system and other sessions. It checks GPU free memory immediately before inference and requires 20,000 MiB free, waiting in 15-second intervals if necessary. Only this target's run uses the GPU, so GPU work is serialized by construction.

The full run starts from `--from environment`, which clears only the wrapper's own normal work and required outputs, then runs environment, data, GPU inference, and evaluation in order. No preflight artifact is reused.

## Projection and reserve

Linear four-worker scaling of the 24.11-second case gives roughly 4.3 minutes for 43 cases; allow 12 minutes for case-size variation and shared-filesystem contention. Scaling the measured GPU inference rate gives 2.2 minutes for cases shaped like the preflight case; allow 15 minutes for larger volumes, cold starts, and contention. Allow 8 minutes for full evaluation and its approximately 2.4 GB accuracy log, plus 3 minutes for the environment. Projected benchmark completion is therefore **38 minutes**.

Hold the final **10 minutes** of the one-hour limit for parsing the evaluator output into `report.json`, checking the required files/log prefix, and clean-copy replay validation. The planned 38-minute run plus 10-minute reserve fits the limit with 12 minutes of contingency. Re-plan after preprocessing and after the first few full-run cases if measured rates project beyond the reserve boundary.

## Rate update at full-run launch

At 00:54, preprocessing completed all 43 cases and checksum verification in 2:46, substantially below its 12-minute allowance. About 2.5 minutes into inference, the growing accuracy log held roughly 1.6 GB of an estimated 2.64 GB hex payload, projecting GPU completion near 5 minutes and below its 15-minute allowance. Trigger: the first substantial fraction of the full GPU workload completed. Change: no schedule or parallelism change; retain the original 10-minute closing reserve.

## Completion and clean replay

The primary full run completed inference in 3:20 and evaluation in 1:45, reporting mean 0.86173, kidney 0.9347, and tumor 0.7888. A separate `clean-replay/` directory containing only a copied `reproduce.sh` beside a fresh `repo/` copy then ran the no-stage command successfully. It recreated every required output and all success markers; its evaluator stdout was byte-for-byte identical to the primary output. Shared-GPU contention made replay inference slower, but the pre-run memory check still observed 70,237 MiB free and the replay completed within the reserve.
