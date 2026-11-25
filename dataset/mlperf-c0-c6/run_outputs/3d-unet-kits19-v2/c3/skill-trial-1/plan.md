# Full reproduction plan

## Workload

- Target `3d-unet`: clean environment build, preprocessing and checksum verification of all 43 manifest cases, one fp32 CUDA Offline AccuracyOnly LoadGen run, then the repository Dice evaluator.
- Inputs are the supplied read-only KiTS19 directory and TorchScript model. `case_00400` resolves to the documented duplicate `case_00185` through a read-only symlink.

## Measured costs and resources

- Host: 8 CPU cores, about 1.8 TiB available RAM, one shared NVIDIA A100 80GB.
- The clean cached environment build took about 40 seconds and produced PyTorch 2.7.1+cu126 and LoadGen 6.0.17.
- One-case preprocessing plus published-checksum verification took under 30 seconds.
- Immediately before bounded inference, 67,781 MiB GPU memory was free; the script requires at least 24,000 MiB immediately before every GPU run.
- The bounded case exercised the workload's image/label preprocessing, checksum verification, fp32 TorchScript backend, Offline AccuracyOnly LoadGen path, postprocessing, and Dice evaluator. Its 50 sliding-window subvolumes took 7.56849 seconds.
- The successful read-only prior trial records 2,761 subvolumes across the same 43-case manifest, implying about 418 seconds of inference at the measured bounded rate.

## Schedule and projection

- Run the delivered script from `--from environment`, which clears and recreates its own environment and all downstream bounded artifacts.
- Use four preprocessing/evaluator workers on the 8-core host. There is only one target, so there is no separate target's CPU work to overlap with its GPU stage.
- Run one GPU job after all 43 published preprocessing checksums pass.
- Budget 2 minutes for the rebuild, 15 minutes for preprocessing, 10 minutes for inference and harness overhead, and 5 minutes for evaluation: 32 minutes total.
- Reserve 10 minutes before the one-hour limit for `report.json`, output validation, and a fresh-copy deliverable check.

## Re-plan triggers

- After full preprocessing, compare elapsed time and preprocessed size with the projection.
- During inference, compare sample rate with 7.56849 seconds per 50 subvolumes.
- If projected completion enters the 10-minute reserve, start no new work and report only measured results.

## Re-plan 1: full preprocessing completed

- Trigger: all full preprocessing outputs and their verification results became available.
- All 43 manifest cases passed the published checksums; environment rebuild plus preprocessing completed well below the combined 17-minute allowance.
- The GPU stage started with 68,923 MiB free, above the 24,000 MiB guard.
- The remaining projection is about 10–15 minutes for inference, evaluation, and artifact checks, so GPU parallelism remains one and the 10-minute closing reserve remains intact.

## Completion

- All 43 manifest cases passed preprocessing checksum verification and completed in one fp32 CUDA Offline AccuracyOnly run.
- The benchmark-owned evaluator printed `Accuracy: mean = 0.86173, kidney = 0.9347, tumor = 0.7888`.
- The required LoadGen accuracy/detail logs, evaluator stdout, and all stage logs were written under `3d-unet/out`; all stage success markers were written.
