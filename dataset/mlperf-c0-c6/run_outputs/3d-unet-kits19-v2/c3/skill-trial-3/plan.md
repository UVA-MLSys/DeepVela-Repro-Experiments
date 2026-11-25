# Reproduction plan

## Workload

- Target `3d-unet`: build a target-local Python environment and LoadGen, preprocess and checksum all 43 KiTS19 manifest entries (mapping documented duplicate `case_00400` to `case_00185`), run the fp32 TorchScript PyTorch backend in Offline accuracy mode, then run `accuracy_kits.py` on the LoadGen accuracy log.
- Dependencies are `build -> preprocess -> run/evaluate`. The delivered `reproduce.sh` records success markers and supports both `--stage` and `--from`.

## Measured costs

- Preflight environment creation and cached package installation took about 34 seconds; LoadGen resolution/build/install took about 74 seconds and peaked at 0.53 GB RSS.
- One-case preprocessing and checksum verification took 15.21 seconds and peaked at 2.56 GB RSS with four workers.
- One-case inference processed 50 sub-volumes in 7.32 seconds; the complete GPU command took 18.81 seconds and peaked at 3.18 GB host RSS.
- One-case official evaluation took 5.06 seconds and peaked at 2.25 GB RSS.
- The prior same-source trial's NIfTI sizing counted 2,761 sub-volumes across all 43 entries. Scaling the measured GPU rate projects about 404 seconds of model inference.

## Resources and schedule

- Run directly on GPU 0 (NVIDIA A100-SXM4-80GB), never through Slurm. `reproduce.sh` records memory immediately before every GPU stage and requires at least 16 GiB free; preflight observed 68,959 MiB free.
- Use four CPU preprocessing workers. There is one target, so stages run in dependency order and only one GPU workload can run.
- Launch the first full benchmark through the unchanged delivered script with `OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet --from build`. This clears the wrapper's own build and downstream state and recreates everything from the fresh repository and fixed inputs.

## Expected time and reserve

- Expected full-run time is 10-20 minutes: roughly 2 minutes to build, 3-8 minutes to preprocess, 7-9 minutes for inference/setup, and under 1 minute to evaluate. The range allows for the observed shared-filesystem and GPU contention.
- Reserve at least 10 minutes before the one-hour limit for `report.json`, required-artifact checks, and final clean-copy/script inspection.

## Re-plan: full-run observations

- Trigger: the clean build and first full-run progress completed. Build took about 35 seconds, full preprocessing took 4:13 with all published checksums passing, and all 43 GPU samples completed in 3:54.
- Change: none. These measured times are faster than the 10-20 minute estimate and leave the reporting reserve intact.
- Final projection: evaluator and validation complete well before the one-hour limit.
