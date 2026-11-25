# 3D-UNet reproduction plan

- Workload: one target, `3d-unet`; environment build -> full 43-case preprocessing ->
  Offline fp32 GPU inference -> repository accuracy evaluator.
- Inputs: 43 cases from `meta/inference_cases.json`; the absent documented duplicate
  `case_00400` is represented by a workspace-local link to read-only `case_00185`.
- Resources: local NVIDIA A100 80GB; GPU stages are serialized. The wrapper checks for
  at least 16 GiB free immediately before inference. CPU preprocessing uses four workers.
- Schedule: build and bounded preflight first, then invoke the delivered wrapper from its
  environment stage for the measured full run. No other target exists to overlap.
- Reserve: retain at least five minutes for evaluator output validation and `report.json`.

## Preflight measurements and projection

- The bounded run used `case_00084`, the largest case by header-derived resampled
  voxel estimate (59.6 million voxels). Its reduced case and checksum manifests both
  contained that one case; checksum verification passed.
- The full preflight took 81.38 seconds. Preprocessing and verification took about
  25 seconds. GPU inference performed 144 fp32 sliding-window subvolumes in 13.453
  seconds. The GPU had 68,559 MiB free immediately before launch.
- Conservative 43x projections are about 18 minutes for preprocessing and 10 minutes
  for inference. Environment construction measured under one minute; evaluation and
  filesystem overhead retain more than the five-minute reserve under the one-hour limit.
- Full-run schedule: invoke the delivered wrapper with `--from environment`, forcing
  its own clean rebuild, then full preprocessing, the single GPU run, and evaluation.

## Completion

- The repaired clean-path invocation completed in 611.70 seconds total.
- All 43 preprocessed volumes passed the repository checksum manifest. The GPU stage
  launched with 81,037 MiB free and processed every LoadGen accuracy sample.
- The repository evaluator printed `Accuracy: mean = 0.86168, kidney = 0.9347,
  tumor = 0.7887`; the required artifacts and logs are in `3d-unet/out`.
