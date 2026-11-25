# 3D-UNet reproduction plan

- Workload: one target, `3d-unet`; environment build -> full 43-case preprocessing ->
  Offline fp32 GPU inference -> repository accuracy evaluator.
- Inputs: 43 cases from `meta/inference_cases.json`; the absent documented duplicate
  `case_00400` is represented by a workspace-local link to read-only `case_00185`.
- Resources: local NVIDIA A100 80GB; GPU work is serialized. The wrapper checks for
  at least 16 GiB free immediately before each GPU launch. CPU preprocessing and
  evaluation use four workers.
- Schedule: the environment and bounded preflight completed first. The full run invokes
  the delivered wrapper from its environment stage, forcing its own clean rebuild, then
  performs full preprocessing, the single GPU run, and evaluation.
- Reserve: retain at least five minutes for result extraction, artifact validation, and
  `report.json`.

## Preflight measurements and projection

- The bounded run used `case_00084`, the largest case by the prior snapshot's
  header-derived resampled voxel estimate. Its reduced case and checksum manifests both
  contained that one case, and checksum verification passed.
- The full preflight took 168.35 seconds, including a clean environment build. GPU
  inference processed 144 fp32 sliding-window subvolumes in 16.22 seconds. The GPU had
  64,855 MiB free immediately before launch.
- A conservative 43x inference projection is 11.6 minutes. The prior same-arm clean run
  completed all stages in 611.70 seconds, while its conservative total projection was
  under 30 minutes. Both estimates fit the one-hour limit with the five-minute reserve.

## Completion

- The delivered wrapper's clean-path full run completed in 738.91 seconds.
- All 43 preprocessed volumes passed the repository checksum manifest. The GPU stage
  launched with 81,037 MiB free and processed every LoadGen accuracy sample.
- The repository evaluator printed `Accuracy: mean = 0.86168, kidney = 0.9347,
  tumor = 0.7887`; the required artifacts and logs are in `3d-unet/out`.
