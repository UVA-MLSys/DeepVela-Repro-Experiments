# 3D-UNet KiTS19 reproduction plan

## Workload

Target `3d-unet` runs four ordered stages: environment/setup, preprocessing of all 43 cases in
`meta/inference_cases.json`, fp32 PyTorch Offline accuracy inference on the local A100, and the
repository Dice evaluator. Setup links the read-only cache inputs into target-local work storage;
`case_00400` links to the documented duplicate `case_00185`.

## Measured costs

The current bounded end-to-end preflight covered `case_00000`, `case_00185`, and duplicate
`case_00400`. Package setup completed from the required uv cache. Preprocessing took 65.41 seconds
and 2,559,940 KiB peak host RSS, with every reduced-manifest checksum matching. The GPU stage took
31.27 seconds for 300 sliding-window subvolumes and 5,986,460 KiB peak host RSS. It began with
81,037 MiB GPU memory free and logged the model on `cuda:0`. Evaluation completed successfully.

The provenance-verified prior clean full run of the same source, manifest, model, runner semantics,
and machine route took 3:36.80 for inference. Its artifacts and all 238 provenance hashes match.

## Resources and schedule

The host exposes one shared NVIDIA A100 80GB. Preprocessing and evaluation use eight CPU processes.
GPU work is a single invocation and begins only after a fresh `nvidia-smi` check reports at least
16,000 MiB free. There is one target, so no second GPU run is queued and there is no independent
CPU target to overlap.

The current bounded rates project roughly 16 minutes for preprocessing by case count, while the
prior full measurement bounds GPU inference near 4 minutes; allowing setup, nonlinear case sizes,
and evaluation gives a 30-minute full-run projection. Ten minutes remain reserved for artifact
validation and `report.json`, within the one-hour limit.

## Established repairs

The target-local benchmark copy applies three preflight-validated compatibility repairs without
changing backend, precision, data, model, scenario, preprocessing, or metric: SciPy's relocated
Gaussian API, the evaluator's missing aggregate row, and the required model-parameter device log.

## Full-run outcome

No schedule change was needed. Full preprocessing completed in 1:47.64, substantially faster than
the projection, and verified every checksum. Inference completed in 4:11.49 and evaluation finished
within the reserve. The evaluator printed `mean = 0.86173`, `kidney = 0.9347`, and `tumor = 0.7888`.
