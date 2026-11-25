# Full-run plan

## Workload

Target `3d-unet` has four restartable stages: build a Python 3.11 environment and repository LoadGen; preprocess and checksum-verify all 43 manifest cases (with the documented `case_00185` duplicate represented as `case_00400`); run fp32 TorchScript Offline accuracy on the local A100; run `accuracy_kits.py` over the full LoadGen accuracy log.

## Measured costs and resources

- Environment preflight: about 2 minutes on CPU/shared filesystem, including source compilation and first imports.
- One-case data preflight: checksum-verified `case_00061`, producing shape `(1, 128, 448, 448)`, in under 1 minute with up to 16 CPU processes available.
- One-case accelerator preflight: 36 sliding windows in 2.74014 seconds. The live check showed 74,260 MiB free on the 80 GiB A100 immediately before launch.
- One-case evaluation completed in under 1 minute. Host resources observed: 128 CPU cores, about 1.8 TiB available RAM, and 4.4 TiB free workspace storage.

## Schedule and estimate

The full delivered script runs from `environment` to ensure the scored run starts with nothing built. Preprocessing uses at most 16 CPU workers. GPU inference runs alone after preprocessing and only after a fresh 16 GiB free-memory gate; evaluation then uses at most 16 CPU workers. There is one target, so no independent CPU target stage is ready to overlap with its GPU stage.

The measured 36-window inference time projects to a few minutes for 43 cases; varying case sizes and shared-filesystem startup dominate uncertainty. Budget 3 minutes for environment, 10 minutes for preprocessing, 10 minutes for inference, and 5 minutes for evaluation: 28 minutes expected. Hold 12 minutes in reserve for one repair, artifact validation, and `report.json`, leaving margin under the 60-minute limit.
