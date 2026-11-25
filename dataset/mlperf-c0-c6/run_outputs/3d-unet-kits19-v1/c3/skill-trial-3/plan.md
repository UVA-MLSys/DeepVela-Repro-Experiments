# Reproduction plan

## Workload

- Target `3d-unet`: build a target-local Python environment and LoadGen, preprocess and checksum all 43 KiTS19 manifest entries (including the documented `case_00400` duplicate), run the fp32 TorchScript PyTorch backend in Offline accuracy mode, then run `accuracy_kits.py` over the LoadGen accuracy log.
- Dependencies are strictly `build -> preprocess -> run/evaluate`. The delivered `reproduce.sh` records success markers and supports `--stage` and `--from`.

## Measured costs

- Cached package installation: 4-5 seconds; LoadGen compilation: 30-33 seconds, 0.53 GB peak RSS.
- One-case preprocessing plus checksum verification: 15-21 seconds, 2.56 GB peak RSS with four workers.
- One-case GPU inference: 50 sub-volumes in 3.70-8.60 seconds; complete run-stage wall time 12-43 seconds under changing GPU contention, 3.16 GB peak host RSS.
- One-case evaluator: 5.6 seconds, 2.25 GB peak RSS.
- NIfTI header sizing for all 43 entries gives 2,761 sub-volumes, with a maximum of 144 for one case. Scaling the observed GPU rate projects 204-475 seconds of model inference.

## Resources and schedule

- Run directly on GPU 0 (NVIDIA A100-SXM4-80GB), never through Slurm. `reproduce.sh` records free memory immediately before its GPU stage and requires at least 16 GiB free. Preflight checks observed 62,969-81,037 MiB free.
- Use four CPU preprocessing workers. There is one target, so no second GPU workload exists to overlap; environment setup, preprocessing, inference, and evaluation run in dependency order.
- Launch the first full benchmark as the unchanged delivered script with `OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet --from build`, which clears its own build and downstream state.

## Expected time and reserve

- Expected full-run time: about 5-15 minutes (roughly 1 minute build, 3-8 minutes preprocessing, 3.4-7.9 minutes inference, and under 1 minute evaluation), plus shared-filesystem cleanup variability.
- Reserve: 10 minutes before the one-hour limit for `report.json`, artifact checks, and a final clean-copy/script inspection.

## Re-plan: first full-run stage complete

- Trigger: the clean full-run build completed; LoadGen compilation was 49 seconds rather than the preflight's 30-33 seconds due to shared-filesystem contention.
- Change: none. The observed 16-second increase does not move the 5-15 minute projection near the one-hour limit or the 10-minute reserve.
- New projection: still under 15 minutes total, with preprocessing followed by the single serialized GPU run and evaluation.
