# Reproduction plan

## Workload

One target, `3d-unet`, with three dependent stages: setup/build; CPU preprocessing of the fixed 43-entry inference manifest; then one local A100 Offline accuracy run followed by `accuracy_kits.py`. The final evaluator writes `mean`, `kidney`, and `tumor`.

## Measured costs

- Bounded setup: 107 s wall time; CPU plus package-cache I/O; no GPU allocation.
- Bounded preprocessing: 19 s for one case. A provenance-verified prior run of the identical wrapper and source took 234 s for all 43 entries.
- Bounded accelerator/evaluator stage: 16 s for one case, including 5.60 s measured inference. The prior identical 43-case run recorded 255.71 s total sample inference.
- GPU preflight: NVIDIA A100-SXM4-80GB, 70,245 MiB free immediately before inference; the wrapper requires at least 20,000 MiB.

## Resources and schedule

Run the delivered script directly on this machine. The full run starts from setup with `./reproduce.sh --target 3d-unet --from setup`, clearing and rebuilding its own work tree and outputs. Setup and preprocessing are CPU-only and sequential dependencies for this single target. Immediately before the accelerator stage the script queries free GPU memory; only that stage uses the GPU. No Slurm job is submitted and no concurrent GPU run is launched by this session.

## Expected time and reserve

The measured/prior costs project roughly 2 minutes setup + 4 minutes preprocessing + 5 minutes inference + evaluator overhead, under 15 minutes total. Reserve 10 minutes for output validation, `report.json`, and deliverable replay checks, well inside the one-hour limit. The prior full preprocessing and inference measurements drive this projection because the one-case startup costs do not scale linearly.

## Status

- Contract inspection: complete.
- One-case end-to-end bounded preflight: complete.
- Full clean run: complete (43/43 unique LoadGen accuracy results; evaluator exited 0).
- Required artifact validation and report: complete. All requested outputs are nonempty; 43 unique QSL indices, required device log line, stage markers, and exact report/evaluator agreement were verified.
