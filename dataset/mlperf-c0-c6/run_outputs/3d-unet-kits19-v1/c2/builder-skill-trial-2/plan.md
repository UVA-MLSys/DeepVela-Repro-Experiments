# Full reproduction plan

## Workload and dependencies

Target `3d-unet` runs three serial stages: (1) create Python 3.10 environment, install cached packages, build supplied LoadGen, and patch only a disposable source copy for masked/missing snapshot facts; (2) create the work-local raw-data view and preprocess all 43 inference entries with four CPU workers; (3) check live A100 memory, run fp32 PyTorch Offline AccuracyOnly once, then run the supplied evaluator. Each stage writes a success marker consumed by the next.

## Measured preflight costs

- Clean environment plus local LoadGen build: approximately 2 minutes under current shared-filesystem contention; 8 host CPU cores available.
- One-case preprocessing (`case_00000`, shape `(1, 192, 384, 384)`): approximately 20 seconds using four workers. Conservative linear projection for 43 cases: 14.5 minutes.
- One-case accelerator inference: 4.33 seconds for 50 sliding-window sub-volumes. Conservative projection allowing larger cases and framework startup: 8 minutes.
- One-case postprocessing/evaluation: under 15 seconds. Conservative full projection: 4 minutes.
- GPU preflight availability: 70,567 MiB free of 81,920 MiB immediately before the measured run. The one-case inference completed on `cuda:0`.

## Resources and schedule

Run directly on the single local NVIDIA A100 80GB, never through Slurm. CPU environment/build and preprocessing use at most four workers because the host exposes eight cores. There is only one target, so no independent CPU target remains to overlap with its accelerator stage. Immediately before accelerator launch, the script rechecks free GPU memory and requires at least 20,000 MiB; it does not touch other sessions' processes.

## Expected time and reserve

Conservative end-to-end estimate is 29 minutes (2 environment + 14.5 preprocessing + 8 inference + 4 evaluation, rounded up). The wall-clock limit is 60 minutes. Hold 10 minutes in reserve for `report.json`, output/log verification, and clean-copy replay checks. The expected completion at 29 minutes is inside the 50-minute launch budget.

## Re-plan triggers

After initial full-run cases, compare observed preprocessing and inference rates with the conservative projections. If the projection exceeds the 50-minute work budget, preserve completed evidence and use the reserve to report measured status; do not change model, precision, split, scenario, preprocessing, or evaluator semantics.

## Re-plan 1: initial full preprocessing rate

Trigger: 14 of 43 cases were written after 57 seconds (approximately 4.1 seconds/case at four-worker throughput). New projection: roughly 3 minutes for preprocessing instead of 14.5 minutes, reducing expected total completion well below 20 minutes. No schedule or parallelism change is needed; retain the full 10-minute reporting reserve.
