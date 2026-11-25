# Full-run plan

## Workload

Target `3d-unet` has three serial stages: (1) create the Python 3.11 environment, copy and repair the app locally, and compile the included LoadGen; (2) preprocess and checksum all 43 KiTS19 inference cases on CPU; (3) check current GPU memory, run fp32 TorchScript Offline accuracy on the local A100, and invoke the repository evaluator. The run stage depends on the complete preprocessing stage.

## Preflight measurements

- Environment: dependency resolution took 0.11 s, cached installation 4.78 s, and the LoadGen wheel build 44.27 s. CUDA import/device initialization added roughly one minute on the shared host. Budget: 3 minutes from a clean target directory.
- Preprocessing: one full CT case (`case_00000`, output shape `1 x 192 x 384 x 384`) completed preprocessing and checksum verification in approximately 105 seconds including Python imports and shared-filesystem I/O. Four worker processes are used. Conservative full-set budget: 20 minutes; this deliberately does not assume perfect four-way scaling.
- Accelerator: immediately before the repaired preflight, 74,572 MiB of 81,920 MiB was free. The selected case used 50 sliding windows in 5.156 s. Model load plus LoadGen and evaluator made the complete repaired run stage 18 s. Conservative full-set budget: 15 minutes for variable volume sizes plus postprocessing/evaluation.
- Host resources: 8 CPU cores, about 1.8 TiB available RAM, one shared A100 80GB; the wrapper requires at least 12,000 MiB free immediately before launch.

## Schedule

There is one target, so environment and preprocessing run first and the sole GPU run starts immediately after they succeed. GPU work is serialized by construction. No unrelated process is modified. With no second target ready for CPU preparation, there is no useful independent stage to overlap with GPU inference.

## Expected time and reserve

Expected full replay is 38 minutes (3 environment + 20 preprocessing + 15 inference/evaluation). This fits the one-hour limit with a 12-minute reserve for `report.json`, artifact checks, and one bounded repair; an additional 10 minutes covers shared-filesystem variance. If the first inference cases project the run beyond the reserve, the measured partial outputs will be recorded and no new repair run will start.

## Re-plan record

- Initial plan written after the successful one-case end-to-end preflight. No re-plan yet.
- Full replay completed without a re-plan. Full preprocessing plus checksum verification took about 4 minutes rather than the budgeted 20; all 43 accelerator queries then completed in about 3.5 minutes, leaving the reserve intact.
