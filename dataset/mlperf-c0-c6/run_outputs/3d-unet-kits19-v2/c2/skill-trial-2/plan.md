# Full reproduction plan

## Workload

- Target `3d-unet`: build Python 3.11 plus supplied LoadGen; preprocess and verify all 43 inference cases; run the fp32 TorchScript model in Offline AccuracyOnly mode on the local A100; run the repository Dice evaluator.
- Keep raw data and model read-only. Represent documented duplicate `case_00400` by a target-owned symlink to `case_00185`.

## Measured costs and resources

- Prior bounded run on this same source/input route: environment plus one complete case took 2:01; LoadGen compilation about 46 seconds; peak host RSS 2.94 GB.
- The bounded case's 36 windows took 5.80 seconds (6.20 windows/s conservative rate). The full 43 cases contain an estimated 2,761 windows, projecting 7.4 minutes inference.
- Prior clean full run completed in 10:02 with peak host RSS 11.54 GB and all 43 evaluator results.
- Current host: 8 CPU cores and 5.3 TiB workspace free. Current A100 check found 68,959 MiB free; the wrapper rechecks immediately before GPU inference and requires 20,000 MiB.

## Schedule

1. Run the delivered `reproduce.sh --target 3d-unet --from environment` from nothing built.
2. Use eight CPU workers for preprocessing and evaluation. There is one target, so this session launches one GPU run only.
3. Start inference only after preprocessing/checksum verification; then evaluate the complete LoadGen accuracy log.
4. Audit required outputs, log prefix, full case coverage, and exact evaluator values; write `report.json` from evaluator stdout.

## Expected time and reserve

- Expected clean total is 10–20 minutes based on the prior 10:02 full run; shared-GPU contention is the main variance.
- Hold 10 minutes of the one-hour limit for reporting and artifact validation.

## Completion

- Clean full run completed successfully in 10:58.80 with 11.56 GB peak host RSS.
- All 43 inputs passed preprocessing checksum verification; LoadGen emitted 43 unique accuracy records and reported no warnings or errors.
- The repository evaluator covered 43 predictions and printed `Accuracy: mean = 0.86173, kidney = 0.9347, tumor = 0.7888`.
- Required outputs, stage logs, success markers, and the required device log prefix passed the final audit.
