# Full reproduction plan

## Workload

- Target `3d-unet`: environment build -> preprocess and checksum verification of all 43 manifest cases -> one fp32 CUDA Offline AccuracyOnly LoadGen run -> repository Dice evaluator.
- Inputs are the supplied raw KiTS19 directory and TorchScript model. `case_00400` resolves to a read-only symlink of the documented duplicate `case_00185`.

## Measured costs and resources

- Host: 8 CPU cores, about 1.8 TiB available RAM, one shared NVIDIA A100 80GB.
- Before bounded inference, 67,469 MiB GPU memory was free; the script requires at least 24,000 MiB immediately before each GPU run.
- Environment creation, pinned dependency installation, and local LoadGen build took about 90 seconds with a warm shared UV cache and occupies 5.2 GiB.
- One-case preprocessing plus published-checksum verification took about 20 seconds and produced 157 MiB.
- The bounded sample used 50 sliding-window subvolumes and took 7.16578 seconds of measured inference; model load and harness overhead brought the run stage to about 23 seconds.
- One-case postprocessing and Dice evaluation took about 6 seconds.

## Schedule and projection

- Run the delivered script from `--from environment` so the first full run clears and recreates all of its own built state.
- Use four preprocessing/evaluator workers on the 8-core host. There is only one target, so no independent CPU target stage is available to overlap with its GPU stage.
- Run exactly one GPU job, after all 43 cases pass preprocessing checksum verification.
- Linear upper estimates are 15 minutes preprocessing, 10 minutes inference/model overhead, and 5 minutes evaluation/output writing. With the approximately 2-minute rebuild, the full run projects to 32 minutes.
- Reserve 10 minutes before the one-hour limit for `report.json`, artifact checks, and a fresh-copy syntax/deliverable check.

## Re-plan triggers

- After preprocessing, compare actual duration and total preprocessed size with the projection.
- During inference, compare the first several reported samples and subvolume counts with the 7.16578 seconds per 50-subvolume measurement.
- If the projected completion enters the 10-minute reserve, start no new run and report only measured results.

## Re-plan 1: preprocessing completed

- Trigger: the full preprocessing stage completed and exposed all output shapes.
- Actual state: all 43 checksums passed; preprocessed data totals 8.1 GiB. Environment rebuild plus preprocessing completed in about four minutes, well below the 17-minute combined estimate.
- The shapes require 2,761 total sliding-window subvolumes. At the bounded rate of 7.16578 seconds per 50 subvolumes, projected full inference is 395.7 seconds (6.6 minutes).
- New projection: inference plus evaluation should finish in roughly 12 additional minutes, leaving more than the 10-minute reserve. GPU parallelism remains one because the reference SUT is sequential and the prompt permits one GPU run at a time.

## Completion

- All 43 LoadGen samples completed in the single GPU run; the run log contains 43 per-sample timing lines.
- The benchmark-owned evaluator completed over the full accuracy log and printed `Accuracy: mean = 0.86173, kidney = 0.9347, tumor = 0.7888`.
- All required outputs and stage logs are present in `3d-unet/out`; all four stage success markers are present.
