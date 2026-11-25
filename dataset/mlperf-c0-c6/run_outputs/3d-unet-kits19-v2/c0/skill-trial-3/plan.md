# Full-run plan

## Workload

Target `3d-unet` has four restartable stages: create a Python 3.11 environment and build the repository LoadGen package; preprocess and checksum-verify all 43 manifest cases (with documented `case_00400` resolved from `case_00185`); run the fp32 TorchScript model in Offline accuracy mode on the local A100; run repository `accuracy_kits.py` over the full LoadGen accuracy log.

## Measured costs and resources

- Clean environment preflight: 53.52 seconds, 526 MiB peak host RSS.
- Two-case data preflight: 21.67 seconds and 4.4 GiB peak RSS for shapes `(1,128,448,448)` and `(1,384,384,384)`; both repository checksums passed.
- Two-case accelerator preflight: 23.25 seconds wall clock and 5.3 GiB peak host RSS. The two inference calls used 36 windows in 4.61 seconds and 125 windows in 11.94 seconds. The A100 had 67,487 MiB free immediately before launch and the logged model device was `cuda:0`.
- Two-case evaluator preflight: 12.01 seconds and 4.2 GiB peak RSS, ending in the repository evaluator's accuracy summary. LoadGen's bounded-run minimum-query error is expected for a two-case manifest and does not occur for the full 43-case manifest.
- Available resources observed before planning: one shared 80 GiB A100, 8 visible CPU cores, about 1.9 TiB available host RAM, and 5.1 TiB free workspace storage.

## Schedule and estimate

The full delivered script runs with `--from environment`, ensuring its own work tree starts with nothing built. Preprocessing and evaluation use at most 8 CPU workers. GPU inference runs alone after preprocessing and only after a fresh 16 GiB free-memory gate. There is one target, so no independent target's CPU stage is available to overlap with inference.

The measured two-case preprocessing scales to roughly 8 minutes conservatively; varying case sizes make count-only scaling approximate. The measured window rates and prior provenance-confirmed same-route artifacts put full inference within about 6 minutes. Budget 2 minutes for environment, 10 minutes for preprocessing, 7 minutes for inference, and 6 minutes for evaluation: 25 minutes expected. Hold 12 minutes for one repair, artifact checks, and `report.json`, leaving margin under the 60-minute limit.

## Re-plan at first full-run progress

The environment stage completed successfully. Full preprocessing produced 24 of 43 cases (3.9 GiB) after roughly two minutes of active processing despite a host load average near 38 from unrelated sessions. That rate projects preprocessing to about four minutes, below the 10-minute allocation. The schedule is unchanged and the overall projection remains at or below 25 minutes.

## Completion

The clean full invocation completed all four marked stages in 11:57.79, within the projection and reserve. The accuracy log contains 43 LoadGen records, and the repository evaluator printed `Accuracy: mean = 0.86168, kidney = 0.9347, tumor = 0.7887`.
