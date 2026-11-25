# Reproduction plan

Updated: 2026-11-19 00:32 EST, after successful bounded end-to-end preflight.

## Workload and dependencies

One target, `3d-unet`, proceeds through four stages in delivered `reproduce.sh`: `env` -> `data` -> `run` -> `eval`. Environment copies the fresh source, applies compatibility-only repairs, creates Python 3.11 environment from the mandated offline uv cache, and builds repository LoadGen. Data materializes 43 case entries (mapping `case_00400` to the supplied `case_00185` data), preprocesses with the repository code, and verifies the published checksums. Run checks current GPU memory then executes the documented fp32 PyTorch Offline AccuracyOnly command. Eval invokes the repository evaluator and writes the required printed scores.

## Measured costs and resources

Bounded preflight used `case_00000`, with its manifest, checksum manifest, and LoadGen minimum query count reduced consistently:

- Environment: package installation 8.6 s plus LoadGen build 90 s on the second clean attempt; all packages resolved offline. Import/startup adds about 30 s under current filesystem contention.
- Data: 39.6 s wall, 2,562,012 KiB maximum RSS for one case. Full preprocessing uses 8 of the machine's 192 logical CPUs; system memory is about 504 GiB.
- GPU: launch check showed 75,722 MiB free of 81,920 MiB. One 192x384x384 case, 50 sliding subvolumes, took 6.83 s inference and 29.4 s including Python/model/LoadGen startup and teardown. Maximum host RSS was 3,055,760 KiB.
- Evaluation: 6.28 s and 2,253,140 KiB maximum RSS for one case.
- Storage: workspace filesystem had about 257 GiB free at preflight.

The bounded evaluator result (`mean=0.90515`, `kidney=0.9700`, `tumor=0.8403`) is sizing evidence only, not the benchmark result.

## Schedule and expected time

Run the delivered script once, unchanged, with no `--stage` option and `OUTPUT_DIR=./3d-unet/out`. It clears its own full-run work/output, then executes all stages sequentially. There is only one target and one GPU stage, so no GPU overlap is possible. Eight-way preprocessing is chosen because the single-case preflight peaked at 2.56 GiB and the host has ample memory; this trades modest CPU concurrency for isolation from other sessions. The GPU stage independently refuses to start below 20,000 MiB free.

Conservative projection:

- environment/build: 2-3 min;
- 43-case preprocessing/checksums: 4-7 min (8 workers, allowing shared-filesystem contention);
- 43-case GPU inference: 5-7 min (case volume sizes vary; linear extrapolation from the preflight plus startup);
- evaluation and artifact checks: 1-2 min.

Expected total is 12-18 min. Hold 10 min in reserve for `report.json`, artifact validation, and a deliverable syntax/clean-copy check. The 1-hour limit therefore has more than 30 min additional contingency.

## Re-plan triggers

After early full preprocessing and the first inference log entries, compare observed rates with the upper projection. Re-plan if projected completion exceeds 45 min elapsed, if free GPU memory falls below the script threshold, or if any stage fails. Do not alter processes belonging to other sessions.
