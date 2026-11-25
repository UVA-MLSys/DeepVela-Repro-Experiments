# Reproduction plan

## Workload

Target `3d-unet` has three stages in `reproduce.sh`: (1) copy the fresh source, create a pinned Python 3.11 environment, and build bundled LoadGen; (2) stage the 43-case inference split, map `case_00400` to read-only `case_00185`, preprocess, and verify repository checksums; (3) check A100 memory, run PyTorch fp32 Offline AccuracyOnly inference, and invoke `accuracy_kits.py` once.

Dependencies are strictly environment -> preprocessing -> GPU run/evaluation. There is one target and one GPU stage, so no accelerator overlap is possible or useful.

## Preflight measurements

Measured 2026-11-19 with a consistent one-case subset (`case_00000`) of both the inference and checksum manifests:

- environment: package installation from shared uv cache took seconds; bundled LoadGen compilation brought the stage to about one minute. The first attempt with `/usr/bin/python3.11` failed because its development header was absent; the successful route uses the available user-local Python 3.11.16 with `Python.h`;
- preprocessing and checksum verification: 17.07 s wall, 2,562,284 KiB maximum RSS;
- run plus evaluator: 25.65 s wall, 3,112,052 KiB maximum RSS;
- core GPU inference: 3.87253 s for 50 128-cubed sliding windows;
- GPU state at run launch: NVIDIA A100-SXM4-80GB, 81,037 MiB free of 81,920 MiB; and
- bounded output: one unique prediction; `Accuracy: mean = 0.90515, kidney = 0.9700, tumor = 0.8403`.

The bounded LoadGen detail reports one expected validity error because the one-case subset is not the full prescribed accuracy set; its payload and benchmark-owned evaluator both completed successfully. It is preflight evidence only, not a reported benchmark result.

## Resources and schedule

The machine supplies the local A100 and host CPU/memory. Package resolution uses only `UV_CACHE_DIR=$CACHE_ROOT/uv`. Preprocessing uses eight CPU workers. The run begins only if at least 24,000 MiB GPU memory is free, records `nvidia-smi`, and uses one GPU process. Unrelated jobs remain untouched.

Run all stages serially via the no-`--stage` path from a clean script-owned build. Based on the preflight, reserve conservatively:

- up to 3 minutes for environment rebuild;
- up to 12 minutes for 43-case preprocessing/checksum verification (linear one-case extrapolation is 12.2 minutes, while eight-way parallelism should improve it);
- up to 15 minutes for inference and evaluation (case size varies, so this is deliberately above the 2.8-minute linear extrapolation of the measured case); and
- about 30 minutes total.

Hold back 10 minutes of the one-hour limit for exact result extraction, `report.json`, artifact checks, and clean-copy replay inspection. The 30-minute projection plus reserve fits the 60-minute limit.

## Re-plan triggers

After preprocessing, compare elapsed time with the 12-minute allowance. After the first full inference cases, use the per-case lines in `run.log` to project completion. If free GPU memory is below 24,000 MiB, the run fails without disturbing other jobs and is retried only after rechecking availability. No new work begins once the 10-minute reserve is reached.

## Full-run outcome

The no-stage full replay from `--from environment` completed successfully in 14:50.22 wall time, with 11,504,152 KiB maximum host RSS. All 43 preprocessing checksums passed. The GPU stage began with 67,487 MiB free and completed all 43 unique QSL indices. The evaluator printed `Accuracy: mean = 0.86168, kidney = 0.9347, tumor = 0.7887`; these exact strings are recorded in `report.json`. No re-plan was required because every stage stayed within its allowance.
