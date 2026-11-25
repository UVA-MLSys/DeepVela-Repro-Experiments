# Reproduction plan

## Workload

Target `3d-unet` runs three serial stages through the delivered `reproduce.sh`: build a pinned Python 3.11 environment and bundled LoadGen; stage and preprocess the 43-case inference split (mapping missing `case_00400` to read-only `case_00185`); then check GPU memory, run fp32 PyTorch Offline AccuracyOnly inference, and invoke `accuracy_kits.py`. Dependencies are environment -> preprocess -> GPU run/evaluation. There is one GPU target, so GPU work cannot overlap.

## Measured preflight costs

The 2026-11-19 bounded run reduced both `inference_cases.json` and `checksum_inference.json` to `case_00000` and completed in 2:00.72 wall time with 3,133,544 KiB peak host RSS. Preprocessing took about 17 seconds and verified the published checksum. GPU launch observed 60,713 MiB free on the A100; 50 sliding windows took 7.84382 seconds. The evaluator printed `Accuracy: mean = 0.90515, kidney = 0.9700, tumor = 0.8403`. The reduced scope causes the expected LoadGen full-set validity warning but does not bypass any manifest or evaluator check.

The permitted prior artifact used byte-identical launcher, evaluator, preprocessing, case manifest, and model inputs. Its 43-case clean run completed in 14:50.22 with 11,504,152 KiB peak RSS and is used as the strongest full-size estimate.

## Resources and schedule

Run all stages serially with `./reproduce.sh --target 3d-unet --from environment`. Preprocessing uses eight CPU workers. The GPU stage starts only after a logged `nvidia-smi` check confirms at least 24,000 MiB free, and uses one local GPU process. Package resolution uses only the fixed shared uv cache with `UV_EXCLUDE_NEWER=2026-11-10`.

Allow 3 minutes for environment rebuild, 12 minutes for preprocessing, and 10 minutes for inference plus evaluation: 25 minutes total. Hold 10 minutes of the one-hour budget for result extraction, `report.json`, artifact validation, and portability checks. The prior full-size measurement and current preflight both fit this projection.

## Re-plan triggers

After preprocessing, compare elapsed time with the 12-minute allowance. During inference, project completion from the per-case timing lines. If fewer than 24,000 MiB are free, the script exits without disturbing unrelated jobs; retry only after another memory check. At the 10-minute reserve, start no new work and report only measured results.

## Full-run outcome

The clean no-stage replay from `--from environment` completed successfully in 13:40.10 wall time with 11,522,124 KiB peak host RSS. All 43 preprocessing checksums passed. The GPU stage began with 81,037 MiB free, processed exactly 43 samples, and the evaluator printed `Accuracy: mean = 0.86168, kidney = 0.9347, tumor = 0.7887`. All required outputs and stage markers exist, so no re-plan was required.
