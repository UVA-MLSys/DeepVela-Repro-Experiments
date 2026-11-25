# Full reproduction plan

Recorded 2026-11-19 after bounded end-to-end preflight.

## Workload and dependencies

Target `3d-unet` has three serial stages implemented by the delivered `reproduce.sh`: `env` builds the Python/LoadGen environment and work-local compatible source; `preprocess` consumes that environment plus all 43 fixed KiTS19 cases; `run` checks current GPU memory, runs fp32 PyTorch Offline AccuracyOnly inference, then invokes the benchmark evaluator. The final report consumes `accuracy.txt`.

## Measured costs and resources

- Preflight `env`: 157 s on this machine's 8 CPU cores; dependencies came solely from the shared offline uv cache.
- Preflight one-case `preprocess`: 18 s with four workers.
- Preflight one-case GPU run plus evaluation: 20 s; model inference used 6.71 s cold, with 67,781 MiB GPU memory free immediately before launch.
- Prior-artifact full replay of the same commit/input/parameters: about 5 minutes preprocessing and 5 minutes 15 seconds for inference plus evaluation, based on saved log timestamps. Its 43 inference samples each took approximately 1.8–8.0 s after warm-up.
- Available resources observed before preflight: one shared NVIDIA A100-SXM4-80GB, 8 CPU cores, about 1.8 TiB available host memory. The script requires at least 20,000 MiB free before starting GPU work.

## Schedule and projection

Run the delivered script unchanged from its first stage with a fresh full `OUTPUT_DIR`:

```bash
OUTPUT_DIR="$PWD/3d-unet/out" ./reproduce.sh --target 3d-unet --from env
```

There is only one target, so no other GPU work overlaps it. Environment build and preprocessing precede the GPU stage. Projected full elapsed time is 13–16 minutes: roughly 3 minutes environment, 5 minutes preprocessing, and 5–8 minutes inference/evaluation. This is driven by the measured preflight build and the prior full replay, not by a reduced-scope accuracy estimate alone.

## Reserve and success checks

Reserve at least 10 minutes before the one-hour limit for parsing the evaluator line, writing `report.json`, checking the 43 unique `qsl_idx` values, validating required files and logs, and testing clean-copy portability assumptions. If GPU free memory falls below 20,000 MiB at launch, the script exits without starting GPU work; a retry must use `--from run` after rechecking availability.

## Re-plan during full run

After preprocessing, the script observed 81,037 MiB free and launched successfully. During inference, two unrelated sessions also began using the shared GPU, bringing utilization to 100%; this run's process used 6,768 MiB and remained active. The trigger is external resource saturation, not a benchmark failure. Revised GPU-stage projection is 10–15 minutes rather than 5–8 minutes. No concurrency or semantic setting is changed: our one GPU run remains serialized, and the revised projection still leaves the 10-minute reporting reserve inside the one-hour limit.

## Actual completion

The clean full replay completed without repair: environment 46 s, preprocessing 253 s, and GPU run plus evaluation 408 s. LoadGen produced 43 records with unique `qsl_idx` values 0–42, and the evaluator printed `mean = 0.86168`, `kidney = 0.9347`, `tumor = 0.7887`. All requested files and stage success markers passed closure checks.
