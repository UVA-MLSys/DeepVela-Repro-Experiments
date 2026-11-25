# Reproduction plan

Updated: 2026-11-19 13:39 EST, after successful bounded preflight.

## Workload and dependencies

Target `3d-unet` has three serial stages implemented by the delivered `reproduce.sh`: `setup` (owned source copy, isolated environment, LoadGen build) → `preprocess` (43 fixed KiTS19 cases, CPU) → `run` (one local A100 GPU accuracy pass, then CPU evaluator). The run stage writes the three scored outputs and evidence logs.

## Measured costs

- Clean preflight setup completed at 13:37:55 after roughly 3.5 minutes, including cached wheel installation and a 1:14 LoadGen build (`3d-unet/preflight/out/build.log`).
- One-case preprocessing: 15.05 s wall, 2.56 GB peak RSS, about one CPU core despite a four-process cap (`preprocess.log`).
- One-case CUDA harness: 15.01 s wall, 3.11 GB peak host RSS; the measured model portion was 7.98 s (`run.log`).
- GPU preflight observed 73.3 GB free by PyTorch, above the wrapper's 16 GiB guard (`gpu-memory-before.log`).
- Evaluator completed successfully and printed all metrics (`accuracy.txt`).
- Provenance-backed prior clean full run of the same source/model route measured 2:44 preprocessing and 3:39 harness execution with 11.5 GB peak host RSS (`prior-artifacts/3d-unet/out/{preprocess.log,run.log}`); this is more reliable than linear one-case scaling because case volumes vary.

## Resources and schedule

This machine exposes 8 CPU cores, about 2 TiB RAM, and one shared A100 80GB. The full run will use four preprocessing/evaluator workers and one GPU run. There is only one target, so no other ready CPU-only target can overlap its GPU stage. Immediately before inference the wrapper rechecks free GPU memory and refuses to start below 16 GiB.

Launch the unchanged delivered wrapper from a clean first stage:

```bash
OUTPUT_DIR="$PWD/3d-unet/out" ./reproduce.sh --target 3d-unet --from setup
```

## Expected time and reserve

Expected full duration is 12–15 minutes: about 4 minutes setup, 3 minutes preprocessing, 4 minutes inference/evaluation, plus filesystem variance. The governing measurements are the prior full-stage times, while current preflight confirms compatibility and current resource fit. Reserve 10 minutes for parsing `accuracy.txt`, writing `report.json`, and validating the clean-copy deliverable. This leaves at least 35 minutes contingency within the one-hour wall-clock limit.
