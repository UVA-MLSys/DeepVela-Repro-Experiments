# Reproduction plan

## Workload

One `3d-unet` target with four staged dependencies: cached Python environment → preprocessing of the 43-case KiTS19 inference manifest (including workspace-local `case_00400` alias) → one serial A100 LoadGen Offline accuracy run → repository evaluator and exact result extraction.

## Preflight measurements

- Environment: about 65 seconds; 31 cached packages plus the cached CPython 3.12 LoadGen wheel; CUDA-enabled Torch 2.5.1 reports CUDA 12.4 and sees the GPU.
- Preprocessing: 20 seconds for `case_00061` on four of the eight host CPUs; the resulting pickle exactly matched the repository MD5.
- GPU: 80,474 MiB free immediately before the test; the case used 36 sliding windows and inference took 3.88483 seconds. Model parameters were on `cuda:0`.
- Evaluation: under 2 seconds for one case, producing the exact required formatted line.

## Resources and schedule

This machine has 8 visible CPU cores, about 2 TiB RAM with about 1.8 TiB available at preflight, and one shared A100 80GB. Environment and preprocessing use CPU only. The GPU stage starts only after preprocessing succeeds, rechecks free memory, requires at least 24 GiB free, and is the only GPU run. Evaluation follows it. There are no other targets whose CPU stages can overlap the GPU stage.

The preflight case is near the small end of the compressed input-size distribution. Scaling preprocessing conservatively by case count and input size gives about 20–25 minutes with four workers. Scaling the 3.88-second/36-window GPU measurement to larger cases gives about 8–12 minutes for 43 cases. Including a clean environment rebuild and evaluation, the expected full run is 30–40 minutes. Eight minutes are reserved for report creation and artifact validation, so the projected run fits the one-hour limit.

## Launch and re-plan triggers

Launch `OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet --from environment`, which removes its own bounded-run state and executes the delivered wrapper from nothing built. Re-plan if preprocessing exceeds 25 minutes, if the first GPU cases project inference beyond 12 minutes, if GPU free memory falls below 24 GiB, or on any stage failure. No backend, model, precision, split, scenario, preprocessing, or evaluator substitution is permitted.

## Re-plan checkpoint

The full preprocessing stage finished in roughly three minutes and all 43 MD5 checks passed, substantially faster than the conservative 20–25 minute estimate. During the full GPU stage, 13 of 43 LoadGen accuracy responses were present after 1 minute 46 seconds, projecting about 6 minutes total. Trigger: first few percent of the full run completed. Change: none; continue the serial GPU run. New projection: full workflow comfortably below 15 minutes, retaining the eight-minute reporting reserve.
