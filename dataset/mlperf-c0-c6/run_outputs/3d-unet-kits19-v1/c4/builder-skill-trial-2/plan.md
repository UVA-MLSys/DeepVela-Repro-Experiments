# Reproduction plan

Updated: 2026-11-19 01:21 America/New_York, after bounded preflight and before the canonical full run.

## Workload and dependencies

One target, `3d-unet`, runs the delivered `reproduce.sh` stages in order: `environment` -> `data` -> `preflight` -> `run` -> `evaluate`. Environment builds a Python 3.11 uv environment and the supplied in-tree LoadGen. Data creates a read-only 43-case view, preprocesses it, and verifies published checksums. Preflight checks current GPU memory, model loading/inference, and the evaluator formula. Run checks GPU memory again and executes the required Offline AccuracyOnly LoadGen command. Evaluate decodes all 43 outputs and reports mean/kidney/tumor Dice.

## Measured costs and resources

- Environment: 213 s wall time; CPU/filesystem only (`3d-unet/out/environment.log`).
- Data: 165 s using four CPU workers; all 43 checksums passed (`3d-unet/out/data.log`).
- Preflight: 59 s including Python/CUDA startup; launch snapshot showed 74,260 MiB free of 81,920 MiB (`3d-unet/out/preflight.log`).
- Steady-state model: 0.038780 s per 128^3 patch over five timed patches after warmup (`3d-unet/out/rate_preflight.log`). The full manifest contains 2,761 patches, hence approximately 107 s of raw network time.
- Available accelerator: one shared NVIDIA A100-SXM4-80GB, used by this reproduction for only one process at a time. GPU availability is rechecked immediately before each GPU stage. CPU preprocessing uses four workers to limit contention.

## Schedule and projection

Run `OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet --from environment`. This explicitly clears the script's own built state and outputs, then tests the delivered script from nothing built. Stages are serial because there is only one target; no unrelated CPU task remains to overlap with the GPU stage. Based on measurements: 3.6 min environment + 2.8 min data + 1 min preflight + 3–7 min LoadGen overhead/inference + 1–3 min evaluation gives 12–17 min total. The patch rate drives the run estimate; measured build/preprocessing times drive their estimates.

Reserve 10 minutes of the one-hour limit for parsing `accuracy.txt`, writing `report.json`, checking required artifacts/log prefixes, and a fresh-copy smoke audit. If the first cases project beyond 17 minutes because of shared-GPU contention, retain the single-process run but update this plan and projection; no backend, precision, scope, or evaluator change is permitted.
