# Reproduction plan

Recorded 2026-11-18 after the successful bounded preflight.

## Workload

One target, `3d-unet`, with three staged dependencies:

1. `setup`: fresh target-local repository copy and Python 3.11 environment; installs only from the required uv cache, installs cached LoadGen 6.0.17, links the supplied model/raw inputs, and adds the required device diagnostic to the local source copy.
2. `preprocess`: preprocesses and verifies all 43 inference-list entries, including target-local `case_00400 -> case_00185` duplication.
3. `run`: checks current GPU free memory, runs fp32 PyTorch Offline accuracy once on the local A100, then invokes the benchmark evaluator and validates required outputs.

The full run is `OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet --from setup`, which recreates its own work tree before executing every stage.

## Measured costs

Bounded end-to-end preflight used `case_00000`, with both the case manifest and checksum manifest consistently reduced:

| Stage | Measured wall time/resources |
|---|---|
| Setup | 27 s; CPU/filesystem only; cached packages; final environment is PyTorch 2.5.1+cu124, NumPy 1.26.4, SciPy 1.10.1, pandas 1.5.3, nibabel 5.3.2, LoadGen 6.0.17 |
| Preprocess + checksum | 19 s for one relatively large `192x384x384` result; 4 CPU processes configured |
| GPU inference | 7.38 s, 50 sliding-window subvolumes; A100 had 69,271 MiB free immediately before launch |
| Postprocess/evaluator | about 8 s for one case; evaluator completed and wrote all required formats |

Evidence: `3d-unet/preflight/out/{build,preprocess,gpu_memory,run,evaluator}.log`, `accuracy.txt`, and LoadGen logs.

## Resources and schedule

- Machine: 8 visible CPU cores, about 2 TiB RAM, one shared NVIDIA A100 80GB.
- GPU work is serialized within this reproduction. The run stage refuses launch below 20,000 MiB free and records `nvidia-smi` immediately before inference.
- With one target, there is no independent second CPU target to overlap. Setup and preprocessing precede GPU inference because the latter reads their artifacts.
- Preprocessing uses four workers, leaving CPU headroom for other machine users.

## Expected time and reserve

- Setup: under 1 minute based on 27 s preflight.
- Full preprocessing: conservatively 4–8 minutes. The simple serial extrapolation is 13.6 minutes, but four workers process the 43 cases concurrently; shared-filesystem variability is the main uncertainty.
- Full inference: approximately 5.3 minutes if cases resemble the preflight's 7.38 s rate, conservatively 5–10 minutes for shape variation and GPU contention.
- Full evaluation: conservatively 2–4 minutes.
- Total projection: 12–23 minutes, below the one-hour limit.
- Reserve: at least 10 minutes for `report.json`, artifact checks, and one evidence-backed repair if necessary.

The measured GPU rate drives the single serialized run; the preprocessing measurement drives four CPU workers; the observed shared-GPU occupancy drives the memory guard and immediate pre-run check.
