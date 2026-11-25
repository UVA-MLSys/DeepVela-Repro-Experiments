# Full reproduction plan

Written after the bounded preflight on 2026-11-19.

## Workload and dependencies

The single target is `3d-unet`, split into three serial stages implemented by the delivered
`reproduce.sh`: `env` builds the Python environment and included LoadGen; `prepare` creates
target-local raw-case links and preprocesses/verifies all 43 inference cases; `run` checks GPU
memory, performs the one fp32 PyTorch Offline accuracy run, and invokes `accuracy_kits.py` on
the resulting LoadGen accuracy log. Each stage leaves a success marker and the next stage reads
its files.

## Measured costs and resources

- Environment: 2m47s elapsed in the clean uv environment build. It used CPU, filesystem, and
  the mandated shared package cache; no GPU.
- One-case preparation: 18s elapsed with four preprocessing processes, including the official
  checksum verification. The host exposes 8 CPU cores and approximately 1.8 TiB available RAM.
- One-case inference: 5.003s for 50 sliding-window subvolumes on the A100. Immediately before
  launch, 74,260 MiB of 81,920 MiB GPU memory was free.
- One-case postprocessing/evaluation: approximately 6s with four processes.
- The bounded run covered the sole input kind and all pipeline stages. It reduced the case
  manifest and checksum manifest together, and produced all required artifact types.

## Schedule and estimate

Run `./reproduce.sh --target 3d-unet --from env` so the first full run clears and rebuilds every
script-owned stage output. There is only one GPU stage, so no GPU overlap is possible or allowed.
CPU build and preprocessing precede it because this target consumes both outputs; no other target
has ready CPU work to overlap.

Linear extrapolation gives about 13 minutes for preparation (`43 * 18s / 1 case`, deliberately
conservative because four-process preprocessing amortizes startup), 3.6 minutes for inference
(`43 * 5.003s`), and 4.3 minutes for evaluation (`43 * 6s`, also conservative). Adding the
measured 2m47s environment build and 5 minutes contingency gives a projected full replay of under
29 minutes.

## Reserve and re-plan triggers

Reserve the final 10 minutes of the one-hour limit for artifact checks and `report.json`. After
the first several cases, compare observed GPU seconds/case with the 5.003s preflight rate. If the
projection exceeds 50 minutes total, record the new rate here and protect the reserve. Any failure
returns to the affected stage via `--from`; the script clears its own downstream outputs and
markers.

## Actual full replay

The clean replay ran from 01:00:41 to 01:14:21 (13m40s). Environment setup took 2m20s with
cache reuse, all 43 cases preprocessed and passed checksums in 4m04s, contended GPU inference
took 5m12s, and evaluation took 2m02s. No re-plan was needed. The GPU check immediately before
the run reported 68,497 MiB free. The evaluator printed `mean = 0.86168`, `kidney = 0.9347`,
and `tumor = 0.7887`; required artifacts and report validation finished inside the reserve.
