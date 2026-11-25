# Full reproduction plan

## Workload

Target `3d-unet` has four stages: create a native Python environment; preprocess and checksum-verify all 43 manifest entries (42 distinct raw cases plus the documented `case_00400` duplicate); run fp32 TorchScript PyTorch inference in Offline accuracy mode on the local A100; then run the repository's `accuracy_kits.py` evaluator over the LoadGen accuracy log.

## Measured costs

The one-case end-to-end preflight used `case_00000`, whose preprocessed tensor shape was `(1, 192, 384, 384)` and whose inference covered 50 sliding windows.

- Environment creation used the shared package cache and imports succeeded with Torch 2.5.1+cu124.
- Preprocessing plus repository checksum verification took 36.21 seconds wall time and 2.56 GB peak host RSS, using up to 8 CPU processes.
- The harness took 84.76 seconds wall time, including 9.87 seconds for GPU inference and slow shared-filesystem startup, with 3.11 GB peak host RSS.
- Official evaluation took 34.36 seconds wall time and 2.25 GB peak host RSS.
- GPU availability at launch was 67,781 MiB free. The wrapper checks immediately before each inference run and requires at least 20,000 MiB free.

The read-only prior snapshot used byte-identical benchmark source and the same model, manifest, package versions, route, and hardware. Its full run measured 2:24.34 preprocessing, 5:07.43 inference, and 1:10.43 evaluation.

## Resources and schedule

The machine exposes 8 CPU cores, about 2 TiB host RAM, and one shared NVIDIA A100 80GB. Package installation and preprocessing use CPU only. This target then performs exactly one local GPU run, followed by CPU evaluation; no Slurm job is submitted.

The current one-case measurement alone gives conservative bounds of about 26 minutes for preprocessing and 8 minutes for inference if case cost scales linearly. The identical prior full measurement is stronger route-specific evidence and projects the measured workload stages at about 9 minutes total. Allowing 10 minutes for a cold environment and shared-filesystem/GPU contention gives a 19-minute expected run. Fifteen minutes are reserved for artifact checks, `report.json`, and deliverable verification, leaving 26 minutes of the one-hour limit as contention margin.

The full run starts with `--from environment`, which removes every wrapper-owned environment, build product, and output and exercises the delivered script unchanged from nothing built. CPU work is not duplicated because there is only one target, and GPU work remains one run at a time.

## Re-plan triggers

After preprocessing and after the first inference progress appears, compare elapsed time with the 19-minute projection. If projection exceeds 45 minutes, preserve the remaining 15-minute reporting reserve and record the measured failure rather than starting additional work.

## Full-run outcome

The clean full run stayed inside the plan: preprocessing completed in 1:42.88 with all 43 checksums matching, the harness completed in 3:59.34 after its immediate 67,487 MiB free-memory check, and official evaluation completed in 1:47.12. No schedule change was needed. The evaluator printed `mean = 0.86168`, `kidney = 0.9347`, and `tumor = 0.7887`.
