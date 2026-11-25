# Full reproduction plan

## Workload

Target `3d-unet` has four stages: create the native Python environment; preprocess and checksum-verify all 43 manifest entries (42 distinct raw cases plus the documented `case_00400` duplicate); run fp32 TorchScript PyTorch inference in Offline accuracy mode on the local A100; then run the repository's `accuracy_kits.py` evaluator over the LoadGen accuracy log.

## Measured costs

The repaired one-case end-to-end preflight used `case_00000`, whose preprocessed tensor shape was `(1, 192, 384, 384)` and whose inference covered 50 sliding windows.

- Environment creation and cached package installation: approximately 1 minute; imports succeeded with Torch 2.5.1+cu124.
- Preprocessing plus repository checksum verification: 32.28 seconds wall time, 2.56 GB peak host RSS, using up to 8 CPU processes.
- Harness: 13.43 seconds wall time, including 4.54 seconds for the 50-window GPU inference, and 3.10 GB peak host RSS.
- Official evaluation: 6.48 seconds wall time and 2.25 GB peak host RSS.
- GPU availability at the repaired preflight launch: 74,260 MiB free. The wrapper requires at least 20,000 MiB free immediately before each inference run.

## Resources and schedule

The machine exposes 8 CPU cores, about 2 TiB host RAM, and one shared NVIDIA A100 80GB. Package installation and all preprocessing use CPU only. They run first and may overlap GPU work belonging to other sessions. This target then performs exactly one local GPU run, followed by CPU evaluation; no Slurm job is submitted.

The one-case cold preprocessing measurement gives a deliberately conservative serial upper projection of 23.1 minutes for 43 entries. Eight-way preprocessing should be materially shorter. Scaling GPU compute by 43 gives about 3.3 minutes for cases of the measured size; allowing for varying case sizes and startup gives a 12-minute inference allowance. Evaluation gets 5 minutes.

Expected total is at most 42 minutes under the conservative projection. Ten minutes are reserved for artifact checks, `report.json`, and a clean-deliverable validation, leaving 8 minutes of the one-hour limit as contention margin. The full run starts with `--from environment`, which removes every wrapper-owned build/output and exercises the delivered script unchanged from nothing built.

## Re-plan after measured progress

Full preprocessing completed in 2:24.34 with every checksum matching, rather than the conservative 23.1-minute bound. Full inference then completed in 5:07.43 despite later GPU contention, and evaluation completed in 1:10.43. The projection therefore moved earlier, not past the reserve: no schedule or parallelism change was needed. Only the evaluator's output-capture stage is repeated during closeout so `accuracy.txt` contains stdout alone; it reuses the completed full LoadGen accuracy log and does not repeat GPU work.
