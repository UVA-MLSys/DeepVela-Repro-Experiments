# Reproduction plan

## Workload

One target, `3d-unet`, runs three file-backed stages in order: environment, data, and run. The environment stage copies the pinned source and creates an offline uv environment; data links the read-only raw cases/model into the work copy and preprocesses all 43 inference IDs (mapping documented `case_00400` to `case_00185`); run checks GPU memory, executes the PyTorch fp32 Offline AccuracyOnly command, and invokes `accuracy_kits.py`.

## Measured costs

The integrity-verified prior preflight ran one manifest/checksum-consistent case end to end: environment package/build operations took 0.03s, 5.72s, and 35.08s (peak 545 MB); preprocessing took 24.11s (peak 2.56 GB); inference took 11.99s (peak 3.06 GB); evaluation took 6.02s (peak 2.25 GB). The prior full 43-case run measured 33.79s for the main environment install/build operation, 2m46.28s preprocessing (peak 3.48 GB), 3m20.27s inference (peak process RSS 11.46 GB), and 1m44.75s evaluation (peak 9.12 GB). Evidence: `prior-artifacts/3d-unet/{build,preprocess,run,evaluator}.log` and corresponding `preflight/*.log`; all 266 files passed `provenance.json` SHA-256 and size checks.

## Resources and schedule

Run directly on GPU 0, NVIDIA A100-SXM4-80GB. Immediately before planning, `nvidia-smi` reported 67,459 MiB free. The script requires at least 20,000 MiB and rechecks before inference. CPU-only environment setup and preprocessing precede the only GPU stage because this single target has no independent ready work to overlap. GPU work is serialized by construction. The shared uv cache is `/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache/uv`; no network or alternate cache is used.

## Expected time and reserve

Expected full wall time is under 10 minutes based on the measured approximately 8-minute prior full pipeline plus filesystem variance. Reserve 10 minutes before the one-hour limit for output validation, report generation, and clean-deliverable checks. If the first case rate projects beyond this allowance or a stage fails, re-plan here from its log and resume only through `./reproduce.sh --from <stage>`.
