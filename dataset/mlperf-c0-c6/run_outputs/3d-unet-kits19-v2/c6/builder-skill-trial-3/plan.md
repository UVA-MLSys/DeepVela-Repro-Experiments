# Reproduction plan

Created 2026-11-19 after a successful one-case end-to-end preflight.

## Workload

Single target `3d-unet`: environment -> preprocess 43 fixed inference cases -> A100 fp32 PyTorch Offline accuracy run -> benchmark evaluator -> required outputs and `report.json`.

## Measured costs and resources

- Environment: about 90 seconds, CPU/filesystem only; Python 3.12.12 and cached wheels, no network.
- One-case preprocessing plus checksum verification: about 30 seconds with four CPU processes.
- One-case A100 inference: 4.3 seconds for 50 windows. The preflight memory check observed 81,037 MiB free.
- One-case evaluator: about 8 seconds with four CPU processes.
- Prior same-arm provenance records a complete 43-case run and shows roughly four minutes of per-case GPU inference; it is supporting evidence, not the measured result used in the report.

## Schedule

Run the delivered script unchanged from its `environment` stage, which clears the bounded preflight artifacts and reconstructs the full 43-case manifest. CPU environment/preprocessing runs first. Immediately before the sole GPU run, query `nvidia-smi`; require at least 24,000 MiB free. Run GPU work serially, then evaluate on CPU. There are no other targets whose CPU stages can overlap this GPU stage.

## Expected time and reserve

Budget 15 minutes for the full clean run, dominated by preprocessing and approximately four minutes of inference. Hold 10 minutes in reserve to parse the exact evaluator line, write `report.json`, and validate the clean-copy deliverable and output/log inventory. Re-plan if early full-run rates project beyond 50 minutes total.
