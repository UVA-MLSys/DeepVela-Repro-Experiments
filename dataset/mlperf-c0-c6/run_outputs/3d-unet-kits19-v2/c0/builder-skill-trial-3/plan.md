# Full reproduction plan

## Workload and dependencies

Target `3d-unet` has three serialized stages: (1) build a pinned Python 3.11 uv environment and compile the supplied LoadGen; (2) stage and preprocess the fixed 43-case inference split on CPU; (3) after an immediate GPU-memory check, run fp32 PyTorch/TorchScript Offline AccuracyOnly inference on the local A100, then run the benchmark-owned CPU evaluator and validate outputs. Each stage writes a success marker consumed by the next.

## Measured costs

The one-case clean preflight completed successfully on 2026-11-19:

- environment: about 3 minutes, including an 87-second LoadGen build and slow initial imports;
- one-case preprocessing: 55 seconds with four workers;
- one-case GPU run: 21 seconds, including model/startup overhead;
- one-case evaluation and validation: 7 seconds;
- GPU launch state: 80,475 MiB free of 81,920 MiB; CUDA device validation passed.

Comparable same-arm prior-artifact timestamps for the complete 43-case scope show about 3 minutes preprocessing, 4.5 minutes inference, and 3.5 minutes evaluation/validation. Its exact 43-record validation succeeded. These full-scope measurements are used because the one-case fixed overhead does not scale linearly.

## Resources and schedule

Run directly on GPU 0 (NVIDIA A100-SXM4-80GB), never through Slurm. The script requires at least 16,384 MiB free immediately before inference and runs one GPU workload. Environment and preprocessing use four CPU workers where supported. With one target there is no independent CPU stage to overlap with its GPU stage.

Launch the delivered script unchanged from a clean script-managed reset:

```bash
OUTPUT_DIR="$PWD/3d-unet/out" ./reproduce.sh --target 3d-unet --from environment
```

Projected total is approximately 14 minutes: 3 minutes environment, 3 minutes preprocessing, 4.5 minutes inference, 3.5 minutes evaluation and validation. Hold a 10-minute reserve within the one-hour limit for result extraction, `report.json`, artifact checks, and failure repair. Read each stage log as it appears. If GPU free memory falls below the gate, do not launch inference; record the evidence rather than competing with other jobs.
