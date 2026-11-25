# Full reproduction plan

Written after the bounded end-to-end preflight on 2026-11-18.

## Workload

One target, `3d-unet`, runs through three stages in `reproduce.sh`: (1) create a Python 3.11 environment, compile the supplied LoadGen, and create a repaired target-local source/data view; (2) preprocess the fixed 43-case inference manifest on CPUs; (3) run FP32 PyTorch Offline accuracy serially on the local A100 and invoke `accuracy_kits.py`. Each stage writes a success marker and the next stage reads only its predecessor's files.

## Measured costs

- Environment/build: the preflight environment completed in about 2 minutes including cached package materialization and a 34-second LoadGen compile. It used CPUs and filesystem only.
- Preprocessing: `case_00000` completed in about 20 seconds including Python startup and produced shape `(1, 192, 384, 384)`. The full stage uses four worker processes; case sizes vary, so a linear single-worker extrapolation is deliberately not used.
- GPU inference: `case_00000` required 50 sliding windows and 5.90220 seconds on the A100 while another job was active.
- Evaluation/postprocessing: the one-case evaluator completed in under 8 seconds.
- GPU memory check immediately before preflight showed 67,799 MiB free with another job using 13,355 MiB. The reference processes one window/query at a time and the bounded run fit comfortably.

## Resources

The machine exposes 8 CPU cores, 2.0 TiB RAM (1.8 TiB available at planning), and one shared NVIDIA A100-SXM4-80GB. The prompt assigns all work locally and prohibits Slurm. GPU stages are serialized by the single target; preprocessing uses four of eight CPU cores. The script rechecks free GPU memory immediately before the full GPU stage.

## Schedule and projection

Run `./reproduce.sh --target 3d-unet --from env` with the canonical `OUTPUT_DIR=./3d-unet/out`. This deliberately rebuilds from no reusable preflight environment or preprocessed files. There is no other target whose CPU work can overlap the GPU stage.

The conservative projection is 2 minutes for environment setup, 5 minutes for four-worker preprocessing, 15 minutes for inference (more than 3x the one-case linear projection of 4.3 minutes), and 2 minutes for evaluation/validation: 24 minutes total. This fits within the remaining task budget at launch.

## Reserve and re-plan triggers

Reserve 8 minutes for evaluator completion, exact-value extraction, `report.json`, output validation, and failure reporting. After the first several full-run cases, compare observed seconds/window and completion rate with the 15-minute inference allowance. If contention projects past the reserve boundary, record the new projection here; do not change backend, precision, data, windowing, model, scenario, or evaluator.
