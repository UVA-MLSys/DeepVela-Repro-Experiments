# Reproduction plan

## Workload and dependencies

One target, `3d-unet`, runs in four file-backed stages: `env -> preprocess -> infer -> evaluate`. Environment copies the source and builds the offline cached Python/LoadGen environment. Preprocessing transforms all 43 ordered cases, including a target-local copy for case_00400. Inference is the sole GPU stage and writes LoadGen logs. Evaluation decodes all predictions and computes benchmark-owned Dice outputs.

## Measured costs and resources

- Preflight environment/build: 4m12s, 563 MiB peak RSS.
- One-case preprocessing: 24.3s, 992 MiB peak RSS; full prior hash-verified local evidence was about 3m with four processes.
- One-case GPU inference: 31.9s, 3.53 GiB peak host RSS; 64,729 MiB free immediately before launch. Hash-verified prior full evidence was about 6m for 43 cases.
- One-case evaluation: 14.4s, 4.14 GiB peak RSS; prior full evidence was about 1m50s.
- Available machine resources at planning: 8 visible CPU cores, 2 TiB RAM (about 1.8 TiB available), one shared NVIDIA A100 80GB. The script requires at least 12 GiB free before inference.

## Schedule and expected time

Run `./reproduce.sh --target 3d-unet --from env` unchanged with `OUTPUT_DIR=./3d-unet/out`. CPU stages use four workers and precede the single GPU stage. There is no second target with ready CPU work to overlap. Projected total is approximately 15 minutes (5m build + 3m preprocess + 6m infer + 2m evaluate), driven by the preflight and hash-verified same-arm full timings.

Hold 10 minutes of the one-hour limit for output validation, exact-string `report.json`, and clean-replay checks. If observed early timing projects beyond 50 minutes, stop starting new work and report measured evidence.
