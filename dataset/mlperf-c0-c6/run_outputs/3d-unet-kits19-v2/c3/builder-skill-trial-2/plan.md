# Full reproduction plan

Planned 2026-11-19 after the required bounded preflight.

## Workload and dependencies

One target, `3d-unet`, runs three stages in order through the delivered `reproduce.sh`:

1. `env`: create a Python 3.11 environment and build the included LoadGen;
2. `prepare`: copy the benchmark app, link the fixed raw cases, and preprocess all 43 cases;
3. `run`: check current GPU memory, run Offline fp32 accuracy inference serially on the local
   A100, then run the benchmark evaluator on the LoadGen accuracy log.

Each successful stage writes its own marker. The full validation run will use
`OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet --from env`, which removes the
bounded preflight's target-local environment, derived data, markers, and outputs before
rebuilding with the full unmodified 43-case manifest.

## Measured costs and resources

The one-case preflight through the same script measured:

| Stage | Wall time | Resource evidence |
|---|---:|---|
| Environment | 1m58s | Local CPU; shared uv cache; LoadGen compiled successfully |
| Preprocessing | 38s | 4 processes; checksum verified for the reduced manifest |
| GPU inference | 15s | 67,487 MiB free at launch; one case took 5.61s model inference |
| Evaluation | 6s | 4 processes; evaluator printed all three metrics |

Machine resources at planning time were 8 visible CPU cores, about 1.8 TiB available host
memory, and one shared A100 80 GB. A separate job occupied about 13.7 GiB during preflight;
the script therefore checks memory afresh and requires at least 12,000 MiB immediately before
each GPU stage rather than relying on the planning snapshot.

The read-only prior full-scope trial measured 2m20s environment setup, 4m04s preprocessing,
5m12s GPU inference, and 2m02s evaluation: 13m38s stage time. This full-scope measurement is
the primary estimate because one-case fixed/model costs do not scale linearly.

## Schedule, projection, and reserve

There is only one target, so all GPU work is serial. Environment and preprocessing precede GPU
work because the run consumes their files; there is no independent second target whose CPU work
can overlap. Projected full-run time is 14 minutes, with 6 minutes allowed for filesystem/cache
variance and GPU sharing (20-minute operating projection). Reserve 10 minutes for one scoped
rerun if needed, exact result extraction, `report.json`, and artifact verification. The resulting
30-minute plan is below the one-hour limit.

Re-plan if a stage fails, if free GPU memory falls below the script's guard, or if observed
stage duration projects beyond 50 minutes (the one-hour limit less the 10-minute closeout
reserve). No semantic changes to model, backend, precision, cases, preprocessing, scenario,
or evaluator are permitted.
