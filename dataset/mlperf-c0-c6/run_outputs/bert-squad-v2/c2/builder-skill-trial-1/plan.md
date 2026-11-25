# Full reproduction plan

## Workload

One target, `bert-squad`, runs three ordered stages through the delivered `reproduce.sh`:

1. `environment`: verify fixed input hashes, create a Python 3.11 environment, install pinned packages from the mandated uv cache, and build the supplied LoadGen source;
2. `preprocess`: link the allowed staged dependency, apply the QSL/device-log repair, and convert all 10,570 questions into 10,833 length-384 BERT windows;
3. `run`: inspect current GPU free memory, process all windows with PyTorch fp32 in LoadGen Offline AccuracyOnly mode, then run `accuracy-squad.py` and `evaluate_v1.1.py`.

Each stage consumes the preceding stage's files and writes a success marker only after all commands succeed.

## Measured costs

- Current environment preflight: 41.15 seconds, 546 MB maximum host RSS; package installation used the prescribed shared cache and local LoadGen compilation dominated the cost (`bert-squad/out/environment.time.log`).
- Current complete preprocessing: 28.21 seconds, 865 MB maximum host RSS; 10,833 windows covering all 10,570 questions (`bert-squad/out/preprocess.time.log`).
- Current 10-question end-to-end preflight: 29.56 seconds, 2.32 GB maximum host RSS, 7.36 seconds inside LoadGen; it passed CUDA device logging, LoadGen, and evaluator output (`bert-squad/out/preflight-run.time.log`, `mlperf_log_detail.txt`, `accuracy.txt`).
- Same-arm prior trial, permitted by `prior-artifacts/provenance.json`: a 256-window run measured 26.237 seconds from query issue to completion (102.5 ms/window), and its clean full replay completed in 1,328.76 seconds (22.15 minutes), as recorded in `prior-artifacts/plan.md`.
- GPU launch inspection before current preflight: 75,558 MiB free of 81,920 MiB; three unrelated processes occupied about 5.6 GiB. The script rejects launches below 8 GiB free and records another inspection immediately before inference.

## Resources and schedule

The machine exposes 8 CPU cores, about 1.9 TiB available host memory, and one shared NVIDIA A100-SXM4-80GB GPU. There is one target, so CPU stages run first and no independent target work is available to overlap. GPU work runs only once at a time, directly on the machine, with no Slurm submission.

The 256-window rate projects about 18.5 minutes for full inference. Adding environment verification, preprocessing, model load, evaluation, filesystem variance, and shared-GPU contention gives a conservative 25–35 minute total. The full replay is launched through the unchanged delivered script with `--from environment`, which invalidates all stage markers and script-owned derived/output files before running every stage.

## Reserve

Reserve 20 minutes before the two-hour limit for metric parsing, `report.json`, output validation, and a standalone-script inspection. The conservative projection leaves at least 65 minutes beyond that reserve.

## Execution outcome

The full replay completed successfully in 1,281.37 seconds (21.36 minutes), within the projected range. LoadGen reported no warnings or errors over all 10,833 feature windows. The full evaluator printed exact match `83.68968779564806` and F1 `90.87487229720105`.
