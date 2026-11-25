# Reproduction plan

## Workload

Target `3d-unet` has three stages: build a Python 3.11 environment and the repository's LoadGen; preprocess and checksum-verify all 43 cases in `meta/inference_cases.json` (mapping documented duplicate `case_00400` to raw `case_00185`); run the fp32 TorchScript model in Offline accuracy mode on the local A100; then run `accuracy_kits.py` over the LoadGen accuracy log.

## Measured costs

A bounded end-to-end run of `case_00000` completed successfully. Environment creation and LoadGen compilation took about 50 seconds. Preprocessing plus checksum verification took about 30 seconds with one process. The case produced shape `(1, 192, 384, 384)` and required 50 sliding-window sub-volumes; GPU inference took 5.215 seconds. The evaluator completed and printed all three requested metrics. Immediately before inference the A100 had 78,276 MiB free.

## Resources and schedule

The machine has one shared NVIDIA A100 80GB and about 1.8 TiB available host memory. Setup and preprocessing are CPU-only. The full preprocessing stage uses four processes; a conservative serial extrapolation is 22 minutes, while four-way execution should be substantially shorter. GPU work is one run at a time and starts only after checking for at least 16,000 MiB free. The measured inference rate projects about four minutes for 43 similarly costly cases; variable case shapes are covered by a conservative 10-minute GPU allowance. Evaluation uses four CPU processes after inference.

Concurrent sessions were observed in the default target work path, so this session uses a distinct target-local work directory and never modifies or cancels their processes. It checks GPU state immediately before its run and does not launch while another owned GPU run is active.

## Expected time and reserve

Expected full-run time is 20 minutes: 1 minute setup, up to 10 minutes preprocessing and verification, up to 7 minutes inference, and 2 minutes evaluation. A 10-minute reserve is held for repair, exact report extraction, artifact checks, and a final clean-copy driver check.

## Full-run re-plan and completion

The full setup marker was written at 02:50:02, preprocessing finished at 02:52:52 (2 minutes 50 seconds), inference finished at 02:56:23 (3 minutes 31 seconds including model/QSL loading), and evaluation finished at 02:57:48 (1 minute 25 seconds). An unrelated process joined the shared GPU after this run's free-memory check, increasing later per-case times, but the measured projection still remained below the reserve boundary, so no schedule change was needed. All 43 queries and all evaluator cases completed.
