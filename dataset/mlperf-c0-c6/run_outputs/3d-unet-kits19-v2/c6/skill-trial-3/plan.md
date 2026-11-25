# Reproduction plan

## Workload

Target `3d-unet` has three stages. Setup creates a Python 3.11 environment from the required shared offline `uv` cache and builds this source tree's LoadGen. Preprocessing reads and checksum-verifies all 43 cases in `meta/inference_cases.json`, mapping documented duplicate `case_00400` to raw `case_00185`. The GPU stage runs the fp32 TorchScript model in Offline accuracy mode on the local A100, and then the repository's `accuracy_kits.py` evaluates the LoadGen accuracy log.

## Measured costs

The current bounded end-to-end preflight used `case_00000`, with both the copied manifest and checksum list reduced to that same input. Environment creation, dependency installation, and LoadGen compilation took about 109 seconds. One-process preprocessing plus checksum verification took under 30 seconds. The case produced shape `(1, 192, 384, 384)` and required 50 sliding-window sub-volumes; inference took 7.619 seconds while another session held substantial GPU utilization. The evaluator completed and printed all requested metric fields. Immediately before inference the A100 had 64,729 MiB free.

The provenance-verified prior snapshot for the same arm and source route completed the full 43-case preprocessing in 2 minutes 50 seconds, inference in 3 minutes 31 seconds including loading, and evaluation in 1 minute 25 seconds. Its required artifacts match the supplied SHA-256 provenance entries.

## Resources and schedule

Setup and preprocessing are CPU-only. The full preprocessing stage uses four processes. GPU work is one run at a time and begins only after the script records current `nvidia-smi` state and confirms at least 16,000 MiB free. The full target runs serially through setup, preprocessing, inference, and evaluation because it is the only target and inference depends on preprocessing. No process from another session will be modified or cancelled.

The current one-case rate conservatively projects about 5.5 minutes for inference if all cases had the same 50-window cost; the prior full run covered the actual shape distribution and took 3.5 minutes. Allowing for shared-GPU contention, the plan budgets 8 minutes for inference, 4 minutes for setup, 5 minutes for preprocessing and verification, and 3 minutes for evaluation.

## Expected time and reserve

Expected full-run time is at most 20 minutes. Ten minutes are reserved for a repair if necessary, exact metric extraction, artifact validation, `report.json`, and a final deliverable check. The full run uses `OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet` from the untouched default work path.

## Full-run completion

The clean full run completed without repair and within the original projection. Setup, all 43 preprocessing operations and checksum verification, the complete 43-query GPU accuracy run, and the 43-case repository evaluator all succeeded. The immediate GPU check recorded 81,037 MiB free and 0% utilization; the run log records `pytorch device:  cuda:0`. The evaluator printed `Accuracy: mean = 0.86168, kidney = 0.9347, tumor = 0.7887`.
