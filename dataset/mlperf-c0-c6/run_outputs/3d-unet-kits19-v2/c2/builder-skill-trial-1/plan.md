# Full reproduction plan

Written after the bounded end-to-end preflight on 2026-11-19.

## Workload

The sole target, `3d-unet`, uses three `reproduce.sh` stages: (1) create Python 3.11 environment, compile supplied LoadGen, and create the repaired workspace-local source/data view; (2) preprocess the fixed 43-case manifest on CPU; (3) run FP32 PyTorch Offline accuracy on the local A100 and invoke `accuracy_kits.py`. Success markers are written only after each whole stage succeeds.

## Measured costs

- Entire two-case clean preflight: 5:05 wall time, 5.5 GiB peak host RSS.
- Environment/build: about 3:48, including cached packages, LoadGen compilation, and the slow shared-GPU CUDA probe.
- Preprocessing: 44 seconds for `case_00185` and its `case_00400` alias with two processes; both produced `(1,384,384,384)`.
- GPU inference: 125 windows took 14.82 and 10.21 seconds while the shared GPU showed 100% utilization from concurrent work.
- Run plus evaluator: 52 seconds for two cases.
- GPU memory: `nvidia-smi` showed 76,045 MiB free and PyTorch showed 79.30 GB free immediately before inference.
- Prior provenance-backed full trial: post-build timestamps show about 4:19 preprocessing, 4:21 inference, and 1:35 evaluation for the 43-case workload (`prior-artifacts/3d-unet/out/*.log`, `prior-artifacts/provenance.json`).

## Resources and schedule

All stages run locally; Slurm is prohibited. CPU preprocessing uses four processes. The one GPU stage runs serially and starts only after preprocessing succeeds. There is no second target whose CPU work can overlap it. Launch the delivered script under clean-run conditions with `OUTPUT_DIR=$PWD/3d-unet/out ./reproduce.sh --target 3d-unet --from env`; this clears the wrapper's own full-run state and rebuilds without reusing preflight artifacts.

The conservative projection is 4 minutes environment, 6 minutes preprocessing, 8 minutes inference under contention, and 2 minutes evaluation/validation: 20 minutes. The observed preflight rate and prior full trial both fit this allowance.

## Reserve and re-plan triggers

Reserve 8 minutes for evaluator completion, exact result extraction, `report.json`, output validation, and failure recording. After several full-run samples, compare seconds per window and completion rate with the 8-minute inference allowance. If contention projects past the reserve boundary, record a revised projection here without changing backend, precision, dataset, preprocessing, model, scenario, or evaluator.

## Completion update

The canonical script completed all stages in 13:12.79, below the 20-minute projection. All 43 samples completed and the evaluator exited successfully. The optional outer transcript tee failed to open before `OUTPUT_DIR` existed, but this did not affect the script; its own timed exit status was 0 and its required internal logs and success markers were written.
