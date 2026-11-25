# Reproduction plan

## Workload

- Target `3d-unet`: build an isolated Python/PyTorch and repository LoadGen environment; preprocess and checksum-verify all 43 entries in `meta/inference_cases.json`; run the fp32 TorchScript model through `run.py --backend=pytorch --scenario=Offline --accuracy`; evaluate the resulting LoadGen accuracy log with `accuracy_kits.py`.
- Build produces the environment and restores the two missing documented PyTorch adapter files. Preprocessing depends on that environment and the fixed raw-data cache. GPU inference depends on all preprocessing and the supplied model. Evaluation depends on the full LoadGen accuracy log.

## Measured costs

- Bounded case: `case_00005`, the largest compressed inference input, with preprocessed shape `(1, 256, 448, 448)`.
- Clean end-to-end preflight: 4m09s wall time, 3.74 GiB peak host RSS, including a cold local Python install and repository LoadGen build.
- Preflight GPU check: 78,277 MiB free immediately before launch; one-case inference and evaluator both completed, and the model reported `pytorch device: cuda:0`.
- The read-only same-arm prior trial measured the complete 43-case route successfully and projected 15–25 minutes total after a 13.35s single-case GPU measurement.

## Resources and schedule

- Run locally on the one NVIDIA A100 80GB; never use Slurm. The launcher checks memory immediately before each GPU stage and requires at least 12,000 MiB free.
- Use 8 preprocessing/evaluation processes to limit contention and memory. There is one target, so there is no independent CPU stage to overlap once its GPU stage begins.
- Launch the delivered script with `--from build`. It removes its own preflight products and executes build, preprocessing, GPU inference, and evaluation in order, leaving success markers for safe resumption.

## Expected time and reserve

- Expected full-run wall time: 15–25 minutes, driven by the prior complete run and bounded measurement. Full preprocessing covers 43 entries; GPU inference covers all 43 sequential LoadGen samples.
- Hold 8 minutes before the one-hour limit for `report.json`, artifact checks, and a clean-runner audit.

## Completion

- The clean full run completed within the projection. All 43 preprocessed checksums matched, all 43 samples were run, and the official evaluator printed `Accuracy: mean = 0.86168, kidney = 0.9347, tumor = 0.7887`.
