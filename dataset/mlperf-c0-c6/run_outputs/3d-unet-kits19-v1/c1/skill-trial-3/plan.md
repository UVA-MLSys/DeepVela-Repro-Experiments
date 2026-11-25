# Reproduction plan

## Workload

- Target `3d-unet`: setup -> preprocess 43 manifest entries -> one local A100
  Offline AccuracyOnly run -> repository `accuracy_kits.py` evaluation.
- Inputs are the fixed raw KiTS19 directory and fp32 TorchScript model. `case_00400`
  is materialized from the documented `case_00185` duplicate.

## Preflight measurements

- Environment setup and local LoadGen build: approximately 2 minutes from an empty
  target work directory, using only the shared uv cache.
- One-case preprocessing (`case_00000`, output shape `(1, 192, 384, 384)`), including
  the repository checksum verification: under 1 minute.
- One-case A100 inference: 4.88118 seconds for 50 sliding-window subvolumes, with
  74,260 MiB free immediately before the run.
- One-case repository evaluation completed and printed all three Dice metrics.

## Resources and schedule

- Machine: 8 CPU cores, approximately 1.8 TiB available host memory, one shared
  NVIDIA A100-SXM4-80GB.
- Preprocessing uses four processes to leave CPU capacity for other sessions.
- There is one target, so its CPU setup and preprocessing run first. The sole GPU
  stage starts immediately afterward and checks for at least 20,000 MiB free. No
  other GPU work is launched by this reproduction.
- Evaluation uses four CPU processes after inference.

## Expected time and reserve

- Conservative projection: 30-40 minutes total. This allows roughly 15 minutes for
  preprocessing, 15 minutes for variable-size inference cases, and 8 minutes for
  setup/evaluation overhead.
- Reserve: 8 minutes for `report.json`, artifact/log validation, and a final wrapper
  syntax/contract check. The projected run fits the one-hour limit.

## Re-plan triggers

- After the first several full-run cases, compare observed per-case inference time
  to the 4.88118-second preflight and revise the projection if it threatens the
  reserve.
- Do not launch a new GPU run if free memory is below the wrapper's 20,000 MiB floor.
- On a stage failure, repair once for the shared failure type and resume through
  `reproduce.sh --from <stage>` so successful earlier stages are retained.
