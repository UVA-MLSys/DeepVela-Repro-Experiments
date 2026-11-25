# Full reproduction plan

- Workload: `3d-unet` environment, 43-case verified preprocessing, one local A100
  Offline fp32 accuracy run, then the repository evaluator.
- Measured costs: environment 27m51s (dominated by first shared-cache PyTorch
  transfer); full preprocessing and checksum verification 2m49s on 4 CPU workers;
  bounded `case_00000` inference 5.47s for 50 sub-volumes; bounded evaluation 1.1s.
- Resources at preflight: 78,276 MiB of 81,920 MiB GPU memory free. GPU work is
  serialized; preprocessing/evaluation use 4 CPU processes.
- Schedule: reuse the successfully marked environment and preparation stages;
  run all 43 cases on the GPU, then evaluate the complete LoadGen accuracy log.
- Expected remaining time: 6--9 minutes inference plus about 1 minute evaluation,
  based on bounded sub-volume rate and the prepared case shapes.
- Reserve: 3 minutes for `report.json`, output checks, and delivered-script checks.

## Actual / re-plan

- Full inference completed all 43 cases in 241.7s; the bounded rate correctly
  projected completion inside the reserve.
- Evaluation completed in about 71s. No schedule change was needed.
- The final accuracy log contains 43 records and 43 unique QSL indices.
