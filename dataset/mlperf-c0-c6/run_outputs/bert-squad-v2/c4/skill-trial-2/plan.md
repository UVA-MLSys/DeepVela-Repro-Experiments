# Full reproduction plan

- Workload: one `bert-squad` target. `setup` copies and patches the pinned source and builds an offline Python 3.11 environment; `preprocess` converts all 10,570 SQuAD v1.1 questions into 10,833 BERT features; `run` performs the PyTorch FP32 Offline accuracy pass on GPU 0 and invokes `accuracy-squad.py`, which calls `evaluate_v1.1.py`.
- Measured costs: current clean setup 44 seconds; current full preprocessing 60.07 seconds at 888,688 KiB peak RSS; current 16-feature GPU stage 17.00 seconds at 2,320,980 KiB peak host RSS. The permitted prior snapshot measured its same-source full GPU stage at 16:27.16.
- Resources: 8 CPU cores, about 1.9 TiB available host RAM, and one NVIDIA A100-SXM4-80GB. The current preflight GPU check reported 72,504 MiB free. GPU work is serial and each invocation checks for at least 8,192 MiB free immediately before launch.
- Schedule: run clean setup, preprocessing, GPU inference, and evaluation in dependency order for the sole target. There is no other target with CPU work to overlap. Preserve all stage logs and success markers so a resume skips completed stages.
- Expected time: approximately 19 minutes total, using the current setup/preprocessing measurements and the prior full inference measurement from the same route.
- Reserve: retain at least 15 minutes before the two-hour limit to parse evaluator output, write `report.json`, and verify the standalone runner and required artifacts.

## Actual

- Clean setup completed in 44 seconds and preprocessing completed in 59.91 seconds.
- Full GPU inference completed in 16:17.51 with no LoadGen warnings or errors; evaluation then completed successfully. The run stayed within the original projection and reserve, so no schedule change was required.
