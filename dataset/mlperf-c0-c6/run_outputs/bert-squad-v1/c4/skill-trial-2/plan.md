# Full reproduction plan

- Workload: one `bert-squad` target. `setup` copies and patches the pinned source and builds an offline Python 3.11 environment; `preprocess` converts all 10,570 SQuAD v1.1 questions into 10,833 BERT features; `run` performs the PyTorch FP32 Offline accuracy pass on GPU 0 and invokes `accuracy-squad.py`, which calls `evaluate_v1.1.py`.
- Measured costs: clean setup 45 seconds; full preprocessing 60.08 seconds at 882,324 KiB peak RSS; 16-feature GPU stage 15.76 seconds; 256-feature GPU stage 32.46 seconds at 2,300,892 KiB peak host RSS. The incremental accelerator cost is about 69.6 ms per feature.
- Resources: 8 CPU cores, about 1.9 TiB available host RAM, and one NVIDIA A100-SXM4-80GB. The most recent preflight GPU check reported 77,395 MiB free. GPU work is serial and each invocation checks for at least 8,192 MiB free immediately before launch.
- Schedule: run setup, preprocessing, GPU inference, and evaluation in dependency order for this single target. There is no second target with ready CPU work to overlap. Preserve all stage logs and success markers so a resume skips completed stages.
- Expected time: approximately 12.7 minutes for full GPU inference from the two bounded measurements, and about 18 minutes total including clean setup, preprocessing, evaluator, and filesystem overhead.
- Reserve: retain at least 15 minutes before the two-hour limit to parse evaluator output, write `report.json`, and verify the standalone runner and required artifacts.

## Re-plan during full run

- Trigger: after 9.4 minutes of the full GPU stage, the accuracy log had reached 35.6 MiB while GPU utilization remained 100%; another session had also raised total device use by about 1.9 GiB.
- Change: revise the accelerator projection from 12.7 minutes to approximately 18 minutes and total completion from 18 to about 21 minutes. Keep the single-GPU serial schedule unchanged because 70.6 GiB remained free and the revised projection is far inside the two-hour limit and 15-minute reserve.
- Actual: full GPU inference completed in 16:27.16 with no LoadGen warnings or errors; evaluation then completed successfully. The revised plan remained within reserve.
