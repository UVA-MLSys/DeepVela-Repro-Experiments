# Reproduction plan

Recorded 2026-11-22 after bounded preflight, before the full run.

## Workload

One target, `bert-squad`, with three ordered stages implemented by the delivered
`reproduce.sh`: (1) create a Python 3.11 environment and build repository LoadGen, (2) stage the
fixed inputs and generate deterministic BERT feature records, and (3) run the PyTorch FP32 Offline
AccuracyOnly path on the local A100 followed by `accuracy-squad.py`/`evaluate_v1.1.py`.

## Measured costs

| Stage | Bounded measurement | Resource use |
|---|---:|---|
| Environment | approximately 45-50 s from an empty work directory | 8 host CPUs available; package cache reads and C++ compilation; no GPU |
| Prepare | approximately 55 s for all 10,570 examples / 10,833 features | CPU-only; feature pickle about 70 MiB |
| Run/evaluate | 20 s for 8 features; 45.564 s for 256 features | one A100; 73,846-75,558 MiB free immediately before bounded launches |

The difference between the 8- and 256-feature measurements is 25.6 s for 248 additional features,
or about 0.103 s/feature after fixed startup. This projects 10,833 features to roughly 1,136 s
(18.9 min) plus full-log post-processing uncertainty. The full target is therefore budgeted at
22 minutes for GPU inference/evaluation and 2 minutes for clean environment/preparation.

## Resources and schedule

The machine provides 8 visible CPU cores, about 1.9 TiB available host memory, and one shared
NVIDIA A100-SXM4-80GB. GPU work runs one at a time and directly on GPU 0, never through Slurm.
The sole target has no independent CPU target to overlap, so its stages run environment -> prepare
-> run. `nvidia-smi` is checked by the script immediately before the run stage; the launch proceeds
because the bounded checks showed more than 73 GiB free while the model requires only a small
fraction of the 80 GiB device.

## Expected time and reserve

Expected clean replay: about 24 minutes. The task limit is 120 minutes. Reserve 20 minutes for
metric extraction, `report.json`, output validation, and a clean-copy/script audit. The projected
finish is therefore comfortably before the 100-minute no-new-work boundary.

The completed clean replay took 1,332.406 seconds (22.21 minutes), so no mid-run re-plan was
required.

Closeout validation found that relative `OUTPUT_DIR` needed normalization before benchmark
subprocesses changed directories. After that repair, the delivered script was replayed again from
the `environment` stage with the exact setting `OUTPUT_DIR=./bert-squad/out`; that final clean
replay took 629.685 seconds (10.49 minutes) under reduced GPU contention and reproduced the exact
same EM/F1 values.
