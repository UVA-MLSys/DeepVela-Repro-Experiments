# Reproduction plan

## Workload

Target `3d-unet` has four ordered stages: build a target-local Python environment and copy the benchmark source; preprocess and checksum-verify all 43 inference cases; run one fp32 PyTorch TorchScript Offline accuracy pass on the local A100; run `accuracy_kits.py` over the complete LoadGen accuracy log. The evaluator depends on all inference outputs, so it follows the GPU stage.

## Measured costs

The clean scripted one-case preflight installed 38 packages from the shared cache in 7.23 seconds. `case_00000` preprocessing took 20.23 seconds at 2,559,720 KiB peak RSS and checksum verification took 2.67 seconds. The GPU stage took 20.10 seconds end to end; its 50 sliding-window inferences took 4.87 seconds and peak host RSS was 3,304,936 KiB. Evaluation took 8.10 seconds at 2,253,396 KiB peak RSS. The GPU check immediately before the preflight showed 78,276 MiB free.

## Resources and schedule

The machine has 128 CPU cores, about 1.8 TiB available RAM, and one shared NVIDIA A100 80GB. Preprocessing uses eight processes: scaling the measured per-case peak gives about 20.5 GB, safely below available memory while limiting filesystem contention. The accelerator stage runs alone and checks free memory immediately before launch. Evaluation uses four CPU processes after inference. With one target, there is no independent ready CPU stage to overlap with its GPU stage.

## Expected time and reserve

A linear upper estimate from the one-case measurements is 14.5 minutes for sequential preprocessing and 14.4 minutes for GPU/evaluation overhead, while the selected CPU parallelism should reduce preprocessing and evaluation substantially. Allowing for larger cases and shared-filesystem variance, the full projection is 25 minutes. Ten minutes are reserved for artifact validation, `report.json`, and one scoped repair if needed, leaving 25 minutes of additional margin under the one-hour limit.

## Re-plan: early full-run rate

After 55 seconds of full preprocessing, 17 of 43 case pickle files existed (0.31 cases/second), projecting about 2.3 minutes for preprocessing. Worker RSS was below 0.4 GB each at that observation. This is faster and smaller than the conservative one-case scaling, so eight preprocessing workers remain appropriate and the total projection remains 25 minutes including variance.

## Completion

Full preprocessing took 2:51 and verification took 0:26. The 43-case GPU pass took 4:29, and evaluation took 2:35. All stages completed in about 10 minutes after launch, within the projection and reserve.
