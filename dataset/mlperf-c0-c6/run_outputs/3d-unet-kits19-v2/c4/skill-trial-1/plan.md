# Reproduction plan

## Workload

Target `3d-unet` has four ordered stages: create a target-local Python environment and copied benchmark worktree; preprocess and checksum-verify all 43 inference cases; run one fp32 PyTorch TorchScript Offline accuracy pass on the local A100; then run `accuracy_kits.py` over the complete LoadGen accuracy log. Evaluation depends on every inference result and therefore follows the GPU stage.

## Measured costs

The clean scripted `case_00000` preflight installed 38 packages from the required shared cache in 8.66 seconds. Preprocessing took 24.73 seconds at 2,560,136 KiB peak RSS and checksum verification took 3.55 seconds. The GPU stage took 32.98 seconds end to end at 3,326,412 KiB peak host RSS; evaluation took 8.84 seconds at 2,254,648 KiB peak RSS. The wrapper's GPU check immediately before inference showed 66,891 MiB free, and the model printed `pytorch device: cuda:0`. The checksum verifier passed and the benchmark evaluator completed.

## Resources and schedule

The machine has 128 CPU cores, about 1.8 TiB available RAM, and one shared NVIDIA A100 80GB. Preprocessing uses eight processes; scaling the measured per-case peak is still far below available memory while limiting shared-filesystem contention. The accelerator stage runs alone and records free memory immediately before launch. Evaluation uses four CPU processes after inference. There is only one target, so no independent ready CPU stage exists to overlap with its GPU stage.

## Expected time and reserve

A conservative linear projection from the preflight is under 30 minutes. More directly, the read-only same-arm prior trial on this source and input route completed full preprocessing in 2:51, the 43-case GPU pass in 4:29, and evaluation in 2:35, about 10 minutes total. The full projection is therefore 15 minutes allowing for current GPU and filesystem contention. Ten minutes are reserved for artifact validation, `report.json`, and a scoped repair, leaving 35 minutes of margin under the one-hour limit.

## Re-plan: early full-run rate

After about 48 seconds in the full preprocessing stage, 2 of 43 pickle files existed (4.7%). A flat-rate projection is roughly 17 minutes for preprocessing, reflecting concurrent shared-filesystem activity visible on the machine. All eight workers were active and available host memory remained ample, so changing parallelism was not justified. The total projection is revised from 15 to 25 minutes, still inside the 50-minute execution budget before the ten-minute reserve.

## Completion

Full preprocessing took 3:13 and verification took 0:13. The 43-case GPU pass took 3:36, and evaluation took 1:39. All timed commands exited 0; LoadGen recorded 43 effective accuracy samples and the evaluator printed `mean = 0.86167, kidney = 0.9347, tumor = 0.7887`. The run completed within both the revised projection and reserve.
