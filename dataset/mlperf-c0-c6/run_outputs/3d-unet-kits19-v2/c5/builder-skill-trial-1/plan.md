# Reproduction plan

Updated: 2026-11-19 14:16 EST

## Workload

One target, `3d-unet`, with four ordered stages: build a pinned environment and add the missing compatibility SUT in the isolated work copy; preprocess and verify all 43 manifest entries; run FP32 PyTorch/Offline/AccuracyOnly inference through LoadGen; run `accuracy_kits.py` and materialize the three required outputs.

## Preflight and measured costs

The task explicitly permits reuse of `prior-artifacts`. Its delivered `reproduce.sh` and this run's script are byte-identical (SHA-256 `f90dafc334f75a0a1d6ca89ef5bc04307168215b90f7b92b0aeb1b514b1564b7`). Its bounded preflight reduced the manifest and all count/config checks consistently to `case_00000`, then completed LoadGen and the evaluator. That case took 8.54 seconds through LoadGen and produced kidney Dice 0.9700025991, tumor Dice 0.8403025546. Environment installation took about 13 seconds plus about 60 seconds for the first CUDA import. The representative case preprocessed in about 5 seconds and matched its published checksum. One-case evaluation took about 6 seconds.

The prior full run of the same source, inputs, host class, and identical script verified all preprocessing checksums and completed at mean 0.86167, kidney 0.9347, tumor 0.7887. It supplies scheduling evidence only; this run will independently create all scored outputs.

## Resources and schedule

Current A100 availability before launch is 81,037 MiB free of 81,920 MiB. Build and eight-way preprocessing are CPU-only. The script rechecks GPU memory immediately before inference and requires 12,000 MiB. There is one target, so there is no independent ready CPU target to overlap with the GPU stage.

Allow 5 minutes for build/preprocessing, 5–15 minutes for the 43 variable-size volumes, and 1–3 minutes for evaluation: projected total 11–23 minutes, with a 25-minute working allowance.

## Reserve and re-plan triggers

Reserve 10 minutes before the one-hour limit for `report.json`, output validation, and deliverable checks. If inference reaches 25 minutes, recalculate from processed samples; at the reserve boundary, start no new work and report only measured values. Record any failure-driven repair and new projection here before resuming with `--from <stage>`.

## Completion

The clean full run started at 14:14:51 EST and completed within the 11–23 minute projection without a repair or re-plan. All 43 accuracy records were evaluated. LoadGen reported no errors, and the final measured values were mean 0.86167, kidney 0.9347, and tumor 0.7887.
