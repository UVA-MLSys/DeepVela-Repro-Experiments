# BERT-SQuAD reproduction plan

## Workload

The single `bert-squad` target runs three ordered stages: create the pinned
Python/LoadGen environment, stage and validate the fixed read-only inputs, then
run the repository's PyTorch FP32 Offline accuracy path over all SQuAD v1.1
features and invoke `accuracy-squad.py`. The delivered no-stage invocation runs
all three stages and writes the run marker only after the evaluator succeeds.

## Measured costs

- Current clean setup: 43.50 seconds and 527,488 KiB maximum host RSS.
- Current input staging: 1.40 seconds and 35,736 KiB maximum host RSS.
- Current clean eight-feature end-to-end GPU/evaluator check: 87.07 seconds and
  2,253,900 KiB maximum host RSS. It passed with CUDA on the A100 and printed
  the required `pytorch device: cuda:0` line.
- Prior same-arm cached 100-feature check (verified by provenance hash): 34.11
  seconds end to end, 2,236,640 KiB maximum host RSS, and a 9.949-second
  LoadGen interval (10.05 features/s).
- The full QSL contains 10,833 features. Immediately before the current bounded
  run, 73,324 MiB of the A100's 81,920 MiB was free.

## Resources and schedule

Run directly on the one local NVIDIA A100 80GB, one owned GPU process at a
time. The wrapper checks GPU free memory immediately before every GPU launch
and requires at least 12,000 MiB. There is one target, so there is no other
target build to overlap with its GPU stage.

At the measured 100-feature steady-state rate, 10,833 features require about
1,078 seconds (18.0 minutes). Setup, preprocessing/model load, and evaluation
bring the projection to roughly 20 minutes. Hold 10 minutes in reserve for
artifact validation and `report.json`. Launch the clean full run with
`--from setup`; it removes only wrapper-owned state and exercises the delivered
script from nothing built.

## Execution update

At the first measurable fraction, the accuracy log contained 946 of 10,833
records after about 121 seconds of timed inference (7.8 features/s) while the
GPU was shared with several other jobs. This projects the full inference pass
to about 23 minutes, slower than the prior 18-minute estimate but still well
inside the two-hour limit and 10-minute reserve. The one-run-at-a-time schedule
therefore does not change.

The clean full invocation completed successfully in 982.24 seconds (16:22.24)
with 2,261,724 KiB maximum host RSS, ahead of both projections.
