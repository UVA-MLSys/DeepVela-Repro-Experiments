# Full-run plan

## Workload

One target, `bert-squad`, with four script stages: `source` stages a writable copy and immutable
input links; `env` creates the pinned offline Python environment and builds checked-in LoadGen;
`preprocess` creates all 10,833 model features for the 10,570 validation questions; `run` checks
GPU memory, performs the full Offline AccuracyOnly run, and invokes the benchmark evaluator.

## Measured costs

- Current source staging completed at 12:52:20.
- Environment setup completed 73.8 seconds later at 12:53:33; LoadGen compilation itself took
  28.93 seconds. The environment occupies 5.3 GiB.
- Full-split preprocessing completed in 53.9 seconds and wrote a 51 MiB feature cache.
- A 16-example end-to-end preflight completed in 27.24 seconds, used 2,395,644 KiB peak host RSS,
  identified `cuda:0`, and produced valid LoadGen and evaluator outputs.
- Trial-1 clean full-run artifacts show the GPU check at 14:25:32 and evaluator completion at
  14:56:23, approximately 30m52s. This is a better full-scope estimate than extrapolating the
  load-dominated 16-example run.

## Resources and schedule

The machine exposes 8 CPU cores, about 2 TiB host RAM, and one shared NVIDIA A100-SXM4-80GB.
The preflight's mandatory immediate check found 71,882 MiB GPU memory free. The script will check
again before the full GPU run and require at least 8,192 MiB. There is only one target, so GPU
parallelism is one; no independent ready CPU target remains to overlap with it.

Run the delivered script unchanged as
`OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from source`. This clears its own
stage markers, rebuilds from the repository and cache, regenerates preprocessing, then performs
the full accelerator run and evaluation. Expected total is under 35 minutes: roughly 2.5 minutes
for clean source/environment/preprocessing plus about 31 minutes for run/evaluation.

## Reserve and acceptance

Reserve 10 minutes before the two-hour limit to parse the full evaluator output into
`report.json`, verify the three declared files, required device log prefix, AccuracyOnly detail
record, success markers, and clean-script portability. If measured progress projects beyond the
reserve, record the trigger and revised projection here before changing the schedule.
