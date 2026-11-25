# Reproduction plan

## Workload

One target, `bert-squad`, runs four ordered stages: build a Python 3.11
environment and local LoadGen wheel; copy the benchmark and generate SQuAD
feature records; run PyTorch BERT-Large fp32 in LoadGen Offline AccuracyOnly
mode; run the repository's `accuracy-squad.py` evaluator. Each stage writes a
success marker and the later stages consume the earlier stages' files.

## Measured costs

- Clean cached environment installation plus local LoadGen compilation took
  about 50 seconds in preflight.
- Preprocessing read 10,570 questions and produced 10,833 feature windows of
  384 tokens in about 25 seconds.
- A delivered-script pass over 256 windows, including model load, LoadGen,
  postprocessing, and evaluation, took 26.89 seconds wall time and 2,411,852
  KiB peak host RSS. It emitted 256 accuracy records and had no LoadGen
  warnings or errors.
- The GPU check immediately before that pass reported 81,038 MiB free on the
  NVIDIA A100-SXM4-80GB.

## Resources and schedule

This session exposes 8 logical CPUs, about 2.0 TiB host memory, and one A100
80GB. Only one target exists, so its stages run serially; no second GPU job is
launched. The full run will start from an empty `bert-squad/work`, recheck GPU
free memory immediately before inference, and use the default complete input
scope. There is no ready independent target whose CPU setup can usefully
overlap the GPU stage.

## Expected time and reserve

Conservatively scaling the complete 256-window wall time without subtracting
fixed startup/evaluator overhead projects 1,138 seconds (19.0 minutes) for
10,833 windows. Including a two-minute clean build/preprocess allowance gives
an approximately 21-minute run. A 15-minute reserve is held for evaluator
inspection, `report.json`, artifact verification, and one bounded repair if
needed. This fits within the two-hour limit with more than 80 minutes of
margin.

## Re-plan during full run

At 72 seconds into full inference, the process was healthy (about 97% of one
CPU core, CUDA allocation present) and the GPU was at 100% utilization. Two
unrelated jobs were sharing the GPU, increasing total device use to about 7.4
GiB. Even a several-fold slowdown from the 19-minute conservative projection
fits the 80-minute margin. With one serial target there is no safe parallelism
change to make, so the schedule and reserve remain unchanged.

Contention later reached six simultaneous GPU processes. Full inference
completed without warnings or errors after about 21 minutes, when contention
had fallen to four processes. The evaluator then completed in under a minute,
so no reserve-consuming repair was needed.
