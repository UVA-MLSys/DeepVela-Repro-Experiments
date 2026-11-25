# Reproduction plan

## Workload

The sole target, `bert-squad`, has four ordered stages: build a Python 3.11
environment and the repository's LoadGen wheel; copy the BERT benchmark and
generate SQuAD feature records; run BERT-Large fp32 with PyTorch in LoadGen
Offline AccuracyOnly mode; and invoke the repository's `accuracy-squad.py`
evaluator. Each successful stage writes a marker, while `--from` clears that
stage and all dependent outputs before rerunning them.

## Measured costs

- The clean cached environment install and local LoadGen build completed in
  approximately 50 seconds.
- Full preprocessing read 10,570 validation questions and generated 10,833
  feature windows of 384 tokens in 22.94 seconds wall time, with 876,876 KiB
  peak host RSS.
- The delivered script's 256-window preflight, including model load, inference,
  LoadGen logging, postprocessing, and evaluation, completed in 49.79 seconds
  wall time with 2,424,728 KiB peak host RSS. It had no LoadGen warnings or
  errors and emitted the required device line.
- The GPU checks immediately before the preflight reported 73,756 MiB free on
  the NVIDIA A100-SXM4-80GB. The device was already at 100% utilization from
  other sessions, so the measured rate includes current contention.

## Resources and schedule

There is one local A100 80GB and one target. The stages therefore run serially,
with no concurrent GPU run. The full invocation starts with `--from build`,
which removes all script-owned preflight state and outputs, then executes the
same delivered script through evaluation. The run stage checks free GPU memory
again immediately before loading the model and requires at least 12,000 MiB.
No independent target exists whose CPU preparation could overlap inference.

## Expected time and reserve

Scaling the complete 256-window preflight without subtracting fixed startup and
evaluator overhead gives a deliberately conservative 2,107-second (35.1-minute)
projection for all 10,833 windows. Adding two minutes for a clean build and
preprocessing yields about 37 minutes. A 20-minute reserve is held for output
inspection, `report.json`, and one bounded repair, leaving roughly 63 minutes of
additional margin within the two-hour wall-clock limit.

## Re-plan triggers

After the first few minutes of inference, process health and the measured rate
will be checked. A projection beyond the 100-minute execution budget will
trigger a schedule review; because there is only one target and GPU work must be
serial, no additional GPU parallelism is planned. Failures will return to the
earliest affected stage with `--from`, preserving successful earlier stages.

## Full-run re-plan

At 66 seconds into full inference, the benchmark process was healthy at 98.6%
of one CPU core and held 1,814 MiB of GPU memory. The A100 remained at 100%
utilization with one unrelated compute process also present. This is consistent
with the contention measured during preflight and leaves the approximately
37-minute full-run projection well inside the 100-minute execution budget.
With one serial target there is no useful parallelism adjustment, so the
schedule and 20-minute reserve remain unchanged.

At 215 seconds, the live accuracy log contained 4,076 of 10,833 records
(37.6%). The observed full-run rate projects about 9.5 minutes for inference,
or roughly 11 minutes including the completed clean setup. This improves the
projection substantially; the serial schedule remains appropriate.

At 636 seconds, the log contained 9,772 records (90.2%). Four unrelated GPU
processes had joined the benchmark, reducing the recent rate. Only 1,061
records remained, and even the contended rate projected completion within a
few additional minutes, so no repair or schedule change was warranted.

The full inference and evaluator subsequently completed without LoadGen
warnings or errors. The added GPU contention did not consume the reserve, and
no repair run was needed.
