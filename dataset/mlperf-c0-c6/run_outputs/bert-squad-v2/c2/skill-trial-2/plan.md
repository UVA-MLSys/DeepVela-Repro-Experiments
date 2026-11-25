# BERT SQuAD reproduction plan

## Workload

The single target, `bert-squad`, has three ordered stages. The environment
stage creates an isolated Python 3.11 environment and compiles this repository's
LoadGen. The preparation stage stages the supplied SQuAD v1.1 data,
BERT-Large fp32 checkpoint and vocabulary, then generates all 10,833 feature
windows for 10,570 validation questions. The GPU stage runs the documented
`python3 run.py --backend=pytorch --accuracy` route in the default Offline
scenario and invokes `accuracy-squad.py`, which calls `evaluate_v1.1.py`.

## Measured costs

- Clean environment resolution, installation, LoadGen compilation and import
  check: 48.02 s, 526,948 KiB maximum process RSS.
- Full CPU preprocessing: 18.98 s, 521,396 KiB maximum process RSS.
- Bounded model load, eight-window GPU run and evaluator: 22.74 s, 1,852,692
  KiB maximum host process RSS. The run used `cuda:0`; the A100 had 79,215 MiB
  free immediately before launch, and the script's 12,000 MiB threshold passed.
- The bounded evaluator completed with exact match and F1 output, while LoadGen
  reported no warnings and no errors.

## Resources and schedule

This machine exposes 8 CPU cores, about 2 TiB of host memory, and one shared
NVIDIA A100-SXM4-80GB. The delivered script runs directly on the machine and
queries free GPU memory immediately before every accelerator run. It requires
at least 12,000 MiB free. GPU work from this reproduction is serialized.

The clean environment and preparation stages run first. There is one target,
so no independent target's CPU stage remains to overlap with its accelerator
stage. The full GPU accuracy pass follows preprocessing, and the CPU evaluator
runs as part of the same final stage.

The eight-window sample is dominated by model startup and does not provide a
useful linear throughput estimate. The provenance-verified prior run of this
same script and target observed a contention-adjusted projection of about 25
minutes for the complete 10,833-window inference. Adding under 2 minutes for a
clean rebuild, preprocessing and evaluation gives a conservative expected
total below 30 minutes. A 15-minute reserve is retained for artifact checks,
`report.json`, and one repair, within the two-hour wall-clock limit.

## Re-plan triggers

During the full run, the accuracy-log record count and elapsed time will be
checked after the first few percent. If projected completion exceeds the
remaining time less the 15-minute reserve, the schedule and projection will be
updated here. Failures will be repaired through `reproduce.sh --from <stage>`;
successful stages will not be repeated independently.

## Re-plan during the full run

At 1,593 of 10,833 accuracy records (14.7%), 87 seconds had elapsed since
the GPU-stage memory check. The observed average of 18.31 records/s, including
model startup, projected 9.86 minutes for the complete GPU stage and about 8.4
minutes remaining. GPU utilization was 100% with 74,432 MiB free. This is well
inside the conservative 25-minute projection and the 15-minute reserve, so the
single-target serial schedule does not change.

The full clean run completed successfully in 816.19 seconds (13.60 minutes),
including environment rebuild, preprocessing, inference and evaluation. The
final accuracy log contains all 10,833 unique QSL indexes.
