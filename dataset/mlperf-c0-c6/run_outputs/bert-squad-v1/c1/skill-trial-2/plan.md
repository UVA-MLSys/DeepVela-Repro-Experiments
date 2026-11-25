# BERT SQuAD reproduction plan

## Workload

Target `bert-squad` runs three stages in order: a Python 3.11 environment and
repository LoadGen build, SQuAD v1.1 feature preprocessing, and one Offline
PyTorch FP32 accuracy run followed by the repository evaluator. The immutable
inputs are `dev-v1.1.json` (10,570 questions), `model.pytorch`, `vocab.txt`, and
the staged NVIDIA BERT tokenization source. Preprocessing produces 10,833
feature samples.

## Measured costs

- Clean environment and LoadGen build: 58 seconds.
- Full-dataset CPU preprocessing: 61 seconds.
- 8-sample GPU run plus evaluation: 34 seconds.
- 256-sample GPU run plus evaluation: 47 seconds. The benchmark portion was
  40 seconds and evaluation was 7 seconds.
- The difference between the bounded benchmark runs is 248 samples in about
  14 seconds, approximately 17.7 samples/second after fixed startup.
- GPU checks immediately before the bounded runs found 79,323 MiB and 77,377
  MiB free. The machine exposes one A100 80GB, 8 CPU cores, and about 1.9 TiB
  currently available host memory.

## Schedule and projection

Run `./reproduce.sh --target bert-squad --from setup` with no sample limit so
the first full run starts from nothing built and exercises the delivered script.
The CPU build and preprocessing run first; the single GPU stage then runs alone,
and evaluation follows it on CPU. No second target exists to overlap.

At the measured steady rate, 10,833 samples require about 10.2 minutes. Adding
the measured clean build, preprocessing, model startup, and a conservative full
evaluation allowance gives a 15-minute projection. Hold 10 minutes after the
run for artifact validation and `report.json`; this is comfortably within the
two-hour wall-clock limit.

## Re-plan during full run

At 17:51:38, the full run had written 1,005 of 10,833 accuracy records. The GPU
was at 100% utilization with 72,026 MiB free. After allowing for model startup,
the observed rate was about 12--13 samples/second, below the bounded-run
estimate and consistent with roughly 14--15 minutes of inference. The revised
total projection is 20 minutes. No schedule change is needed; the run remains
well inside the limit and the 10-minute reporting reserve remains intact.

## Completion

The clean invocation ran from 17:48:19 through 18:09:14, 20 minutes 55 seconds,
consistent with the revised projection. It produced 10,833 unique accuracy
records, and LoadGen reported no warnings or errors. The official evaluator
completed within the reporting reserve.
