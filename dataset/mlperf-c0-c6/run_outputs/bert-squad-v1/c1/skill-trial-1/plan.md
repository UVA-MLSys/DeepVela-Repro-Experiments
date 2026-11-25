# BERT-SQuAD reproduction plan

## Workload

One target, `bert-squad`: build an isolated Python 3.12 environment and local
LoadGen extension; preprocess SQuAD v1.1 into 10,833 sequence features; run
BERT-Large fp32 in Offline AccuracyOnly mode on GPU 0; evaluate the full
LoadGen accuracy log with `accuracy-squad.py` / `evaluate_v1.1.py`.

The fixed inputs contain 48 articles, 2,067 paragraphs, and 10,570 questions.
Their SHA-256 values are recorded in the session evidence; the source files
remain read-only.

## Measured costs

- Clean environment plus local LoadGen build: about 60 seconds, CPU and shared
  package cache; no GPU.
- Full-split feature preprocessing: about 57 seconds, CPU; 10,833 features and
  a 53,468,824-byte cache.
- Bounded GPU stage: about 23 seconds including model startup and teardown.
  LoadGen itself processed 8 samples in 0.721 seconds, about 11.1 samples/s.
- Bounded official evaluation: about 8 seconds.
- GPU availability immediately before preflight: 77,395 MiB free of 81,920
  MiB. The runner requires at least 8,192 MiB free before starting.

## Resources and schedule

Run the clean-copy stages in dependency order: environment, prepare, GPU run,
evaluation. There is only one target and one accelerator stage, so no GPU
parallelism is useful or allowed. The CPU build and preprocessing are short and
must precede this target's GPU stage; evaluation follows the completed accuracy
log. The runner checks current free GPU memory immediately before inference.

The 8-sample LoadGen rate projects 10,833 samples to about 16.3 minutes.
Including clean setup, preprocessing, model startup, output writing, and full
evaluation, the expected wall time is about 20 minutes. Hold 15 minutes of the
two-hour budget for artifact checks, `report.json`, and one bounded repair if
needed.

## Re-plan triggers

After roughly the first few percent of the full accuracy run, compare observed
progress with the 11.1 samples/s preflight rate. If projected completion exceeds
105 minutes total, do not start another full run; preserve measured artifacts
and report the evidence. A run/evaluator failure returns to the earliest failed
stage using `reproduce.sh --from`.

## Re-plan 1: first few percent

At about 22 seconds into the full LoadGen interval, the live accuracy log was
2,000,866 bytes, approximately 3% of the projected final size from the bounded
run's bytes per sample. GPU utilization was 100%, with 75,576 MiB free. This is
consistent with or slightly ahead of the measured 11.1 samples/s rate, so the
schedule and roughly 20-minute end-to-end projection are unchanged.
