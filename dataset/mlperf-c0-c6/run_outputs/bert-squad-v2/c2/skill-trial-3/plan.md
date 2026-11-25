# Reproduction plan

## Workload

One target, `bert-squad`: build a Python 3.11 environment and the repository's
LoadGen binding; generate fixed BERT/SQuAD feature records; run BERT-Large fp32
on the local A100 in Offline AccuracyOnly mode; run the repository evaluator;
retain all requested LoadGen logs and evaluator stdout in `bert-squad/out`.

## Measured costs

- Clean setup preflight: 52.3 seconds on the 8 available CPU cores, 527 MiB
  peak host RSS; no GPU work.
- Bounded preprocessing: 0.79 seconds for 2 examples / 2 features, 57 MiB
  peak host RSS; no GPU work.
- Bounded end-to-end accelerator run and evaluation: 26.8 seconds for two
  examples, 2.15 GiB peak host RSS. Model construction dominates this small
  measurement.
- GPU free memory immediately before the bounded run: 73,756 MiB of 81,920
  MiB. Other sessions were using the shared GPU, so memory is checked again
  immediately before the full run.
- Host memory: about 1.9 TiB available.
- Provenance-verified prior trial on the same arm: 10,570 examples / 10,833
  features, 10 seconds preprocessing, and roughly 10--12 minutes total GPU
  inference and evaluation.

## Schedule and estimate

Run setup, preprocessing, inference, and evaluation serially for the sole
target through the delivered script's `--from setup` path. Setup and
preprocessing are CPU-only; inference is the only GPU stage. No other target
has CPU work to overlap. The prior same-arm full measurement is the strongest
rate estimate and projects completion within 12 minutes after setup. Allow 15
minutes for clean setup and preprocessing and hold a 20-minute reserve for a
repair, artifact validation, and `report.json`; the projection is comfortably
inside the two-hour limit.

## Re-plan triggers

After the first few percent of the full-run accuracy log is available, compare
its growth and GPU state with the 12-minute inference projection. A failure
returns through `reproduce.sh --from` at the failed stage. When the reserve is
reached, start no new run and report only measured values.

## Re-plan checkpoint

During the first minute of inference, the accuracy log grew from 507,899 to
2,267,248 bytes in 39 seconds while the shared A100 remained at 100%
utilization with several independent processes. After competing jobs ended it
reached 20,018,085 bytes and was growing around 130 KB/s. Each of the 10,833
records contains 384 by 2 float32 logits encoded as hex, so the corrected final
log-size estimate is about 67 MB; the observed rate still projects completion
inside the 12-minute inference estimate. No schedule change is needed.

## Outcome

The clean full run completed successfully in 13 minutes 54 seconds including
environment setup, preprocessing, shared-GPU inference, and evaluation. It
used at most 2.20 GiB host RSS. LoadGen wrote 10,833 accuracy records covering
QSL indices 0 through 10,832 exactly once and reported no warnings or errors.
The repository evaluator printed exact match 83.68968779564806 and F1
90.87487229720105. All required outputs, logs, stage success markers, launcher
syntax, and report structure passed final validation.
