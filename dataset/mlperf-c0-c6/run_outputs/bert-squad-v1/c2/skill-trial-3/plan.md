# Reproduction plan

## Workload

One target, `bert-squad`: build a Python 3.11 environment and the repository's
LoadGen binding; generate fixed BERT/SQuAD feature records; run BERT-Large fp32
on the local A100 in Offline AccuracyOnly mode; run the repository evaluator;
retain all requested LoadGen logs and evaluator stdout in `bert-squad/out`.

## Measured costs

- Clean setup: about 49 seconds, 8 CPU cores available; no GPU work.
- Full preprocessing: 10 seconds for 10,570 examples / 10,833 features; no GPU work.
- Bounded end-to-end run: about 16 seconds for model construction, 20 features,
  LoadGen, and evaluation. Model construction dominates this small measurement.
- GPU free memory immediately before both bounded runs: 79,197 MiB of 81,920 MiB.
- Host memory: about 1.9 TiB available.

## Schedule and estimate

Run the stages serially for this sole target. Setup and preprocessing are CPU
only; inference is the only GPU stage. The harness issues one feature at a time,
so a conservative 50 ms/feature estimate gives about 9 minutes for inference.
Allow 2 more minutes for model loading and evaluation: expected total under 12
minutes. Hold a 15-minute reserve for repair, artifact validation, and
`report.json`, comfortably inside the two-hour limit.

## Re-plan triggers

After the first full-run log becomes available, compare its completion time to
the 12-minute projection. Any failure returns to the failed stage through
`reproduce.sh --from`; no unrelated successful target exists to reschedule.

## Re-plan checkpoint

At 86 seconds of the GPU process, the accuracy log was 9,466,337 bytes while
the shared A100 was at 100% utilization with two independent BERT processes.
The observed growth projects roughly 10--12 minutes for inference, within the
original estimate. No schedule change is needed; continue the sole serial run.

## Outcome

The full run completed successfully. LoadGen wrote 10,833 accuracy records,
covering QSL indices 0 through 10,832 exactly once, and reported no warnings or
errors. The repository evaluator printed exact match 83.68968779564806 and F1
90.87487229720105. Required outputs and the delivered script passed the final
artifact and syntax checks.
