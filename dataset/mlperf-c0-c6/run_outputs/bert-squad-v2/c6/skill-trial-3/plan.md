# BERT-SQuAD reproduction plan

## Workload

The single `bert-squad` target has four ordered stages: build an isolated Python
environment and repository LoadGen; validate and stage the fixed read-only
SQuAD, model, vocabulary, and DeepLearningExamples inputs; run the PyTorch fp32
Offline accuracy pass; and invoke the repository's `accuracy-squad.py`
evaluator. Each stage leaves a success marker and each later stage consumes the
earlier stage's files.

## Measured costs and resources

- The two-feature end-to-end preflight used the delivered `reproduce.sh`.
- Environment creation and local LoadGen compilation took about 40 seconds.
- Input validation and staging took about 2 seconds.
- Model loading, full validation preprocessing, and bounded inference took
  about 69 seconds. Preprocessing produced the repository's full feature cache
  (10,833 features for 10,570 questions), so every input sub-workload was
  represented before the full run.
- The bounded evaluator took about 8 seconds and printed JSON successfully.
- Immediately before preflight inference, the shared A100 80GB had 72,486 MiB
  free. The next availability check showed 76,967 MiB free but 100% utilization
  from unrelated work. GPU memory is therefore not limiting; shared compute
  contention is the principal uncertainty.
- Read-only prior trial evidence records a 33.81-second warm-cache 100-feature
  run under contention and a 627.10-second complete clean run with the same
  target, inputs, and runner design.

## Schedule and expected time

Launch the target once with `reproduce.sh --from environment`, which clears the
runner's own preflight outputs, rebuilds from nothing, and executes all four
stages in order. Only this target uses the GPU, so no local GPU work overlaps.
There is no second target with CPU preparation to overlap.

The conservative contention projection is 10,833 / 100 * 33.81 seconds = 61.0
minutes, plus about 3 minutes for environment setup, preprocessing, and
evaluation, for a planned completion of about 64 minutes. The prior complete
run suggests the likely duration is closer to 11 minutes. Hold 15 minutes of
the two-hour limit for output verification, evaluator-only repair if required,
and `report.json`.

## Re-plan triggers

After the first few percent of inference, compare accuracy-log growth against
the 64-minute projection. If the projection exceeds the 105-minute launch
budget, preserve successful stage outputs, record the trigger here, and adjust
only scheduling or validated resource parameters before any replacement run.
No new work starts once the 15-minute reserve is reached.

## Re-plan: first full-run progress sample

The full feature cache was written at 16:44:40 and the accuracy log reached
1.02 MB by 16:44:49. Relative to the approximately 67 MB full-log size from
the prior same-arm measurement, this projects about 10 minutes of inference.
Two BERT processes were sharing the GPU and utilization was 100%, but memory
remained ample at 77,359 MiB free. The new projection is well inside the
64-minute conservative plan, so the stage order and single local GPU run remain
unchanged and the 15-minute reporting reserve is retained.

## Completion

The clean delivered-script run completed successfully within the revised
projection. LoadGen emitted 10,833 accuracy records with 10,833 unique QSL
indices spanning 0 through 10,832 and reported no warnings or errors. The
repository evaluator completed over the full validation set and printed
`exact_match` 83.68968779564806 and `f1` 90.87487229720105.
