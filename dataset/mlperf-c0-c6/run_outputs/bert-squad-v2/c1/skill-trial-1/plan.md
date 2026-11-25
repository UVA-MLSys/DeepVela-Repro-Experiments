# BERT-SQuAD reproduction plan

## Workload

One target, `bert-squad`: create an isolated Python 3.12 environment and build
the pinned local LoadGen extension; preprocess the complete SQuAD v1.1
validation set into features; run BERT-Large fp32 with the PyTorch backend in
Offline AccuracyOnly mode on GPU 0; evaluate the complete LoadGen accuracy log
with the repository's `accuracy-squad.py` and `evaluate_v1.1.py`.

The fixed input has 48 articles, 2,067 paragraphs, and 10,570 questions. Its
SHA-256 is `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`;
the checkpoint and vocabulary SHA-256 values are respectively
`71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f`
and `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`.

## Measured costs

The clean-copy bounded preflight completed every stage and the official
evaluator. Environment creation and LoadGen build took about 47 seconds on CPU;
full-split preprocessing took 55 seconds and produced 10,833 features; model
startup plus an eight-feature GPU run took 19 seconds; evaluation took 8
seconds. Immediately before that GPU run, 79,215 MiB was free on the local A100.
The evaluator reported `exact_match=75.0` and `f1=75.0` on its bounded input.

The hash-verified prior trial of the same runner and inputs measured about 11.1
features/s in preflight and projected the full 10,833-feature pass at 16.3
minutes. Its complete projection, including clean setup and evaluation, was
about 20 minutes.

## Resources and schedule

Run the clean-copy stages in dependency order: environment, prepare, GPU run,
evaluation. There is one target and one accelerator stage, so GPU work remains
serial. The script checks live free GPU memory immediately before inference and
requires at least 8,192 MiB; the final pre-launch observation was 76,908 MiB
free of 81,920 MiB. CPU build and preprocessing precede inference, and the
official evaluator follows the completed accuracy log.

Expected wall time is about 20 minutes. Reserve 15 minutes of the two-hour
limit for artifact validation, `report.json`, and a bounded repair if needed.

## Re-plan triggers

After the first few percent of the accuracy log is written, compare its growth
and GPU state with the 11.1 features/s prior measurement. If projected total
time exceeds 105 minutes, preserve measured outputs rather than begin a second
full run. Any failure returns to the earliest failed stage via
`reproduce.sh --from`.

## Re-plan 1: first few percent

At the first live checkpoint, roughly 20--30 seconds into inference, the
accuracy log was 2,861,968 bytes, about 4% of the size projected from the
eight-feature bounded log. GPU utilization was 100% and 75,576 MiB remained
free. This is consistent with or ahead of the measured projection, so the
schedule and approximately 20-minute end-to-end estimate are unchanged.

## Completion measurement

The clean full run completed successfully. Environment/build finished at
12:19:49, preprocessing at 12:20:43, the full GPU pass at 12:41:56, and the
official evaluation at 12:42:29 local time. Thus preprocessing took about 55
seconds, the full inference stage about 21 minutes 13 seconds, and evaluation
about 33 seconds. The accuracy log contains all 10,833 unique QSL indices, and
LoadGen reported no warnings or errors.
