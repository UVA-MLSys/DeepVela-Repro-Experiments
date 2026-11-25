# BERT SQuAD reproduction plan

## Workload

One target, `bert-squad`, runs in three stages: build an isolated Python 3.11
environment and compile the pinned repository's LoadGen; stage the supplied
SQuAD v1.1 data, BERT-Large fp32 checkpoint and vocabulary and generate the
10,833 input feature windows; run Offline AccuracyOnly inference on the local
A100 and invoke `accuracy-squad.py` on the resulting LoadGen accuracy log.

## Preflight measurements

- Environment resolution, installation, and LoadGen compilation: about 45 s.
- Full CPU preprocessing: about 17 s for 10,570 questions / 10,833 windows.
- Bounded GPU run and evaluator: about 30 s for model load, 8 windows, and
  scoring. The model used `cuda:0`; 71,918 MiB of 81,920 MiB was free before
  launch, and the target-local 12,000 MiB safety threshold passed.
- The bounded evaluator completed and LoadGen reported no warnings or errors.

## Resources and schedule

The script uses this machine directly, one GPU run at a time. Environment and
preprocessing run first on CPU; there is only one target, so no independent CPU
stage remains to overlap with its GPU stage. Immediately before the accelerator
stage, the script records `nvidia-smi` output and requires at least 12,000 MiB
free. The full validation inference is then followed by the CPU evaluator.

The 8-window timing is dominated by checkpoint/model startup. Conservatively
allowing 30 ms per window projects about 5.5 minutes of inference, plus under
2 minutes for clean environment rebuild, preprocessing, and evaluation. The
expected total is under 8 minutes. A 15-minute reserve is retained for artifact
verification, reporting, and one repair, well inside the 2-hour limit.

## Re-plan during full run

The first live interval produced 359 accuracy-log records while several other
sessions were saturating the shared GPU. That contention-adjusted rate projects
about 25 minutes for full inference. The target remains well inside the limit
and reserve; with one target and the required serial GPU policy, the schedule
does not change.
