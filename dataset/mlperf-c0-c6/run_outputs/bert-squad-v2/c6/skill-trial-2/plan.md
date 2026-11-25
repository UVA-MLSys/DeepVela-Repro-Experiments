# BERT SQuAD reproduction plan

## Workload

Target `bert-squad` has three ordered stages: copy the pinned benchmark and staged NVIDIA BERT
support code and build a uv environment plus LoadGen; generate the complete SQuAD v1.1 feature
cache on CPU; then run PyTorch fp32 AccuracyOnly in Offline mode on the local A100 and invoke
`accuracy-squad.py` / `evaluate_v1.1.py`.

The fixed inputs are the supplied 10,570-question SQuAD v1.1 file, BERT-Large
`model.pytorch`, `vocab.txt`, and the separately staged DeepLearningExamples tree. Preprocessing
produced 10,833 features.

## Measured costs and resources

- Clean environment setup: 1m01.90s elapsed, 0.50 GiB peak host RSS.
- Complete CPU preprocessing: 57.94s elapsed, 0.85 GiB peak host RSS.
- Bounded 8-feature GPU run plus official evaluator: 33.36s elapsed, 2.26 GiB peak host RSS.
- The GPU check immediately before the bounded run reported 76,985 MiB free of 81,920 MiB.
- The same-arm prior snapshot records a full inference observation of roughly 4–6 minutes and
  5,988 MiB GPU use. This is the best full-scope rate estimate because most bounded-run time is
  fixed checkpoint/import/evaluator cost.
- Available compute is one shared local NVIDIA A100-SXM4-80GB. GPU stages run one at a time;
  setup, preprocessing, and evaluation are CPU work.

## Schedule and expected time

There is one target, so no cross-target overlap is useful. Run the delivered no-stage script from
nothing built for the full target: setup, complete preprocessing, one full GPU run, and evaluator.
The current measured CPU stages total about two minutes. Combining those measurements with the
prior full inference observation projects the complete run below 10 minutes.

Reserve 15 minutes before the two-hour limit for a repair/rerun if needed, `report.json`, required
artifact checks, and copied-alone script validation. No new GPU work begins in that reserve.

## Re-plan trigger

Once the incrementally written accuracy log shows the first few percent of full inference, compare
its growth with the sub-10-minute projection. Re-plan only if the projection approaches the
15-minute reserve or a stage fails.

## Re-plan during full run

After about 1m33s of full inference, the incrementally written accuracy log was 10.08 MB. Scaling
the bounded run's 49,523-byte log for 8 features gives roughly 67 MB for all 10,833 features, so
the observed rate projected about 10--11 minutes of inference under current shared-GPU contention.
This is slightly slower than the initial complete-run projection but remains far inside the
two-hour limit and 15-minute reserve. With one serial GPU target, the schedule is unchanged.

## Clean-copy repair

Validation after the first full run found that installing LoadGen directly from `repo/loadgen`
left build metadata in the source tree. The wrapper was repaired to copy LoadGen into the target's
isolated work directory before building. A setup preflight confirmed the isolated build path, the
source tree was restored to source-only state, and the repaired script was then run unchanged from
a new `bert-squad` directory. This repair changes no benchmark semantics.
