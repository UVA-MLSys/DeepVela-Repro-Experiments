# Reproduction plan

## Workload

Target `bert-squad` has three stages: create a Python 3.11 environment and
build this source tree's LoadGen; stage the fixed SQuAD v1.1, BERT-Large fp32,
and DeepLearningExamples inputs and preprocess the full validation set; then
run `run.py --backend=pytorch --accuracy` in its default Offline scenario and
invoke the repository's `accuracy-squad.py`. Each stage writes a success marker
only after all of its commands complete.

## Measured costs

- Environment/build: 53 seconds, CPU-only, with a 5.2 GiB target-local venv.
- Prepare: 60 seconds, CPU-only, producing 10,833 features in a 51 MiB cache
  from 10,570 SQuAD v1.1 questions.
- Bounded run and evaluation: 30 seconds for checkpoint load, eight samples,
  and evaluator. LoadGen recorded Offline / AccuracyOnly and no warnings or
  errors. The GPU had 76,125 MiB free immediately before this run.
- The read-only prior trial for this same arm reports that its full GPU pass
  took about seven minutes under contention. Allowing startup, evaluation, and
  filesystem variance, the conservative full target projection is 12 minutes.

## Resources and schedule

Run the one target directly on local GPU 0. The environment and preprocessing
stages run first on local CPUs. The script records `nvidia-smi` immediately
before its accelerator stage and refuses to start below 8 GiB free. There are
no other targets or ready CPU stages to overlap, and GPU concurrency remains
one as required.

## Expected time and reserve

The projected clean run is at most 14 minutes end to end: about two minutes of
CPU setup plus a conservative 12 minutes for GPU work and evaluation. Hold 15
minutes in reserve to extract the evaluator's exact printed values, write
`report.json`, validate every required artifact, and inspect clean-copy
portability. This fits well inside the two-hour wall-clock limit.

## Re-planning triggers

After the accuracy log begins growing, compare its rate with the 12-minute GPU
projection. If failures occur, repair through `reproduce.sh --from <stage>` and
record the changed projection here. Start no new work when the 15-minute reserve
is reached.

## Re-plan during full run

At about 62 seconds after inference output began, the accuracy log contained
716 of 10,833 samples (about 11.5 samples/second). The shared A100 reported 100%
utilization and three other compute processes alongside this run. This projects
about 16 minutes of inference, so the GPU allowance is revised from 12 to 18
minutes. Target order and concurrency do not change: this is the only target
and the only GPU workload started by this session, and the new projection still
leaves substantially more than the 15-minute reporting reserve.

At 2,951 samples, increased shared-GPU contention reduced the sustained rate to
about 9.1 samples/second, projecting roughly 20 minutes of inference. The GPU
allowance is therefore revised again from 18 to 22 minutes. No workflow change
is warranted because output remains continuous, memory is safe, and the target
still finishes well before the reporting reserve.
