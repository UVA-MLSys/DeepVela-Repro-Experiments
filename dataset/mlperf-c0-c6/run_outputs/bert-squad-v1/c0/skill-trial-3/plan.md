# BERT SQuAD reproduction plan

## Workload

Target `bert-squad` runs the repository's PyTorch BERT-Large FP32 SUT in the
default Offline scenario and AccuracyOnly mode. Setup builds the repository's
LoadGen and stages the supplied DLE tree. Preflight and full run share the
SQuAD v1.1 feature cache because dataset, vocabulary, sequence length (384),
stride (128), query length (64), and producer are identical. The repository's
`accuracy-squad.py` and `evaluate_v1.1.py` evaluate the complete LoadGen log.

## Measured costs and resources

- Setup: about 40 seconds on CPU, including cached package installation and
  LoadGen compilation.
- Preprocessing: about 70 seconds on CPU for all 10,570 questions, producing
  10,833 features. It is complete and reusable by the full run.
- Bounded accelerator check: 8 features completed without LoadGen warnings or
  errors and evaluated successfully. Its inference portion completed within
  one output polling interval (under 10 seconds); fixed checkpoint loading was
  about 10 seconds.
- Accelerator: one local NVIDIA A100-SXM4-80GB. Checks immediately before the
  bounded passes observed 75,666--79,323 MiB free, far above the runner's
  8,192 MiB minimum. GPU work remains serial.

## Schedule and projection

Setup and preprocessing are complete. Run the single full accelerator pass,
then the CPU evaluator. The bounded rate gives a conservative upper projection
of 15 minutes for 10,833 features plus 5 minutes for evaluation and artifact
checks. No other target or ready CPU stage exists to overlap. Reserve 20
minutes for a repair, `report.json`, and final validation; the projected total
is well within the two-hour limit.

If the first full-run progress or elapsed time projects past 20 minutes,
re-plan from the observed rate. Do not start another GPU pass while this one is
active.

## Re-plan: measured full-log growth

At 10:25 elapsed, `mlperf_log_accuracy.json` was 33.0 MB, approximately half
the expected complete size for 10,833 FP32 outputs of shape `(384, 2)`. This
moves the inference projection to about 21 minutes under current contention,
with a 25--30 minute projection including evaluation and checks. That exceeds
the initial 20-minute run allowance but remains far inside the two-hour task
limit and its 20-minute closing reserve. Continue the existing pass; restarting
would discard useful work and would not reduce contention.

## Completion

LoadGen completed after about 20 minutes under shared-GPU contention and
reported no warnings or errors. CPU post-processing and evaluation completed
in about 20 seconds. The complete log contains 10,833 unique QSL indices and
the evaluator printed exact match `83.68968779564806` and F1
`90.87487229720105`.

## Re-plan: shared-GPU saturation

At 2:16 elapsed in the full run, the process was healthy at about 97% CPU and
1,814 MiB GPU memory. The A100 was at 100% utilization with five independent
benchmark processes, only one of which belongs to this run. The contention can
extend runtime toward the conservative 15-minute inference bound, but does not
move the projection beyond the 20-minute run allowance. Keep one local GPU
pass active and do not alter the other sessions.
