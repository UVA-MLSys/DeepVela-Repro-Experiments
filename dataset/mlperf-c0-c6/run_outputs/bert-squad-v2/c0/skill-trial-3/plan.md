# BERT SQuAD reproduction plan

## Workload

Target `bert-squad` stages the pinned source and supplied DeepLearningExamples
tree, builds LoadGen in a Python 3.11 environment, preprocesses all SQuAD v1.1
validation examples, runs the PyTorch BERT-Large FP32 SUT in Offline
AccuracyOnly mode, and invokes the repository's `accuracy-squad.py` evaluator.
The full scope is 10,570 questions represented by 10,833 tokenized features.

## Measured costs and resources

- Environment setup took about 35 seconds on CPU, using the mandated uv cache.
- Full-dataset preprocessing took about 55 seconds on CPU and produced 10,833
  features.
- The bounded 8-feature accelerator check, including fixed checkpoint loading,
  took about 25 seconds; its evaluator took about 2 seconds.
- Immediately before that GPU pass, the local NVIDIA A100-SXM4-80GB had
  77,377 MiB free. The model used CUDA and LoadGen reported no warnings or
  errors.
- The compatible prior campaign measured about 20 minutes for the same 10,833
  feature GPU pass under heavy shared-GPU contention and about 20 seconds for
  evaluation.

## Schedule and projection

There is one target, so GPU work remains serial. Launch the deliverable from
`--from setup` to prove it works from nothing built: setup, preprocessing, the
bounded preflight, the full GPU pass, then CPU evaluation. There is no second
target with ready CPU work to overlap. The comparable full run drives a
conservative 30-minute projection, with a 20-minute reserve for one repair,
artifact validation, and `report.json`; this fits within the two-hour limit.

After the first few percent of the full log is written, compare its growth and
GPU utilization with the prior 20-minute pass. Continue while the projection
stays inside the closing reserve; do not start a second GPU pass concurrently.

## Re-plan: early full-run rate

After 61 seconds, the full accuracy log was 3.13 MB while the GPU was at 100%
utilization with five independent benchmark processes. Only PID 604324 belongs
to this reproduction. Scaling the first roughly 5% of expected log output
projects 21--25 minutes for inference, consistent with the compatible prior
run. Keep the serial pass active; the total remains inside the 30-minute
working projection and 20-minute closing reserve.

## Completion

The clean `--from setup` execution completed the full inference pass in about
21 minutes despite shared-GPU contention. LoadGen reported no warnings or
errors. The accuracy log contains all 10,833 unique QSL indices with no gaps or
duplicates, and the repository evaluator printed exact match
`83.68968779564806` and F1 `90.87487229720105`.
