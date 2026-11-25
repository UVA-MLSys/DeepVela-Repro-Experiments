# Reproduction plan

Updated: 2026-11-18 23:49 EST, after bounded preflight.

## Workload

One target, `3d-unet`, with three wrapper stages: (1) offline uv environment and target-local
source/data view setup, (2) CPU preprocessing plus repository checksum verification for all 43
inference-list entries, and (3) serialized A100 Offline accuracy inference followed by the
benchmark-owned CPU evaluator. Each stage leaves a success marker. The run stage depends on the
preprocessed pickles and writes the required LoadGen logs before evaluation.

## Measured costs

The bounded preflight used `case_00185`, a 384×384×384 preprocessed case covering 125 sliding
windows. Environment setup completed in about 1 minute. Preprocessing plus checksum verification
took 26.805 seconds. Inference took 19.236 seconds end to end, of which model inference was 9.123
seconds. Evaluation and prediction materialization took 53.593 seconds. GPU free memory immediately
before inference was 81,037 MiB; the host exposes 192 CPU cores and about 1.5 TiB RAM. The
preflight's three required outputs and device-prefix log were present.

## Resources and schedule

Run locally, never through Slurm. Setup and preprocessing use CPU and GPFS only; preprocessing uses
four processes to limit shared-filesystem pressure. Check free GPU memory immediately before the
single full GPU run, then run that accelerator stage alone. No other target exists whose CPU work
could overlap the GPU stage. The evaluator uses four CPU processes after LoadGen finishes because
it consumes the complete accuracy log.

## Expected time and reserve

Scaling the bounded measurements linearly gives about 19 minutes for 43 cases, but four-way
preprocessing/evaluation parallelism and the large chosen preflight case should reduce this. Budget
25 minutes for the clean full run: 1 minute setup, 8 minutes preprocessing/checksums, 8 minutes
inference, and 8 minutes evaluation/I/O. Hold back 8 minutes before the one-hour limit for
`report.json`, required-file validation, log inspection, and a staged clean-replay check. The
preflight's 81,037 MiB free-memory observation supports the unbatched reference implementation;
no throughput or batching change is planned.

## Re-plan triggers

After the first few full-run cases, compare observed seconds per case with the 8-minute inference
allowance. If projected completion exceeds the reserve boundary, retain the fixed benchmark
semantics and report the measured partial result rather than changing cases, backend, precision,
scenario, or evaluator. Any failure returns to the failed wrapper stage using `--from`, which
clears that stage's own markers and outputs.

## Actual full run

The clean full preprocessing stage completed in 2m33s with all 43 checksums passing. The first
full GPU stage completed in 4m01s and evaluation in 1m50s. After correcting stdout/stderr routing
for `accuracy.txt`, `--from run` replayed the identical workload in 4m08s plus 2m13s evaluation
and printed identical metrics. Both GPU launches recorded free memory first (78,276 MiB and
67,487 MiB respectively). The run remained inside the reserve, so no workload re-plan was needed.
