# Full-run plan

## Workload

Target `bert-squad` runs three resumable stages: build a Python 3.11 environment and
repository LoadGen; preprocess all SQuAD v1.1 validation questions into 10,833
384-token features; run fp32 BERT-Large in LoadGen Offline AccuracyOnly mode on the
local A100 and invoke the repository accuracy evaluator. The evaluator reads the
full LoadGen accuracy log and reports `exact_match` and `f1`.

## Measured costs

The final clean 8-example preflight took 153.02 seconds end to end. Stage endpoints
showed approximately 60 seconds for the environment and LoadGen build, 61 seconds
for all-data preprocessing, 22 seconds for model loading plus one inference batch
and LoadGen, and 9 seconds for bounded postprocessing/evaluation. The input has
10,570 questions and 10,833 derived features. Immediately before preflight, the
A100 had 75,102 MiB free of 81,920 MiB; the script therefore selected batch 32.

## Resources and schedule

Run stages in dependency order using the delivered script from a clean `--from
build` invocation. CPU build and preprocessing use the local host. GPU work uses
the single local A100 directly and begins only after a fresh `nvidia-smi` memory
check recorded in `gpu-before-run.log`. There is one target and no independent
ready CPU work remaining to overlap with its accelerator stage. GPU work is
single-run only.

## Expected time and reserve

The full inference has 339 batch-32 steps. Allow 25 minutes for model loading and
inference under shared-GPU contention, 5 minutes for full evaluation, and about 2
minutes for the already measured build/preprocess stages: 32 minutes expected.
Hold 15 minutes for artifact checks, `report.json`, and a final clean-deliverable
audit. This fits within the two-hour wall-clock limit.

## Re-plan at first full-run progress

At 2:39 elapsed in the full harness process, the LoadGen accuracy log was 22 MiB,
GPU utilization was 100%, and 74,718 MiB remained free. The observed output rate
projects the accelerator stage to approximately 7--9 minutes, below the original
25-minute allowance. The stage order and batch size remain unchanged; the overall
projection is now about 12 minutes plus the 15-minute reporting reserve.
