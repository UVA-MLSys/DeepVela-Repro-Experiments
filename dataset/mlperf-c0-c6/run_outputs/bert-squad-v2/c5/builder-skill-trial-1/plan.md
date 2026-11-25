# Full reproduction plan

Written after successful bounded preflight on 2026-11-23.

## Workload and dependencies

The single target `bert-squad` is executed by the delivered `reproduce.sh` through:

1. `env`: stage the pinned source privately, supply the missing PyTorch SUT, remove the one
   nonexistent dependency-only LoadGen header entry, create Python 3.11 from the offline shared
   uv cache, and compile LoadGen.
2. `prepare`: convert the fixed SQuAD v1.1 validation split into BERT features.
3. `preflight`: recheck GPU memory and repeat the bounded two-feature LoadGen/evaluator test.
4. `gpu`: recheck GPU memory, conservatively choose batch 16 at >=24 GiB free, batch 8 at
   >=12 GiB, or batch 1 at >=6 GiB, then run the complete FP32 Offline accuracy pass.
5. `evaluate`: post-process the full LoadGen accuracy log and evaluate all 10,570 questions.

The clean replay command is:

```bash
OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from env
```

`--from env` makes the wrapper clear and recreate its own environment, outputs, and markers. This
is the first full benchmark run from nothing built under the deliverable conditions.

## Measured costs and resources

- Environment stage: approximately 53 seconds (13:19:50 through 13:20:42); LoadGen 6.0.17
  compiled and imports passed with PyTorch 2.4.0+cu121, Transformers 4.31.0, and NumPy 1.26.4.
- Preprocessing: approximately 57 seconds (13:20:48 through 13:21:49); 10,833 features for
  10,570 questions, with a 51 MiB feature cache.
- Bounded end-to-end preflight: approximately 25 seconds (GPU check 13:21:54 through evaluator
  completion 13:22:19). It loaded the complete 1.34 GB checkpoint, processed two features, and
  returned finite exact-match and F1. Because startup dominates, this proves correctness and fit
  but is not extrapolated linearly.
- Accelerator at bounded launch: NVIDIA A100-SXM4-80GB, 81,920 MiB total, 74,734 MiB free,
  driver 595.71.05; batch 16 selected. The wrapper checks again immediately before each GPU run.
- Host: eight visible CPU cores and 1.9 TiB available RAM at planning time.
- `user.conf` specifies Offline target QPS 32. At that rate, 10,833 features require about 339
  seconds, before startup and evaluation. The prior read-only trial took approximately 18.5
  minutes end to end under shared-GPU contention, so that is the stronger comparable estimate.

## Schedule, projection, and reserve

There is one target and one GPU, so GPU work is serial. Environment setup and preprocessing are
dependencies and cannot overlap the measured pass. Evaluation depends on the accuracy log. With
no independent target ready, the five stages run sequentially.

The expected replay is 20 minutes; the launch budget is 35 minutes to tolerate shared-GPU and
filesystem variation. This fits within the two-hour limit with a 20-minute reserve for reporting,
artifact validation, and standalone checks. Batch 16 is driven by the measured 74,734 MiB free;
if current free memory is lower, only batch size changes, never model, precision, data, scenario,
or evaluator.

## Re-plan triggers

- At the first measurable fraction of the growing accuracy log, compare its size/rate with the
  prior approximately 67 MB full-log estimate. Record a re-plan if projected completion changes
  materially.
- If GPU free memory is below 6 GiB, stop rather than interfere with other sessions.
- If projection exceeds 100 minutes (two hours less the 20-minute reserve), start no replacement
  full run and preserve measured failure evidence.
- A failure returns to its failed stage using `reproduce.sh --from <stage>` after one
  evidence-backed repair. Do not retry one failure type more than three times.

## Re-plan 1: first measurable full-run fraction

At 2026-11-23 13:26:56-05:00, `mlperf_log_accuracy.json` was 10,011,581 bytes, about
15% of the prior approximately 67 MB full-log estimate, 109 seconds after the 13:25:07 GPU
preflight. The GPU was at 100% utilization with multiple unrelated jobs, while this run used
2,248 MiB and 69,901 MiB remained free. A simple contention-adjusted projection is about 12
minutes for the GPU stage and about 15 minutes for the complete clean replay. This remains inside
both the 20-minute expectation and 35-minute launch budget, so batch 16, schedule, and reserve are
unchanged.

## Completion

The clean replay ran from 13:23:02 through 13:37:14-05:00, approximately 14 minutes 12
seconds. The complete GPU accuracy pass finished with no LoadGen warnings or errors, after which
the canonical evaluator reported exact match 83.68968779564806 and F1 90.87487229720105. Required
artifact hashes were written by the delivered script to `bert-squad/out/output-hashes.log`.
