# Reproduction plan

Written after the successful bounded end-to-end preflight on 2026-11-22.

## Workload and dependencies

There is one target, `bert-squad`, with three serialized stages owned by `reproduce.sh`:

1. `env` copies the pinned source into target-local work space, creates a Python 3.11 environment from the authorized uv cache, stages the pinned DeepLearningExamples input, applies the required device-log instrumentation to the staged copy, and builds LoadGen 6.0.17.
2. `preprocess` depends on `env` and transforms all 10,570 validation questions into 10,833 BERT features using sequence length 384, query length 64, and document stride 128.
3. `run` depends on `preprocess`, checks current free GPU memory, loads the BERT-Large fp32 checkpoint on the A100, runs all 10,833 features through LoadGen in Offline AccuracyOnly mode, and invokes `accuracy-squad.py`/`evaluate_v1.1.py` once.

The full run will use `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from env`, which clears only script-owned target artifacts and exercises the delivered script from nothing built.

## Measured preflight costs

| Stage | Bounded measurement | Time/resource evidence |
|---|---|---|
| `env` | complete environment and LoadGen build | 45 seconds, from `bert-squad/build.log` timestamps; cached install resolved 47 packages and local LoadGen compiled successfully |
| `preprocess` | complete validation preprocessing | 60 seconds, from `bert-squad/preprocess.log`; produced 10,833 features and a 53,468,824-byte cache |
| model load + bounded `run` | eight LoadGen accuracy samples and evaluator | GPU check recorded 75,684 MiB free of 81,920 MiB; model parameters logged on `cuda:0`; LoadGen recorded 1,882.730841 ms for eight samples (235.34 ms/sample) |
| outputs | bounded evidence only | both required LoadGen files and evaluator output were nonempty; evaluator returned one parseable result object |

The machine exposes 8 CPU cores and approximately 1.9 TiB available host memory at planning time. The script requires at least 6,144 MiB free GPU memory immediately before each run; the measured preflight had much more. GPU peak memory was not sampled, so the plan does not claim a measured peak.

## Schedule and expected time

The single target runs `env`, `preprocess`, then `run` serially. There is no other target or ready independent CPU workload to overlap. Only one GPU run is launched, satisfying the shared-GPU rule.

The inference projection is `10,833 × 235.34 ms = 42.49 minutes`. Adding the measured 1.75 minutes for environment plus preprocessing, model-loading/evaluator overhead, and filesystem/logging margin gives a 47-minute expected completion. This fits the two-hour limit.

The scheduling decision is driven by the measured per-sample inference time, not the README's target QPS (which is irrelevant in AccuracyOnly mode). The first few percent of the full accuracy log will be checked against the 42.49-minute inference projection if observable; the script's synchronous per-sample SUT provides no safe batching or parallelism change without altering the documented route.

## Reserve and closeout

Reserve 15 minutes before the two-hour deadline for parsing the authoritative evaluator line, writing `report.json`, validating all required artifacts and logs, and checking a separate clean-copy replay of the delivered script if time permits. No new GPU run will begin once that reserve is reached.

## Completion re-plan

The full run completed before an intermediate progress fraction was observable because AccuracyOnly mode buffered the single Offline query's result. LoadGen's full-query duration was materially faster than the eight-sample linear projection, so no schedule change was needed. The remaining work is CPU-only artifact validation and report closeout; no further GPU run is scheduled.
