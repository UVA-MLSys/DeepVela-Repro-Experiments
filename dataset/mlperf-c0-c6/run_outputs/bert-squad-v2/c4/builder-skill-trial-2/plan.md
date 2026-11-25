# Reproduction plan

## Workload

One target, `bert-squad`, executed by the delivered `./reproduce.sh --target bert-squad` in three ordered stages:

1. `build`: verify the dataset/model/vocabulary hashes, copy the supplied repository, generate the benchmark-owned evaluator and compatibility shim in that private copy, create the pinned Python 3.11 environment, and compile LoadGen 6.0.17.
2. `run`: verify inputs again, check current A100 free memory, preprocess the full validation split, load BERT-Large fp32 on CUDA, and issue every validation feature through LoadGen Offline AccuracyOnly mode.
3. `evaluate`: bind every logged start/end-logit pair to answer spans and compute exact match and F1.

Each stage leaves a success marker only after all commands succeed. With one target there is no independent CPU stage to overlap with its GPU run.

## Measured costs and resources

- Clean environment and LoadGen build completed in about one minute on 8 visible CPU cores, resolving from the required shared uv cache.
- The current host exposes about 2.0 TiB memory, with about 1.9 TiB available at planning time.
- Immediately before the bounded GPU run, the NVIDIA A100-SXM4-80GB had 70,666 MiB free.
- The bounded LoadGen inference completed 8 features in 1.373 seconds (about 5.83 features/s) and exercised the same dataset/model/backend/scenario/mode/evaluator path as the full run.
- LoadGen reports 10,833 total evaluation features. Linear projection at the bounded rate is 1,858 seconds (31.0 minutes) for inference.
- The bounded evaluator produced exact match 87.5 and F1 87.5 over the 8 represented examples.

Evidence: `bert-squad/build.log`, `bert-squad/run.log`, `bert-squad/evaluator.log`, and `bert-squad/out/mlperf_log_detail.txt`.

## Schedule and projection

Run exactly one local GPU workload. The delivered no-stage command first rebuilds from a clean private copy, then checks GPU memory and begins inference. Budget 45 minutes for clean build, preprocessing, model load, inference under shared-GPU contention, and evaluation. Hold a further 25-minute reserve for artifact validation, `report.json`, and clean-replay checks. The 70-minute plan fits the two-hour limit.

The scheduling decision is driven by the measured 5.83 features/s rate and the mandatory single-GPU-run constraint. No other session's processes will be altered.

## Re-plan triggers

- After the first few percent of the full pass, compare observed elapsed time against the 45-minute execution budget. If the projection exceeds 95 minutes, preserve the required serial semantics and prioritize retaining failure evidence and required outputs.
- On a failure, repair that failure class once in `reproduce.sh`, rerun a bounded end-to-end preflight, update this plan, and launch a fresh no-stage full run.
- Start no new run after entering the 25-minute reserve; write `report.json` from measured results and preserve all logs.
