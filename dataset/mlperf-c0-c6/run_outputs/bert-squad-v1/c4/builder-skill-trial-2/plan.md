# Reproduction plan

## Workload

One target, `bert-squad`, executed by the delivered `./reproduce.sh --target bert-squad` in three ordered stages:

1. `build`: verify the SQuAD/model/vocabulary hashes, copy the supplied repository, generate the benchmark-owned evaluation binding and compatibility shim in that private copy, create the pinned Python 3.11 environment, and compile LoadGen 6.0.17.
2. `run`: verify inputs again, check current A100 free memory, preprocess the full validation split, load BERT-Large fp32 on CUDA, and issue every validation feature through LoadGen Offline AccuracyOnly mode.
3. `evaluate`: bind all logged start/end logits to answer spans and compute exact match and F1.

The stages depend on their predecessor's success marker. No CPU work from another target exists to overlap with the single GPU run.

## Measured costs and resources

- Clean environment plus LoadGen build: approximately 45 seconds on 8 visible CPU cores; package artifacts came from the required shared uv cache.
- Preflight GPU availability: NVIDIA A100-SXM4-80GB, 77,395 MiB free immediately before the bounded run.
- Host memory: about 2.0 TiB total and 1.9 TiB available at initial inspection.
- Bounded LoadGen inference: 8 features in 0.748 seconds, or 10.7 features/s.
- Full feature count: 10,833 (`effective_performance_sample_count` in the bounded detail log).
- Bounded evaluation: approximately 7 seconds, producing exact match 87.5 and F1 87.5 for the represented examples.
- One-time full feature conversion in preflight took under one minute; the full clean replay will regenerate it.

Evidence: `bert-squad/build.log`, `bert-squad/run.log`, `bert-squad/evaluator.log`, and `bert-squad/preflight/out/mlperf_log_detail.txt`.

## Schedule and projection

Run exactly one GPU workload, locally and directly. The no-stage script first performs its CPU build and preprocessing, then starts the GPU pass only after its own `nvidia-smi` memory check. No other session's processes will be altered.

At the measured 10.7 features/s, inference projects to about 1,013 seconds (16.9 minutes). Budget 30 minutes total for clean rebuild, preprocessing, model load, shared-GPU contention, inference, and evaluation. Hold a further 20-minute reserve for `report.json`, artifact checks, and a standalone replay audit. Total planned occupancy is 50 minutes against the two-hour limit.

## Re-plan triggers

- Compare progress after roughly 300 logged/completed features or the earliest observable few percent. If the projection exceeds 100 minutes total, retain the required sequential source semantics but begin final-artifact risk assessment.
- On a failure, repair that failure class once in `reproduce.sh`, rerun a bounded end-to-end preflight, update this file, then launch a fresh no-stage full run.
- Begin no new run once the 20-minute reserve is reached; write `report.json` from measured results and preserve failure evidence.
