# Full reproduction plan

## Workload

Target `bert-squad` runs four ordered stages from the delivered `reproduce.sh`:

1. `setup`: copy the pinned metadata-free source, stage the provided NVIDIA BERT
   subtree, create an isolated Python 3.12 environment from `UV_CACHE_DIR`, and
   build LoadGen from the pinned source.
2. `preprocess`: convert the complete SQuAD v1.1 validation set into 384-token
   features.
3. `run`: check current A100 free memory, load BERT-Large fp32, and run LoadGen
   Offline accuracy over all features.
4. `evaluate`: run the repository's `accuracy-squad.py`, which invokes
   `evaluate_v1.1.py`, and retain its stdout.

## Measured costs and resources

- Machine: 8 CPU cores, about 2 TiB host RAM, and one shared NVIDIA A100 80GB.
- Clean bounded setup and LoadGen build: about 1 minute, using only the required
  offline uv cache and the `2026-11-10` resolution cutoff.
- Complete preprocessing: 10,833 features and a 51 MiB cache in about 1 minute.
- Bounded GPU run: 128 samples in 13.974 seconds of LoadGen inference, or 9.16
  samples/s, with 73,482 MiB free at launch. Model startup added about 25 seconds.
- Bounded official evaluation: about 13 seconds for 128 samples.
- GPU memory gate: launch only with at least 6,000 MiB free; the fp32 model needs
  much less than the observed free memory.

## Schedule and projection

There is one target, so GPU work is serial. Setup and preprocessing precede the
GPU stage because it consumes their files, and evaluation follows its accuracy
log. The measured inference rate projects 10,833 samples to 19.7 minutes. To
allow for shared-GPU contention, the clean full run budget is 27 minutes:
2 minutes setup, 1 minute preprocessing, 22 minutes inference, and 2 minutes
evaluation. A further 15 minutes is reserved for report generation, artifact
validation, and a bounded repair if required.

The full run uses the unchanged delivered script with `--from setup`, clearing
and recreating only its own full-target work and output state.

## Re-plan: first 6.8% of full inference

At 13:23:29, the growing accuracy log held 742 records. Its latest interval
added 439 samples in about 38 seconds, or 11.6 samples/s, versus 9.16 samples/s
in preflight. The revised inference projection is about 16 minutes and the
clean-run projection about 19 minutes. This remains well below the 27-minute
run budget, so stage order and single-GPU parallelism are unchanged.

## Actual completion

The first clean full run completed all four stages successfully. LoadGen
processed 10,833 unique accuracy samples in 1,040.081 seconds (17.33 minutes)
and reported no warnings or errors. The official evaluator printed
`exact_match=83.68968779564806` and `f1=90.87487229720105`.
