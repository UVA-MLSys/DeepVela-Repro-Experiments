# Full reproduction plan

## Workload

Target `bert-squad` runs four ordered stages from the delivered `reproduce.sh`:

1. `setup`: copy the pinned metadata-free source, stage only the imported NVIDIA
   BERT subtree, create an isolated Python 3.12 environment from `UV_CACHE_DIR`,
   and build the pinned LoadGen source.
2. `preprocess`: convert all SQuAD v1.1 validation examples to 384-token features.
3. `run`: check current A100 free memory, load BERT-Large fp32, and run LoadGen
   Offline accuracy over all 10,833 features.
4. `evaluate`: run the repository's `accuracy-squad.py`, which invokes
   `evaluate_v1.1.py`, and save its stdout.

## Measured costs and resources

- Machine: 8 CPU cores, approximately 2 TiB host RAM, one shared NVIDIA A100
  80GB. The two bounded GPU launches observed 64,774 and 69,273 MiB free.
- Clean setup plus LoadGen build: approximately 3 minutes on the shared
  filesystem; package resolution is offline from the required cache.
- Full preprocessing: 10,833 features, approximately 30 seconds once imports
  are warm; the resulting cache is 51 MiB.
- Bounded GPU run: 128 features in 22.149 seconds of measured LoadGen inference,
  or 5.78 features/s. Model startup and log setup add approximately 24 seconds.
- Bounded evaluation: under 20 seconds for 128 features.
- GPU memory gate: do not launch below 6,000 MiB free. The fp32 model used far
  less than the 64+ GiB observed free in preflight.

## Schedule and projection

There is one target, so accelerator work remains serial. CPU setup and
preprocessing precede it because the run consumes their files; evaluation
follows the completed accuracy log. At the measured steady-state rate, 10,833
features require about 1,874 seconds (31.2 minutes). Allowing 3 minutes for
setup, 1 minute for preprocessing, 35 minutes for contended inference, and 2
minutes for evaluation gives a 41-minute expected run. The launch budget is 55
minutes, retaining about 15 minutes for `report.json`, artifact checks, and one
bounded repair if needed.

The full run starts with `--from setup`, so it tests the delivered script from
nothing built and clears only state owned by its full-run target directory.

## Re-plan: first 3.5% of full inference

At 14:51:54, the growing accuracy log contained 383 records. Excluding the
approximately 24-second fixed model startup, the observed shared-GPU rate was
about 4.7 features/s, versus 5.78 features/s in preflight. The revised GPU
projection is approximately 38 minutes and the revised clean-run projection is
about 43 minutes. This remains below the 55-minute launch budget, so stage order
and single-GPU parallelism are unchanged.

## Actual completion

The clean full run completed all stages successfully at 15:12. GPU contention
fell during the pass, so all 10,833 features completed in about 21 minutes of
process time, faster than the revised projection. The official evaluator then
printed `exact_match=83.68968779564806` and `f1=90.87487229720105`.
