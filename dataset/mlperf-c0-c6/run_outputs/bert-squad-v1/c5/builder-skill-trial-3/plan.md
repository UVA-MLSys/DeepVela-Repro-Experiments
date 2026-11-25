# Reproduction plan

## Workload and dependencies

Target `bert-squad` runs as one pipeline:

1. `environment` stages the supplied repository and pinned DeepLearningExamples tree,
   creates a Python 3.11 uv environment, installs the fixed packages, generates the missing
   PyTorch SUT wrapper, and builds the repository's LoadGen.
2. `preprocess` converts the complete SQuAD v1.1 validation set into 10,833 BERT feature
   windows and saves the feature cache.
3. `run` checks current A100 memory, loads the FP32 BERT-Large checkpoint, runs all 10,833
   feature windows in LoadGen Offline AccuracyOnly mode, and writes the required LoadGen
   logs.
4. `evaluate` decodes that accuracy log with `accuracy-squad.py`, invokes the benchmark
   evaluator, and writes `accuracy.txt`.

Each stage depends on the previous stage's success marker. The final measured replay starts
with `./reproduce.sh --target bert-squad --from environment`, which removes all script-owned
work and output files before rebuilding.

## Preflight measurements

The final bounded clean replay used 256 feature windows and batch size 16:

| Stage | Measured cost | Resource evidence |
|---|---:|---|
| environment | 58 s | 8 host cores; uv cache; LoadGen compiled locally |
| preprocess | 61 s | CPU-only; 53,468,824-byte cache with 10,833 features |
| run | 33 s total | A100; 73,006 MiB free immediately before launch |
| LoadGen inference within run | 16.90 s / 256 features | 15.15 features/s from `mlperf_log_detail.txt` |
| evaluate | 8 s | CPU-only; evaluator returned finite EM and F1 |

The bounded result was `exact_match=92.1875`, `f1=93.76302083333331`; it is validation
evidence only and will not be reported as the full result.

## Resources and schedule

The machine exposes 8 CPU cores, about 2 TiB host RAM, and one shared NVIDIA A100 80GB.
Other sessions may continue to use the GPU; the script records free memory immediately before
its only GPU stage. Batch size 16 consumed a small fraction of the available memory in preflight,
so it is retained rather than increasing shared-GPU pressure.

Only this target exists. Its CPU environment and preprocessing stages run serially before its
single GPU stage, followed by CPU evaluation. No process started by this reproduction overlaps
the GPU stage with another reproduction GPU job.

## Projection and reserve

At the measured LoadGen rate, full inference is
`10833 / 15.15 = 715 s` (11.9 min). Adding the observed model/load overhead, environment,
preprocessing, evaluator, and filesystem variance gives a 16-minute central estimate and a
25-minute conservative projection. This is below the two-hour wall-clock limit. A 15-minute
reserve is held for `report.json`, artifact verification, and one clean replay repair if needed.

The rate from the first full LoadGen progress evidence will be compared with 15.15 features/s.
If projected completion exceeds 105 minutes from launch, no new work will be started and the
required report/artifacts will be written from available measurements.
