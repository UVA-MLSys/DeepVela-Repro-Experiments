# Reproduction plan

Written 2026-11-22 after successful bounded preflight.

## Workload and dependencies

One target, `bert-squad`, runs three ordered stages from `reproduce.sh`:

1. `environment`: validate the three immutable input hashes; copy the fresh source;
   stage the supplied DeepLearningExamples tree; create Python 3.11 environment;
   compile/install LoadGen.
2. `preprocess`: convert the SQuAD v1.1 validation questions to BERT features with
   sequence length 384, query length 64, and stride 128.
3. `run`: check GPU memory; run PyTorch fp32 BERT-Large in Offline AccuracyOnly mode;
   evaluate the LoadGen log with `accuracy-squad.py` / `evaluate_v1.1.py`; copy scored
   outputs.

## Measured costs

The successful four-question preflight measured about 43 s for a cold environment,
6 s preprocessing, 22 s GPU stage (dominated by model startup), and 5 s evaluation.
The successful 100-question preflight measured 5 s preprocessing, 24 s GPU stage,
and 6 s evaluation. The latter printed `exact_match=94.0`, `f1=94.8`; these are
bounded evidence, not reportable full-run values.

The incremental GPU estimate from 4 to 100 questions is approximately
`(24 - 22) / 96 = 0.021 s/feature`. At roughly 10,833 validation features, inference
projects to about 4 minutes including the measured startup. Full feature conversion
and evaluator scaling are less certain, so the planned total is 15 minutes.

The environment occupies 5.2 GiB. This machine exposes 8 CPU cores, about 1.9 TiB
available RAM, and one NVIDIA A100-SXM4-80GB. Immediately before the 100-question GPU
run, `nvidia-smi` reported 81,034 MiB free.

## Schedule, resources, and reserve

Run a clean replay serially because there is one target and one GPU. Environment and
preprocessing are CPU-only; the GPU stage begins as soon as they succeed. No other GPU
work is launched. With a 15-minute projection and a two-hour wall-clock limit, reserve
20 minutes for result extraction, `report.json`, artifact validation, and a final
deliverable check. If the first few percent imply more than 85 minutes total, stop new
work and preserve measured outputs before the reserve.
