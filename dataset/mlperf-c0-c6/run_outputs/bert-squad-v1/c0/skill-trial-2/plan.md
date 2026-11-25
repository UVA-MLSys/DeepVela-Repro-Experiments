# Reproduction plan

## Workload

- Target `bert-squad`: build an isolated Python 3.11 environment; stage the pinned BERT source and supplied DeepLearningExamples commit; preprocess the SQuAD v1.1 validation set; run fp32 BERT-Large with PyTorch in LoadGen Offline AccuracyOnly mode; evaluate the LoadGen accuracy log with `accuracy-squad.py`.
- Inputs: 10,570 validation questions producing 10,833 QSL features, cached model `model.pytorch`, cached `vocab.txt`, and the supplied DeepLearningExamples commit `729963dd47e7c8bd462ad10bfac7a7b0b604e6dd`.

## Measured costs and resources

- Environment build: approximately 25 seconds, CPU/filesystem only.
- Full preprocessing: approximately 60 seconds, CPU only; generated 10,833 features.
- Eight-feature end-to-end preflight: approximately 18 seconds including model load and evaluation; GPU check immediately before launch showed 77,377 MiB free of 81,920 MiB.
- One local NVIDIA A100 80GB is used by the run stage. GPU work is serialized; no other target exists.

## Schedule and expected time

Run `reproduce.sh --target bert-squad --from build` with no preflight bound. This recreates the environment and preprocessing cache, checks current GPU memory, runs all 10,833 features, and invokes the repository evaluator. The measured setup costs are under two minutes; extrapolating the bounded execution and allowing for per-query inference projects the complete run at several minutes and safely below the two-hour limit.

Reserve 20 minutes for a repair if the clean run fails, exact result extraction, `report.json`, artifact checks, and a final clean-copy/script inspection.

## Re-plan during full run

- Trigger: after roughly 30 seconds of inference, the accuracy log reached about 0.8 MB while four unrelated BERT processes shared the GPU.
- Revised projection: observed early output rate implies approximately 35–45 minutes for all 10,833 features, rather than several minutes.
- Schedule decision: keep this session's single GPU run unchanged. The revised projection remains below the two-hour limit with the 20-minute reporting/repair reserve, and starting another local GPU run would violate the serialization rule.

- Second rate check: at the first several percent, the log reached 3.9 MB after about 75 seconds of inference. The steadier rate revises the full inference projection to approximately 15–20 minutes. No schedule change is needed.

## Completion

- Full inference completed in approximately 19 minutes under shared-GPU contention, followed by successful repository evaluation.
- The final accuracy log contains 10,833 unique QSL records spanning indices 0 through 10,832.
- Evaluator result: exact match `83.68968779564806`, F1 `90.87487229720105`.
