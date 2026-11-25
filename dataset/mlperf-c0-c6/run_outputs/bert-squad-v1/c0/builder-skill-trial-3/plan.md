# Reproduction plan

Written after successful bounded preflight on 2026-11-22 (America/New_York).

## Workload and dependencies

Target `bert-squad` has three script stages: `env` copies the pinned source/input support tree, creates Python 3.11 environment, installs pinned packages, and builds repository LoadGen 6.0.17; `preflight` depends on `env`, checks GPU memory, creates the complete validation feature cache, runs eight LoadGen features, and invokes the repository evaluator; `run` depends on both, checks GPU memory again, runs every cached validation feature in Offline AccuracyOnly mode, and evaluates all 10,570 questions.

## Measured costs and resources

- CPU environment stage: approximately 57 seconds elapsed, using up to 8 available CPU cores for compilation and shared host RAM. Persistent environment is under `bert-squad/`; package artifacts use only the specified shared uv cache.
- Bounded end-to-end preflight: 79 seconds elapsed. It included full dataset feature conversion, loading the 1.34 GB model, eight GPU features, and evaluation. The GPU check immediately before launch reported 75,558 MiB free of 81,920 MiB.
- The preflight emitted `pytorch device: cuda:0` and evaluator JSON `{"exact_match": 75.0, "f1": 75.0}`. Those scores cover only the bounded evidence run and are not reportable target results.
- Full scope: 10,570 questions and approximately 10,833 features (the repository configuration's BERT Offline count). GPU inference is serial in the reference SUT; one local NVIDIA A100 80GB is used, with no Slurm and no concurrent run started by this session.

## Schedule and projection

Launch `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from env`. The script itself clears only its owned stage products, rebuilds `env`, repeats `preflight`, then starts `run`; it checks free GPU memory before both GPU invocations. There is only one target, so there is no useful independent CPU target preparation to overlap with full inference.

The measured fixed setup plus preflight cost is about 2.3 minutes. Because the eight-feature preflight is dominated by setup and cannot provide a stable per-feature throughput estimate, budget a conservative 30 minutes for full serial inference and postprocessing, for an expected total below 35 minutes. This fits the two-hour limit while reserving at least 20 minutes for report/artifact validation and one repair cycle. After initial full-run progress appears, compare the LoadGen/run elapsed rate with this projection; if it projects beyond 100 minutes total, preserve measured artifacts and stop new work at the reserve boundary.

## Re-plan: initial full-run rate

At 76 seconds of full-run process time, the streaming accuracy log contained 434 of 10,833 feature records (4.0%), or approximately 5.7 features/second under current shared-GPU contention. Linear projection is about 32 minutes for inference, plus setup/evaluation, consistent with the original below-35-minute estimate. No schedule change is needed; continue the single serial GPU run and retain the 20-minute closeout reserve.
