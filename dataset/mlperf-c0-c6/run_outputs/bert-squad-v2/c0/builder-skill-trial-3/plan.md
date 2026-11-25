# Reproduction plan

Written after successful bounded preflight on 2026-11-23 (America/New_York).

## Workload and dependencies

The single target `bert-squad` has three script stages. `env` privately stages the pinned source and supplied DeepLearningExamples tree, creates Python 3.11, installs pinned packages, and builds LoadGen 6.0.17. `preflight` depends on `env`, checks GPU memory, creates the complete validation feature cache, runs eight LoadGen features, and invokes the repository evaluator. `run` depends on both, checks GPU memory again, runs every validation feature in Offline AccuracyOnly mode, and evaluates all 10,570 questions.

## Measured costs and resources

- CPU environment stage: 60 seconds elapsed. The host exposes 8 CPU cores and about 2 TiB RAM; persistent environment state is under `bert-squad/`, and package artifacts use only the required shared uv cache.
- Current bounded preflight: 79 seconds elapsed, including full dataset feature conversion, loading the 1.34 GB model, eight GPU features, and evaluation. The immediate pre-run check reported 77,377 MiB free of 81,920 MiB.
- It emitted `pytorch device: cuda:0` and `{"exact_match": 75.0, "f1": 75.0}`. Those bounded scores are startup/evaluator evidence only, not target results.
- Full scope is 10,570 questions and about 10,833 BERT features. Inference is serial in the reference SUT. One local NVIDIA A100 80GB is used, with no Slurm and no concurrent GPU work started by this session.
- Provenance-matched trial 3 measured an initial full-run rate of about 5.7 features/second (434/10,833 records in 76 seconds) and the same preflight duration. It is secondary projection evidence; the current run remains authoritative.

## Schedule and projection

Launch `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from env`. The wrapper clears only its own stage outputs, rebuilds `env`, repeats bounded `preflight`, then runs the full target, checking free GPU memory before each GPU invocation. With one target there is no independent CPU target preparation to overlap with full inference.

Measured setup and preflight total about 2.3 minutes. The matched prior rate projects roughly 32 minutes for serial inference, plus setup/evaluation, for an expected total under 35 minutes. This fits the two-hour limit with at least 20 minutes reserved for report/artifact validation and one repair cycle.

After the first few percent of the current full run, compare observed accuracy-log record growth with this rate. If projected completion exceeds 100 minutes total, update this plan and preserve enough reserve to write the measured report and validate required artifacts.

## Re-plan: initial current full-run rate

At 71 seconds of full-run process time, the streaming LoadGen accuracy log contained 411 of approximately 10,833 feature records (3.8%), or 5.79 records/second including model startup. The linear projection is about 31 minutes for inference, consistent with the planned 32 minutes. GPU telemetry showed about 11,055 MiB used by all current jobs and 70,099 MiB free; this session's run was active at 100% utilization. No schedule change is required; continue the single serial run and retain the 20-minute closeout reserve.

## Completion

The full run and evaluator completed successfully in 1,333 seconds, faster than the conservative projection after shared-GPU contention eased. LoadGen wrote all 10,833 feature records with no warnings or errors. Final validation confirmed the three declared outputs, all stage logs, CUDA device prefix, exact report/evaluator agreement, success markers, and unchanged input checksums.
