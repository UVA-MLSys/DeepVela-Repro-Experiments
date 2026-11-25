# Reproduction plan

Planned 2026-11-23 after a successful bounded preflight.

## Workload and dependencies

The sole target, `bert-squad`, has three ordered stages:

1. `environment`: create a Python 3.10 environment from the authorized uv cache and install pinned
   PyTorch, Transformers, NumPy, LoadGen, and compatibility dependencies.
2. `prepare`: copy the harness to target-local work, generate the missing PyTorch SUT wrapper,
   verify fixed input hashes, and convert the full SQuAD development set into cached CPU features.
3. `run`: check GPU free memory, execute one fp32 Offline AccuracyOnly pass, invoke the benchmark
   evaluator, and validate required files and metric JSON.

## Measured costs

- Environment: 23.37 seconds wall; resulting packages are resolved from `UV_CACHE_DIR` offline.
- Full preprocessing: 80.33 seconds wall; 10,833 features in a 51 MiB cache.
- Repaired replay from `prepare`, including preprocessing and 16-feature GPU/evaluator preflight:
  109.04 seconds wall.
- The successful preflight LoadGen GPU interval was 2.058 seconds for 16 features while the shared
  GPU reported 100% utilization from other work. Its evaluator printed exact match 81.25 and F1
  81.25; these bounded values are validation evidence, not benchmark results.
- Immediately before preflight, the A100 had 69,178 MiB free. The launcher refuses to run below
  18,000 MiB free. Host storage has 5.5 TiB available and host RAM has about 1.9 TiB available.

Evidence: `bert-squad/out/build.log`, `prepare.log`, `gpu_memory.log`, `run.log`,
`mlperf_log_detail.txt`, `evaluator.log`, and `accuracy.txt`.

## Resources and schedule

- All work runs locally, never through Slurm.
- Exactly one reproduction GPU run executes. Batch size 16 is validated and comfortably fits the
  observed free memory; batching does not change fp32 accuracy semantics.
- Environment and preprocessing precede inference because this single target consumes both; there
  is no independent second target whose CPU work can overlap the GPU stage.
- The full run starts from `environment`, causing the launcher to clear and recreate all of its own
  build state and outputs. No source or read-only cache input is modified.

## Projection and reserve

Linear scaling of the current 2.058-second GPU interval gives about 23.2 minutes for 10,833
features. This is conservative because it scales fixed per-run overhead. The verified prior trial's
plan records roughly nine minutes for the same full feature count, providing a comparable lower
estimate. Budget 30 minutes for accelerator work under present contention, three minutes for a
clean environment/preprocessing rebuild, five minutes for post-processing/evaluation, and hold 15
minutes for artifact verification and `report.json`. The projected 53-minute envelope fits the
two-hour limit.

Full clean-replay command:

```bash
OUTPUT_DIR="$PWD/bert-squad/out" ./reproduce.sh --target bert-squad --from environment
```

## Full-run outcome

The clean replay completed in 580.69 seconds total (540.05 seconds user CPU, 20.52 seconds system
CPU), substantially inside the conservative envelope. LoadGen processed all 10,833 accuracy
samples, reported no warnings or errors, and the benchmark evaluator completed over all 10,570
questions. No re-plan, retry, or launcher edit occurred during the full run.
