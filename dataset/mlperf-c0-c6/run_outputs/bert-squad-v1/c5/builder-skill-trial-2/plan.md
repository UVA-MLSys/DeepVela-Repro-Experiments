# Reproduction plan

Planned 2026-11-22 after successful bounded preflight.

## Workload and dependencies

One target, `bert-squad`, runs in three ordered stages:

1. `environment`: create Python 3.10 environment from the authorized uv cache and install pinned
   PyTorch, Transformers, NumPy, LoadGen, and compatibility dependencies.
2. `prepare`: copy the BERT harness, supply the missing PyTorch SUT wrapper, verify the fixed input
   hashes/counts, and convert all SQuAD examples into 10,833 cached features on CPU.
3. `run`: check current GPU free memory, run one fp32 Offline AccuracyOnly pass, then run the
   benchmark-owned evaluator and validate required artifacts.

## Measured costs

- Environment creation/install: about 15 seconds when the shared uv cache is warm; environment
  size 5.2 GiB.
- Full CPU feature conversion: approximately 90 seconds; feature cache 51 MiB and 10,833 features.
- Bounded run/evaluation (128 features): 32.22 seconds wall, 19.07 seconds user CPU, 9.12 seconds
  system CPU, and 3.23 GiB maximum host RSS. The LoadGen output timestamp is 21.3 seconds after the
  pre-run GPU check; evaluator completion is another 10.7 seconds.
- Bounded evaluator result: exact match 92.96875, F1 94.68749999999999. This is validation evidence,
  not the full benchmark result.

Evidence: `bert-squad/out/build.log`, `prepare.log`, `gpu_memory.log`, `run.log`, `evaluator.log`,
`accuracy.txt`, and the `/usr/bin/time -v` result recorded in the session transcript.

## Resources and schedule

- Host: 128 logical CPUs and approximately 2 TiB RAM; stages execute locally, never via Slurm.
- Accelerator: one NVIDIA A100-SXM4-80GB. Immediately before the measured run it had 81,034 MiB
  free. The script refuses to launch below 18,000 MiB free. Other users' processes are left alone.
- GPU parallelism is exactly one reproduction run. Environment and preprocessing finish before the
  single accelerator stage because this target consumes their outputs; there is no second target
  with independent CPU work to overlap.
- Inference batch size is 16, chosen conservatively for a shared 80GB device. Accuracy semantics are
  batch-independent and remain fp32.

## Projection and reserve

Conservatively scaling the entire 21.3-second bounded GPU interval by 10,833/128 projects about
30.1 minutes for full inference, even though fixed model-load overhead makes this an overestimate.
Allow 3 minutes for environment/preprocessing, 5 minutes for full post-processing/evaluation, and
10 minutes for shared-GPU contention: projected total about 48 minutes. Hold 15 minutes of the
two-hour limit for report generation, artifact verification, and one bounded repair if necessary.
The plan therefore fits with roughly 57 minutes additional margin.

The full run command is:

```bash
OUTPUT_DIR="$PWD/bert-squad/out" ./reproduce.sh --target bert-squad --from environment
```

## Full-run re-plan and outcome

At 2,608/10,833 features after about two minutes, the observed full-run rate projected accelerator
completion in roughly nine minutes rather than the conservative 30.1-minute bound. No schedule or
batch-size change was needed. The run reached all 10,833 features, LoadGen reported no warnings or
errors, and evaluation completed within the original reserve.
