# Reproduction plan

Recorded 2026-11-22 after successful bounded end-to-end preflight.

## Workload and dependencies

Target `bert-squad` has four serial stages implemented by the delivered `reproduce.sh`:

1. `setup`: copy the pinned metadata-free source under `bert-squad/source`, apply the evaluator
   and device-log repairs, create a Python 3.11 environment, install pinned dependencies from the
   required uv cache, and build LoadGen 6.0.17 from the copied source.
2. `preprocess`: link the three immutable input artifacts and generate the complete 10,833-feature
   SQuAD cache on CPU.
3. `run`: check GPU free memory, load BERT-Large FP32 on `cuda:0`, and run all 10,833 features in
   LoadGen AccuracyOnly/Offline mode. This is the only accelerator stage.
4. `evaluate`: decode all logits, compute EM/F1 over all 10,570 questions, and write evaluator
   stdout plus required outputs.

## Measured costs

- Clean-ish setup/rebuild observed about 51 seconds; LoadGen compilation itself was about 30
  seconds. It used ordinary CPU compilation and target-local disk.
- Full preprocessing observed about 62 seconds and produced 10,833 features. It used CPU and
  host memory only.
- Four-feature bounded LoadGen took 3.997 seconds inside LoadGen and was startup dominated.
- The representative 100-feature bounded LoadGen took 10.586 seconds inside LoadGen, or 9.45
  features/s including warm-up. End-to-end GPU-stage wall time, including model import/load, was
  about 25 seconds.
- Bounded evaluation took about 4 seconds.
- The environment/model used roughly 6-9 GiB of GPU memory by before/after free-memory evidence;
  the two preflight checks saw 73,152 MiB and 75,576 MiB free of 81,920 MiB.

Evidence: `bert-squad/logs/{setup,preprocess,run,evaluator}.log` and
`bert-squad/out/mlperf_log_detail.txt` from bounded preflight.

## Resources and schedule

The machine exposes 8 CPU cores, one shared NVIDIA A100-SXM4-80GB, about 1.9 TiB available host
memory, and 4.8 TiB free workspace storage. No Slurm work is permitted. Since there is one target,
there is no ready independent CPU target to overlap with its GPU stage. Stages therefore run in
dependency order, and only one GPU run is launched after a fresh `nvidia-smi` check; the script
refuses to start below 8,192 MiB free.

The 100-feature rate linearly projects `(10.586 / 100) * 10833 = 1,147` seconds (19.1 minutes)
inside LoadGen. This deliberately retains warm-up overhead in every block and is conservative;
adding clean setup, preprocessing, model load, evaluation, and file copies gives an expected total
under 23 minutes.

## Limit and reserve

The wall-clock limit is two hours. Hold back 20 minutes for result parsing, required-file checks,
`report.json`, and the clean-copy replay. The expected primary run finishes more than 75 minutes
before that reserve begins. If the first few percent project beyond 100 minutes total, stop no
other work (none exists), preserve logs, and reassess from the measured LoadGen timestamps.

## Re-plan during full run

At 859/10,833 responses (7.9%), concurrent jobs were holding part of the shared GPU and the
observed full-run rate was approximately 6 features/s rather than the bounded pass's 9.45. The
revised accelerator projection is approximately 30 minutes. The run remains well inside the
100-minute launch window and there is no independent target or CPU stage to reschedule, so the
schedule remains serial and the GPU pass continues unchanged.

## Completion and clean replay

The primary LoadGen pass completed all 10,833 unique QSL indices in 1,206.508 seconds and reported
no warnings or errors. A clean replay was then launched from `clean-replay/`, whose initial top
level contained only an identical `reproduce.sh` and a fresh `repo/` copy. It rebuilt the Python
environment, LoadGen extension, evaluator repair, and complete feature cache; performed its own
GPU check (81,038 MiB free); and completed all 10,833 unique QSL indices with no warnings or
errors. Both runs printed exactly `83.68968779564806` exact match and
`90.87487229720105` F1. The delivered and replayed scripts have identical SHA-256
`e03f55fb523aaf217e6b1bee6c19b7f4e411612aec87c3b3210e896d3b381815`.
