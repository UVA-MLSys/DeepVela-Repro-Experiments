# Reproduction plan

## Workload

Target `bert-squad` has three stages: (1) create a Python 3.11 environment and
build the pinned repository's LoadGen, (2) stage the fixed local inputs and
preprocess all SQuAD v1.1 validation examples into 10,833 features, and (3) run
BERT-Large fp32 with the PyTorch CUDA backend in Offline AccuracyOnly mode and
invoke `accuracy-squad.py`. Each stage leaves a success marker and the next
stage reads only its predecessor's files.

## Measured costs

- Environment: about 55 seconds in preflight, 8 CPU cores available, 5.2 GiB
  target-local environment, no GPU.
- Prepare: about 80 seconds for all 10,833 features, 51 MiB feature cache, CPU
  only.
- GPU run: 8 samples completed in 31.25 ms of LoadGen query duration (about
  256 samples/s). Including checkpoint I/O and CUDA/model startup, the bounded
  run log completed in about 29 seconds. The full inference projection is about
  43 seconds plus startup.
- Evaluator: about 8 seconds for the bounded output; allow 60 seconds for the
  full log and post-processing.
- Preflight A100 state immediately before inference: 3,795 MiB used and 77,359
  MiB free out of 81,920 MiB. Host memory available was about 1.9 TiB; workspace
  free space was about 4.8 TiB.

## Schedule and resources

Run the target once from a fresh `bert-squad/` target directory using the
delivered no-`--stage` path. Environment build and preprocessing run first on
the local CPUs. Immediately before the accelerator stage, record `nvidia-smi`;
then run the sole GPU workload directly on GPU 0. There are no other targets or
ready CPU stages to overlap, and GPU concurrency remains one as required.

## Expected time and reserve

The measured projection is roughly 4 minutes end to end, allowing filesystem
variance. Budget 15 minutes for post-run artifact checks, exact metric
extraction, `report.json`, and a clean-copy portability inspection. This fits
well within the two-hour wall-clock limit.

## Re-plan during full run

At 120 seconds of full GPU execution, `mlperf_log_accuracy.json` had reached
9.46 MB while the GPU reported 100% utilization. A separately owned BERT job
was concurrently active on the shared GPU. Extrapolating the response-log
growth revises full inference to about 7 minutes and the end-to-end run to less
than 10 minutes. No concurrency or target order changes are needed: this remains
the only GPU run started by this session, and the revised projection still fits
comfortably before the 15-minute reporting reserve and two-hour limit.
