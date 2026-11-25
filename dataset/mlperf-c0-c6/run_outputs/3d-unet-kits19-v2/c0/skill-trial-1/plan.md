# Full reproduction plan

## Workload

- Target: `3d-unet`, the only entry in `tasks.json`.
- Setup: create a Python 3.11 uv environment from the mandated offline cache, copy the pinned app and LoadGen 6.0.17 source, apply only the SciPy API compatibility repair and required device log line, and build LoadGen.
- Preprocess: all 43 cases in `meta/inference_cases.json`, mapping documented duplicate `case_00400` to the supplied `case_00185`, with the repository checksum verifier enabled.
- Run and evaluate: repository `run.py --backend=pytorch --scenario=Offline --accuracy` on the fp32 TorchScript model, followed by repository `accuracy_kits.py` over LoadGen's complete accuracy log.

## Measured costs and resources

- Clean setup: 151 seconds, CPU-only, including offline package installation and the local LoadGen build.
- One-case preprocessing: 49 seconds with 8 CPU processes; its checksum matched the repository manifest.
- One-case GPU run plus evaluator: 42 seconds; model inference was 8.79 seconds for 50 sliding-window subvolumes.
- GPU immediately before preflight: 74,573 MiB free on the NVIDIA A100 80GB. The wrapper requires at least 24,000 MiB and checks again immediately before every run.
- Machine resources observed: 8 CPU cores and about 1.8 TiB available host memory.
- The bounded evaluator completed and printed mean, kidney, and tumor metrics. LoadGen reports the known multiple-config-file submission-validity error because documented `run.py` loads both `mlperf.conf` and `user.conf`; accuracy logging and evaluation complete normally.

## Schedule and projection

1. Launch the delivered wrapper with `--from setup`, no case limit, so the accepted full result starts from nothing built and overwrites the bounded artifacts itself.
2. Use all 8 available CPU cores for preprocessing, then check current GPU memory and run the sole accelerator workload serially.
3. Evaluate with 8 CPU processes and validate every required output and log prefix.

Linear one-case scaling gives about 35 minutes, but the prior-trial clean 43-case pass on this same arm completed in under 10 minutes because setup is fixed-cost and preprocessing/evaluation parallelize. Allow 15 minutes for the full pass under current shared-GPU contention. There is no other target or independent ready CPU work to overlap. Reserve 10 minutes before the one-hour limit for `report.json`, output checks, and wrapper portability review.

## Completed-run re-plan

- Trigger: the full pass completed, providing exact rather than projected stage costs.
- Measurement: setup took 53 seconds, full preprocessing took 2 minutes 54 seconds, and the GPU run plus evaluator took 5 minutes 57 seconds; total clean-wrapper time was 9 minutes 44 seconds.
- Outcome: all 43 cases completed, all preprocessing checksums matched, and every required artifact and metric validated. No schedule or implementation change was needed.
