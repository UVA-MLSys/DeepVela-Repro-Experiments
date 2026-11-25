# Reproduction plan

Planned 2026-11-23 after the required bounded preflight.

## Workload and dependencies

One target, `bert-squad`, runs in three logical stages: (1) CPU-only staging, Python environment
creation, and local LoadGen build; (2) GPU preflight/full inference after input validation and an
immediate GPU-memory check; (3) CPU evaluator over the LoadGen accuracy log. The full deliverable
path is `OUTPUT_DIR=$PWD/bert-squad/out ./reproduce.sh --target bert-squad --from setup`, so it
starts from nothing built and exercises the delivered wrapper unchanged.

## Measured costs and resources

- Setup completed in approximately 52 seconds on this host, including package installation from
  the required shared uv cache and the local LoadGen build. It used CPU only.
- Eight-example preflight completed end to end in 78 seconds. Its LoadGen detail log measured
  1.48 seconds for the eight serial GPU feature inferences (~0.185 s/feature); preprocessing and
  checkpoint/evaluator startup dominate the remainder.
- The complete preprocessing cache contains 10,833 features (53,468,824 bytes). Linear inference
  projection is about 2,004 seconds (33.4 minutes); this is the scheduling driver.
- At preflight launch, the A100 had 73,828 MiB free of 81,920 MiB. The wrapper requires at least
  4,096 MiB immediately before each run. The host exposes 128 CPUs and about 1.9 TiB available RAM.

## Schedule and expected time

Run the single target serially. No other target has ready CPU work to overlap. The clean replay
rebuilds setup, regenerates preprocessing, performs the sole GPU run, and immediately evaluates.
Expected total is 35-40 minutes from the measurements; budget 45 minutes to absorb shared-GPU
contention. After the first few percent, compare elapsed inference rate with the ~0.185 s/feature
projection. A material overrun triggers a re-plan, not a semantic change.

## Reserve

Hold 15 minutes before the two-hour wall-clock limit for evaluator completion, exact extraction
into `report.json`, output/log validation, and a final clean-deliverable inspection. No reduced
run may substitute for the full 10,833-feature accuracy run.

## Progress check

At approximately one minute into full inference, `mlperf_log_accuracy.json` contained 577 of
10,833 feature records (5.3%), projecting roughly 19 minutes for inference. This is faster than
the conservative 33.4-minute preflight projection and does not trigger a schedule change.
