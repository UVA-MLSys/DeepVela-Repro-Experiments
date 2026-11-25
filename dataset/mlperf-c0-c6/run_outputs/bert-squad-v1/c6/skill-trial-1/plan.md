# BERT-SQuAD reproduction plan

## Workload

One target, `bert-squad`: build an isolated Python/LoadGen environment, stage the
repository BERT harness with the supplied read-only inputs, run the unchanged
PyTorch FP32 Offline accuracy path over the full SQuAD v1.1 validation feature
set, and invoke the repository's `accuracy-squad.py` evaluator.

Dependencies are strictly ordered: `setup` -> `prepare` -> `run` -> evaluator.
The delivered no-stage invocation performs this sequence and writes a success
marker only after evaluation succeeds.

## Measured costs

- Clean eight-sample end-to-end preflight: 140.23 s wall time, 2,242,992 KiB
  maximum host RSS. This included environment construction, LoadGen compilation,
  hashing the fixed inputs, generation of the complete 51 MiB feature cache,
  model load, eight GPU samples, and evaluation.
- Eight-sample LoadGen interval: 5.156 s; this is warm-up dominated.
- Cached 100-sample bounded run: 34.11 s end-to-end, 2,236,640 KiB maximum host
  RSS, with a 9.949 s LoadGen interval (10.05 samples/s).
- Full QSL size established by the repository/configuration: 10,833 features.
- GPU memory immediately before the 100-sample run: 73,322 MiB free of 81,920
  MiB. The run wrapper requires at least 12,000 MiB free before launching.

## Resources and schedule

Use the local NVIDIA A100 80GB directly. Run exactly one owned GPU pass at a
time. There is only one target, so no other target's CPU stage can overlap it.
The clean full run starts with `--from setup`, which removes only wrapper-owned
state and then executes all stages. GPU availability is checked again directly
before the run.

The 100-sample steady-state rate projects 10,833 samples to about 1,078 s
(18.0 minutes). Adding approximately 2 minutes for the measured clean setup,
preprocessing, model load, and evaluator gives a roughly 20-minute full-run
projection. Hold 10 minutes in reserve for artifact validation and
`report.json`; this fits comfortably inside the two-hour wall-clock limit.

## Execution update

At the first few percent, observed throughput remained near the planned 9--10
samples/s, so the schedule did not change. As other users' GPU contention eased,
throughput improved and the completed clean run took 900.92 s end-to-end with
2,243,100 KiB maximum host RSS, earlier than the roughly 20-minute projection.
