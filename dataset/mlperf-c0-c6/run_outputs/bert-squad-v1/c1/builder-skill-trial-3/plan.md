# Reproduction plan

Recorded 2026-11-22 after the successful bounded preflight.

## Workload and dependencies

One target, `bert-squad`: (1) create a Python 3.11 uv environment and build the
supplied LoadGen source; (2) hash/stage the fixed SQuAD validation data,
BERT-Large fp32 checkpoint, vocabulary, and DeepLearningExamples tree; (3)
preprocess the complete validation split to features; (4) run all 10,833
features in LoadGen Offline AccuracyOnly mode, serial single-sample PyTorch
inference on the local A100; (5) run `accuracy-squad.py`, which invokes the
pinned SQuAD evaluator, then validate required artifacts and write
`report.json`.

## Measured costs and resources

- Initial environment plus local LoadGen build: approximately 45 seconds,
  using CPU/disk and the mandated shared uv cache. Re-resolution after caching
  is about 2 seconds; a clean local LoadGen compile is about 30 seconds.
- Model load plus complete feature conversion before the bounded query:
  approximately 69 seconds. The resulting feature cache is 51 MiB.
- Bounded GPU run: 8 samples in 0.828 seconds according to
  `bert-squad/out/preflight/mlperf_log_detail.txt`, or about 0.1035 seconds per
  sample. GPU preflight recorded 77,485 MiB free immediately before launch.
- Bounded evaluator: approximately 7 seconds and negligible accelerator use.
- Machine capacity observed: 128 CPU cores, about 1.9 TiB available RAM, one
  NVIDIA A100-SXM4-80GB. Other sessions occupied about 3.4 GiB GPU memory; they
  are not controlled by this run. This target intentionally uses one GPU run
  and `MLC_MAX_NUM_THREADS=1` because the pinned SUT issues samples serially.

## Schedule, projection, and reserve

Run the clean setup/preprocessing, then the only full GPU run, then its CPU
evaluator. No second target exists to overlap with GPU work; setup/evaluation
are short and have direct dependencies. Linear projection is
10,833 * 0.1035 s = 1,121 s (18.7 minutes) for inference. Allowing contention,
model load, preprocessing, build, and evaluator overhead gives a 25-minute
expected completion. Hold 10 minutes after completion for parsing exact printed
values, writing `report.json`, and checking the copied-alone script and all
required logs/artifacts. This is comfortably inside the two-hour limit.

## Re-plan triggers

After early full-run progress becomes visible, compare elapsed rate with the
18.7-minute inference projection. If it projects past 100 minutes, preserve the
measured logs/results and begin the reporting reserve rather than launch any
new run. Any failure returns to bounded preflight repair; only processes begun
by this session may be changed.

## Re-plan record

- **19:37 on 2026-11-22 — first few percent complete:** the accuracy log had
  618/10,833 samples after about 64 seconds of inference, or 9.66 samples/s.
  This exactly matches the bounded preflight rate and projects approximately
  18.7 minutes total inference. GPU utilization was 100% with 71,918 MiB still
  free. Schedule and 25-minute overall projection are unchanged.
- **19:56 on 2026-11-22 — completed:** the clean no-stage invocation exited 0
  after 22m17s wall time, inside the 25-minute projection. LoadGen wrote all
  10,833 unique feature responses and the evaluator completed, leaving the
  reporting reserve unused.
