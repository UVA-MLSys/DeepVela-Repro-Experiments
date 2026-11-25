# Full reproduction plan

Written after successful bounded preflight on 2026-11-23.

## Workload

One target, `bert-squad`, with three delivered-script stages:

1. `environment`: validate fixed inputs, stage source locally, create Python 3.10 environment from the offline uv cache, compile pinned LoadGen, and record hashes.
2. `preflight`: check GPU memory, load BERT-Large, construct validation features, infer and evaluate 8 features, and verify the required device log line and evaluator JSON.
3. `run`: execute Offline AccuracyOnly over all tokenized features derived from 10,570 questions, then run the benchmark evaluator and preserve outputs. The bounded run cannot establish the final QSL size; the completed full run reported 10,833 features.

The acceptance run starts from nothing built by the script:

```bash
OUTPUT_DIR="$PWD/bert-squad/out" ./reproduce.sh --target bert-squad --from environment
```

## Measured costs

| Stage/work | Wall time | Evidence |
|---|---:|---|
| Environment/install/LoadGen build | 53 s | `bert-squad/out/build.log` timestamps |
| Bounded preflight (8 features plus full feature construction and evaluator conversion) | 167 s | `bert-squad/out/preflight/timing.log` |
| GPU inference within preflight | 31.25 ms / 8 = 3.91 ms/feature | `bert-squad/out/preflight/mlperf_log_detail.txt`, `generated_query_duration` |
| Initial projected full inference | about 83 s | bounded LoadGen reported an effective configured query size of 21,120; this was a conservative planning value, not the final QSL count |

Preflight started with 77,395 MiB GPU memory free. Afterward, observation found 73,756 MiB free, 128 CPU cores, and about 1.9 TiB available host memory. The explicit wrapper guard requires at least 8,192 MiB free immediately before each GPU run.

## Resources and schedule

- Run locally only, without Slurm, on the shared NVIDIA A100-SXM4-80GB.
- Only one target exists; its GPU stages run serially. No independent ready CPU target exists to overlap.
- Environment build, feature construction, inference, and evaluation follow their file dependencies.
- Preserve source batch-one behavior. Do not touch other sessions' processes; contention may extend runtime but present memory headroom is ample.

## Expected time and reserve

Expected clean replay is 6–10 minutes: about one minute for setup, three minutes for preflight, and two to six minutes for full conversion/inference/evaluation under light contention. Hold 15 minutes of the two-hour limit for report extraction, artifact validation, and final script checks.

## Re-plan triggers

- Compare early full-run behavior with the 3.91 ms/feature inference rate; record any projection that threatens the reserve.
- Repair each new failure type in the delivered script and restart with `--from` at the producing stage.
- Never start a second session-owned GPU process while this target is running.

## Re-plan 1 — 2026-11-23 12:27 EST

**Trigger.** During the full pass, a read-only process/GPU check found six BERT-sized GPU processes. Only PID 153161 belongs to this session; five are unrelated. GPU utilization was 100% with 70,117 MiB still free.

**Change.** Continue the one already-progressing session-owned batch-one run and do not touch unrelated processes. Extend the full-stage estimate from the uncontended projection to roughly 10–25 minutes under observed contention. No memory or batching change is warranted because headroom remains ample and benchmark semantics must remain fixed.

**Projection.** Completion remains comfortably before the two-hour limit and preserves the 15-minute reporting reserve.

## Closeout correction — 2026-11-23 12:49 EST

The completed full LoadGen detail log establishes `qsl_reported_total_count=10833` and `generated_samples_per_query=10833`; the accuracy log contains exactly 10,833 records. The 21,120 value observed in preflight was `effective_samples_per_query` from configuration, not the source-derived QSL size. This correction does not reduce benchmark scope: the evaluator consumed the full 10,570-question validation set and printed both requested metrics.
