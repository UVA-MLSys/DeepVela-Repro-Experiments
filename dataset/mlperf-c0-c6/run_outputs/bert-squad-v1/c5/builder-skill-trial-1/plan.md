# Full reproduction plan

Written after successful bounded preflight on 2026-11-22.

## Workload and dependencies

Single target `bert-squad`, executed by the delivered `reproduce.sh` in this order:

1. `env` — stage the pinned source, repair only the two evidenced packaging gaps, create the
   Python 3.11 environment from the offline shared uv cache, and compile supplied LoadGen.
2. `prepare` — convert the fixed SQuAD v1.1 validation set into 10,833 BERT features.
3. `preflight` — check current GPU free memory and repeat the bounded two-feature LoadGen and
   official-evaluator test.
4. `gpu` — check current GPU free memory again, select a conservative batch (16 at >=24 GiB,
   8 at >=12 GiB, 1 at >=6 GiB), and run one complete FP32 Offline accuracy pass.
5. `evaluate` — post-process the full LoadGen accuracy log and run `evaluate_v1.1.py` over all
   10,570 validation questions.

The full replay command is:

```bash
OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from env
```

`--from env` makes the script clear and recreate its own build products and outputs; it is the
first full run from nothing built under the deliverable conditions.

## Measured costs and resources

- Environment build: about 101 seconds from the clean stage's first logged timestamp to its
  success marker. LoadGen 6.0.17 compiled successfully. The first failed attempt and dependency
  diagnosis are not part of this clean-stage estimate.
- Preprocessing: about 64 seconds; output count was 10,833 features.
- Bounded end-to-end preflight: about 28 seconds total (GPU check at 14:44:35, LoadGen output at
  14:44:52, evaluator output at 14:45:02). It loaded the full 1.34 GB checkpoint, processed two
  features, and returned finite EM/F1. The two-sample measurement is dominated by imports/model
  loading, so it is evidence of correctness and fit, not a linear throughput estimate.
- GPU availability at bounded launch: NVIDIA A100-SXM4-80GB, 81,920 MiB total and 69,273 MiB
  free; batch 16 selected. Driver 595.71.05. The script rechecks immediately before each GPU run.
- Host availability: 8 visible CPU cores, 2.0 TiB RAM total, 1.9 TiB available at planning time.
- Full input: 10,833 features for 10,570 questions. `user.conf` sets the documented Offline
  target to 32 samples/s, which corresponds to 338.5 seconds for 10,833 features if achieved.

## Schedule and projection

There is one target and one GPU, so accelerator work is serial. Environment build and feature
preprocessing precede the accelerator because both are dependencies. No independent CPU target is
ready during the full GPU stage; evaluation must wait for its accuracy log. The script therefore
runs the five stages sequentially and uses all 8 visible CPUs only for build/preprocessing.

Projected clean replay time is 10 minutes at the configured 32 samples/s, including the measured
build/preprocessing/preflight costs and roughly two minutes for full post-processing. To account
for shared-GPU interference and filesystem variance, the launch budget is 35 minutes. This is
below the two-hour limit after reserving 20 minutes for evaluator/report generation, artifact
hashing, and final standalone-script checks.

The main scheduling decision is batch 16, driven by the measured 69,273 MiB free GPU memory and
the script's conservative >=24 GiB threshold. If the full-run preflight observes less memory, the
script automatically reduces only batch size (not model, precision, data, scenario, or evaluator).

## Re-plan triggers

- If GPU free memory falls below 6 GiB at launch, the run stops rather than risking another
  session's allocation; the measured failure is reported.
- Shared-GPU contention is assessed from the run log/process and GPU telemetry. If projected
  completion exceeds 100 minutes (the two-hour limit minus the 20-minute reserve), no new run is
  started; available measured results and failure evidence are written instead.
- Any full-run failure returns to its failed stage through `reproduce.sh --from <stage>` after one
  evidence-backed repair. The same failure type is not retried more than three times.

## Re-plan 1 — first measurable full-run fraction

At 2026-11-22 14:52:46-05:00, the growing full `mlperf_log_accuracy.json` was 12,192,562
bytes. The bounded log used 12,383 bytes for two samples, projecting roughly 67 MB for 10,833
features; therefore about 18% of the full output had been written after approximately four
minutes of GPU-stage wall time. The A100 was at 100% utilization with several unrelated jobs, but
the run retained ample memory (its process used about 2.2 GiB and the GPU still had more than 65
GiB free).

The contention-adjusted projection is about 22 minutes for the GPU stage and 27 minutes for the
whole clean replay. This remains inside the 35-minute launch budget and far inside the 100-minute
stop threshold, so stage order, batch size, and reserve are unchanged.

## Completion

The clean replay ran from 14:46:23 to 15:04:54-05:00, about 18 minutes 31 seconds. LoadGen's
detail log records 915.974 seconds (15 minutes 16 seconds) between the full pass's power begin and
power end events. All 10,833 accuracy samples completed with no LoadGen warnings or errors, and
the evaluator stage finished at 15:04:54. The result and output hashes are recorded in
`report.json` and the final validation output.
