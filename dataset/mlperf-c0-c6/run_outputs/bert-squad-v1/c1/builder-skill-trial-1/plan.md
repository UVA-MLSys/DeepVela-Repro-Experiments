# Full reproduction plan

Written after successful bounded preflight on 2026-11-22.

## Workload

One target, `bert-squad`, with three script stages:

1. `environment`: validate fixed inputs, stage source without modifying `repo/`, create Python 3.10 environment from the offline uv cache, compile pinned LoadGen source, and record input hashes.
2. `preflight`: load BERT-Large on the A100, construct all validation features as required by the source, infer/evaluate 8 feature samples, and verify the device log prefix and evaluator JSON.
3. `run`: repeat the prescribed Offline AccuracyOnly route over all 21,120 tokenized features derived from 10,570 SQuAD questions, then invoke the benchmark-owned evaluator and preserve required outputs.

The full-run command is the delivered launcher from a clean script-owned state:

```bash
OUTPUT_DIR="$PWD/bert-squad/out" ./reproduce.sh --target bert-squad --from environment
```

## Measured costs

| Stage/work | Wall time | Resource evidence |
|---|---:|---|
| Clean environment/install/LoadGen build | about 75 s | `bert-squad/out/build.log`; 128 host CPU cores and 1.9 TiB available RAM observed before setup; install is offline from fixed uv cache |
| Bounded preflight (8 features, including first feature construction and evaluator conversion) | 169 s | `bert-squad/out/preflight/timing.log` |
| Bounded second replay (8 features, including evaluator conversion) | 92 s | `bert-squad/out/timing.log` |
| GPU inference within bounded replay | 31.25 ms / 8 features = 3.91 ms/feature | `bert-squad/out/mlperf_log_detail.txt`, `generated_query_duration` and `generated_samples_per_query` |
| Projected full inference | about 83 s for 21,120 features | linear projection from the bounded LoadGen measurement |

Immediately before the last bounded GPU replay the A100 had 75,576 MiB free; after preflight, concurrent usage increased and 71,936 MiB remained free. Both are well above the BERT-Large fp32 requirement. (`bert-squad/out/run.log`; post-preflight `nvidia-smi` observation.)

## Resources and schedule

- Local machine only: 128 CPU cores, approximately 1.9 TiB available host RAM, one shared NVIDIA A100-SXM4-80GB.
- The single target's GPU work runs serially. The script checks free GPU memory immediately before both preflight and full inference.
- There is no second target or independent ready CPU workload to overlap. Environment compilation, feature conversion, inference, and evaluation therefore follow their data dependencies in order.
- Shared-GPU contention is handled by the immediate pre-run memory check and by retaining the source's batch-one execution, which has ample headroom at the measured free-memory level.

## Expected time and reserve

Expected clean replay: approximately 6–8 minutes (75 s environment + 169 s preflight + roughly 3 minutes full inference/conversion/evaluation, allowing shared-filesystem variance). Hold at least 15 minutes of the two-hour budget for report extraction, artifact validation, and a final standalone-script check. The projection is far below the available wall-clock budget.

## Re-plan triggers

- At early full-run progress, compare feature rate to 3.91 ms/feature. If projected completion exceeds the reserve boundary, record the changed projection here before changing scheduling.
- On a new failure type, repair it once in the delivered script and rerun using `--from` at the producing stage.
- Do not start another GPU process while this target's GPU stage is active.

## Re-plan 1 — 2026-11-22 14:02 EST

**Trigger:** The first clean replay printed `run stage already complete` after rebuilding and rerunning preflight. Inspection showed that `reset_from` used Bash `;;&`, which re-tested the same stage name and therefore did not cascade from `environment` into downstream cleanup.

**Change:** Replace both `;;&` terminators with unconditional fall-through `;&`; restart the delivered command with `--from environment`. This removes only the script-owned environment, stage markers, and enumerated outputs before rebuilding.

**Projection:** The successful clean environment plus cached preflight took about 2 minutes. Repeating them and adding the projected 2–3 minute full stage gives about 5 minutes from restart, still leaving far more than the 15-minute reserve.

## Re-plan 2 — 2026-11-22 14:09 EST

**Trigger:** During full inference, a read-only process/GPU check found another session's BERT process sharing the A100. Our process (PID 66909 at observation) and the unrelated process (PID 71336) each held about 1.8 GiB; GPU utilization was 100% with 77,377 MiB still free. Our inference had run about 4.6 minutes, exceeding the uncontended 83-second estimate.

**Change:** Keep the one session-owned GPU run active and do not alter the unrelated process. Extend the shared-contention projection to roughly 8–12 minutes. No memory/batch change is needed because free memory remains ample and the benchmark semantics stay fixed.

**Projection:** Completion remains comfortably before the two-hour limit and leaves the 15-minute reporting reserve intact.

## Re-plan 3 — 2026-11-22 14:18 EST

**Trigger:** The full-scope evaluator produced EM/F1 successfully, but its log showed `Loading cached features from .../bert-squad/full/eval_features.pickle`. Those evaluator caches were created by the earlier bounded rehearsal and were outside `OUTPUT_DIR`, so `--from environment` had not cleared them.

**Change:** Make downstream cleanup remove the explicit script-owned `bert-squad/preflight` and `bert-squad/full` derived-cache directories. Replay from `environment` so the accepted run begins with no built environment, staged source, feature cache, stage marker, or benchmark output.

**Projection:** The contented measured run stage took 802 seconds. A final clean replay including environment and feature conversion is expected within 20 minutes under similar contention, still leaving over an hour of the wall-clock budget and the reporting reserve.

## Re-plan 4 — 2026-11-22 14:30 EST

**Trigger:** Shared occupancy rose to eight GPU processes during the final full pass. Our process had run 365 seconds at observation; the A100 was 100% utilized with 63,686 MiB still free.

**Change:** Continue the already-progressing session-owned batch-one run. Do not interrupt unrelated jobs or change batching/model semantics. Extend the completion range to 20–35 minutes under the observed contention.

**Projection:** Even the upper estimate completes before the wall-clock limit with the reporting reserve intact; GPU memory remains safe.
