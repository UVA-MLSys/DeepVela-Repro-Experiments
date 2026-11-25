# Full reproduction plan

## Workload

One target, `3d-unet`, runs the delivered wrapper unchanged from a clean state through four stages: offline environment/LoadGen build, 43-case preprocessing, one local A100 Offline AccuracyOnly inference, and the repository evaluator. Each stage writes a success marker consumed by the next.

## Measured costs

- Current one-case preflight: preprocessing 16.26 s and 2,564,012 KiB maximum RSS; GPU run 16.82 s and 3,072,800 KiB maximum host RSS, with 7.59 s measured inference; evaluation 5.30 s and 2,253,132 KiB maximum RSS.
- Same-contract prior full run: preprocessing 1:46.66 and 3,476,588 KiB maximum RSS; GPU run 3:10.00 and 11,447,836 KiB maximum host RSS; evaluation 1:32.36 and 9,066,116 KiB maximum RSS.
- The offline environment build completed successfully from the mandated shared uv cache in preflight; local LoadGen compilation dominated that stage at about 1.5 minutes.

## Resources and schedule

The host exposes 8 CPUs, about 2 TiB RAM, and one shared NVIDIA A100-SXM4-80GB. Immediately after preflight, 68,931 MiB GPU memory was free. The wrapper checks free memory again immediately before inference and requires at least 20,000 MiB. CPU stages use 8 processes where supported. The stages run in dependency order; there is no second target whose CPU work can overlap the single GPU stage, and no GPU work overlaps it.

## Expected time and reserve

Prior full-stage measurements plus a fresh environment rebuild project under 10 minutes. Allowing shared-filesystem and GPU contention, budget 15 minutes for the full clean run. This leaves more than 40 minutes of the one-hour limit for one repair/replay if necessary and for report/artifact validation. The prior full timings drive this estimate; the current preflight confirms compatibility and current-device execution.

## Re-plan triggers

After early full-run samples, compare per-case inference with the prior 3:10 total. If projected completion exceeds 45 minutes from launch, record the new projection and preserve at least 10 minutes for evaluation, `report.json`, and validation. A failed stage resumes only through `./reproduce.sh --from <stage>` after a bounded validation of any repair.
