# Full reproduction plan

Recorded 2026-11-19 after bounded end-to-end preflight.

## Workload and dependencies

One target, `3d-unet`, runs three staged steps in order: (1) create a pinned Python 3.11 environment and build the included LoadGen, (2) preprocess and checksum all 43 fixed inference cases on CPU, (3) inspect current GPU memory, run the fp32 PyTorch Offline accuracy pass on the local A100, and evaluate the LoadGen log. Each stage writes a success marker only after all its commands succeed.

## Measured costs and resources

The preflight environment stage took 2:56.55 wall time and 0.52 GiB peak RSS. One-case preprocessing took 17.36 seconds and 2.45 GiB peak RSS; its published checksum matched. One-case GPU inference plus evaluation took 20.32 seconds and 2.93 GiB host peak RSS, with 69,271 MiB GPU memory free immediately before launch. The accelerator processed 50 sub-volumes in 5.61 seconds.

This host exposes 8 CPU cores, 2.0 TiB RAM (1.8 TiB currently available), and one shared A100 80GB. At planning time one unrelated process held 6,768 MiB GPU memory. The wrapper requires at least 12,000 MiB free immediately before its GPU stage.

The verified prior same-arm clean run is additional scheduling evidence: environment completed at 02:35:07, 43-case preprocessing at 02:38:25 (3:18), and GPU plus evaluator at 02:42:54 (4:29). Its 43 inference calls totaled 164.25 accelerator seconds.

## Schedule, expected time, and reserve

Run the delivered script unchanged from nothing built with `./reproduce.sh --target 3d-unet --from environment`. Environment, preprocessing, and GPU work are sequential because there is one target and each reads the preceding stage. Preprocessing uses four worker processes; GPU work is singular. No other-session process will be modified.

Expected clean replay time is approximately 11 minutes: 3 minutes environment, 4 minutes preprocessing (allowing shared-filesystem variation over the prior 3:18), and 4.5 minutes inference/evaluation. Hold a 10-minute reserve for report generation, output validation, and standalone clean-copy validation. This fits the one-hour task limit with substantial margin.

## Re-plan triggers

Compare progress after the first few inference cases with the measured/prior rate. If projected completion approaches the reserve, stop launching new work and record measured results. On failure, group by the first causal log line, apply at most the skill-permitted repairs through `reproduce.sh --from <stage>`, and preserve every log.
