# Full reproduction plan

## Workload

Target `bert-squad` runs three ordered stages: cached environment/source setup, full SQuAD v1.1 feature preprocessing, and one local-GPU Offline accuracy run followed by the repository evaluator. The GPU stage depends on the first two stages and is the only accelerator work.

## Measured costs and resources

- Clean bounded execution (all stages, 8 feature windows): 2:46.89 wall time, 2.20 GiB peak host RSS.
- Full preprocessing produced 10,833 windows of length 384.
- LoadGen execution for 8 windows: 1.896 seconds, or 0.237 seconds/window, including bounded-run overhead.
- Preflight GPU check: A100 80 GB with 75,684 MiB free immediately before launch. The fp32 model loaded successfully on `cuda:0`.
- Available execution resource: this machine's one shared NVIDIA A100 80 GB; no scheduler. GPU work is serial. CPU setup and preprocessing precede it because this task has one target and the GPU stage consumes their artifacts.

## Schedule and projection

Run `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad` once with no pre-existing target build. The measured accelerator rate projects `10833 / 8 * 1.896 = 2,567` seconds (42.8 minutes). Allowing about 3 minutes for cached setup, compilation, preprocessing, model loading, and evaluation gives a 46-minute expected run; budget 60 minutes for shared-GPU variability.

Reserve 15 minutes before the two-hour limit for parsing the evaluator's exact printed values, writing `report.json`, and validating required files and the standalone-script contract. Re-plan if early full-run rate projects beyond 105 minutes total or if GPU memory falls below the script's 10,000 MiB safety threshold.

## Re-plan at 14:00

Trigger: the first full launch found four unrelated sessions concurrently opening the same fixed LoadGen output paths. The launched process was stopped without touching those sessions. The driver was repaired to write accelerator and evaluator artifacts under a per-process directory and publish them only after successful evaluation; an 8-window bounded rerun validated this behavior and again scored 75.0 exact match / 75.0 F1.

Schedule change: wait for the pre-existing output writers and GPU jobs to drain, then restart this target from the `run` stage using its already-successful clean setup and preprocessing. At the preflight rate, the remaining accelerator work is about 43 minutes plus evaluation. Starting by 14:40 leaves more than the 15-minute reserve before the 15:44 wall-clock limit.

## Re-plan at 14:12

Trigger: the first 1,352 of 10,833 full-run responses completed in about 96 seconds. The sustained rate is approximately 14.1 windows/second; the 8-window preflight had been dominated by initialization and synchronization overhead.

No schedule change is needed. The measured full-run rate projects accelerator completion around 14:23, followed by evaluation, leaving well over one hour for validation and reporting.
