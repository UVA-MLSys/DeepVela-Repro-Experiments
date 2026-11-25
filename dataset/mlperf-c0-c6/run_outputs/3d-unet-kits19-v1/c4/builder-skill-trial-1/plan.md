# Full-run plan

Written 2026-11-18 after the required bounded end-to-end preflight.

## Workload

One target, `3d-unet`, runs these dependency-ordered stages through the delivered
`./reproduce.sh --target 3d-unet`:

1. `environment`: copy the fresh supplied source, apply the two semantics-preserving repairs,
   create the Python environment from `UV_CACHE_DIR`, and build the supplied LoadGen.
2. `preprocess`: stage 43 read-only raw case directories (mapping `case_00400` to
   `case_00185`), preprocess with 8 CPU workers, and verify every pickle against the published
   43-entry checksum manifest.
3. `run`: immediately query GPU free memory, then run the PyTorch FP32 Offline accuracy command
   once on CUDA and retain LoadGen logs.
4. `evaluate`: invoke the benchmark evaluator over all 43 predictions, retain its exact stdout,
   and validate record count, QSL indices, and finite printed metrics.

## Measured costs

The consistent bounded run used `case_00005`, the largest evaluation case by raw byte size
(314,518,686 bytes), with a one-entry case manifest and its corresponding published checksum.
Its evidence is in `3d-unet/preflight_out/*.log`.

| Stage | Preflight wall time | Measured resource evidence |
|---|---:|---|
| Environment | 54 s | local compile; offline cache; 8 host cores |
| Preprocess + checksum | 27 s | 8 CPU workers; 287 MiB output for the largest raw case |
| GPU run | 20 s stage / 7.58 s inference | A100; 108 windows; 78,276 MiB free before launch |
| Evaluator | 14 s | 8 CPU workers; 99 MiB output directory including the accuracy log |

The logical raw scope is 3.818 GiB over 43 IDs (including the duplicated case's second logical
weight). The host currently exposes 8 CPU cores, 2.0 TiB RAM with 1.8 TiB available, 4.8 TiB free
workspace storage, and one shared 80 GiB A100.

## Schedule and expected time

There is one target, so the stages run in dependency order. Preprocessing uses all 8 exposed CPU
cores. No other GPU work is launched, and the script refuses to start its GPU stage below 20,000
MiB free. The measured largest-case rate bounds individual-case GPU memory and gives a conservative
inference projection of `43 * 7.58 s = 326 s`; the actual sum should be lower because case_00005 is
the largest raw case. Scaling CPU/data work conservatively and allowing for shared-filesystem I/O
gives:

- environment: 1 minute;
- preprocessing and hash verification: up to 8 minutes;
- GPU inference and LoadGen accuracy-log writing: up to 10 minutes;
- postprocessing/evaluation: up to 12 minutes.

Expected total: at most 31 minutes. This fits the one-hour limit while leaving a 15-minute reserve
for reporting, artifact validation, and one in-scope repair. The remaining 14 minutes are schedule
slack for contention. The full run will start from the untouched `work-full` path, so none of the
preflight environment or derived data is reused.

## Re-plan triggers

After preprocessing and after the first logged inference samples, compare elapsed time and the
per-case rate with the 31-minute projection. If projected completion exceeds 45 minutes from full
run launch, record the trigger and revised projection here before changing any safe parallelism.
At the 45-minute point, launch no new work; preserve completed measured artifacts and use the
15-minute reporting reserve.

