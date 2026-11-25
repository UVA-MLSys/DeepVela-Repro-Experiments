# Reproduction plan

## Workload

One target, `3d-unet`, runs four ordered stages through `./reproduce.sh --target 3d-unet`:

1. `env`: copy the supplied repository into target-local state, add the missing PyTorch SUT
   compatibility modules, create the cached Python environment, and compile bundled LoadGen 6.0.17.
2. `preprocess`: construct a read-only view of the 43 fixed KiTS19 cases (copying case 00185 as
   case 00400), preprocess with four CPU processes, and verify all published checksums.
3. `infer`: check current GPU free memory, then run fp32 TorchScript, PyTorch backend, LoadGen
   Offline AccuracyOnly. This is the only GPU stage.
4. `evaluate`: run the benchmark-owned evaluator and validate/copy the three required outputs.

## Measured costs and resources

Bounded preflight used `case_00000` with all checked manifests reduced consistently.

| Stage | Measured wall time | Measured resources/evidence |
|---|---:|---|
| env | about 71 s | local CPU; 38 cached packages; local LoadGen build 32.18 s (`preflight.build.log`) |
| preprocess | 18.8 s | 4 CPU processes; checksum passed; `(1,192,384,384)` volume (`preflight.preprocess.log`) |
| infer | 20.2 s including imports/model load and LoadGen teardown; LoadGen timed region 5.3 s | A100; 78,276 MiB free before launch; about 3 GiB observed GPU use (`preflight.gpu-memory.log`, detail log) |
| evaluate | 6.2 s after compatibility repair | 4 CPU processes; 56,623,153-byte one-case accuracy log; official evaluator completed |

The machine exposes 168 CPU cores and roughly 1 TiB RAM, but preprocessing is held to four workers
to limit shared-machine and per-volume memory pressure. GPU work is serialized within this run.
Other sessions are using the GPU, so the script rechecks free memory immediately before inference
and requires at least 12,000 MiB; the model's measured need is far below that threshold.

## Schedule and projection

Run the delivered script unchanged from a clean full state with `--from env`. Environment and
preprocessing are CPU-only and may overlap GPU work belonging to other sessions. This target's GPU
stage begins as soon as its preprocessing succeeds and runs alone for this reproduction.

A conservative linear bound from the one-case measurements is 14 minutes preprocessing plus 15
minutes inference, while four-way preprocessing and shared import/model overhead make the expected
time 8–15 minutes. Environment plus evaluation add about 2 minutes. Projected total is therefore
10–17 minutes, comfortably inside the remaining wall-clock budget.

Hold a 10-minute reserve for report extraction, output validation, and one evaluator-only repair.
At the first several completed inference cases, compare the per-case rate in the run log with the
15-minute inference projection. If total projection exceeds the available time less reserve, stop
starting new work and report only measured results as required by the no-human-reply contract.
