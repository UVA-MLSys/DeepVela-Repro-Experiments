# Reproduction plan: 3d-unet

## Workload and dependencies

One target, `3d-unet`, runs through three stages in `reproduce.sh`:

1. `environment`: copy the supplied source, create a Python 3.11 uv environment, install pinned runtime dependencies, build LoadGen, and apply the required device-log patch.
2. `data`: create a read-only symlink view of the 43 inference cases (mapping `case_00400` to `case_00185`) and run the repository preprocessor and checksum verification with four CPU workers.
3. `run`: check current GPU free memory, run PyTorch Offline accuracy serially on GPU, then run the benchmark evaluator with four CPU workers and validate required artifacts.

Each stage writes a success marker. The full run starts from `environment`, removing only script-owned build/output state and exercising the delivered script from nothing built.

## Preflight measurements

- Environment: about 73 seconds from the shared uv cache, including a 35.4-second LoadGen build. All required imports succeeded with Torch 2.4.1 / CUDA 12.1.
- One-case preprocessing plus the first failed checksum-cardinality attempt: preprocessing itself completed in about 17 seconds, used 2,562,452 KiB maximum RSS, and the repaired attempt matched the published checksum.
- One-case repaired data-to-evaluator run: 1:46.79 wall time, 3,055,528 KiB maximum host RSS. The measured model inference was 7.07005 seconds for 50 sliding-window sub-volumes.
- GPU preflight: 74,260 MiB free immediately before launch; this process used about 3,528 MiB while another job held about 6,768 MiB.
- One-case evaluator output: `Accuracy: mean = 0.90515, kidney = 0.9700, tumor = 0.8403` (preflight evidence only, not the reported benchmark result).

## Resources and schedule

The machine exposes 8 CPU cores, about 1.8 TiB available host memory, and one shared NVIDIA A100-SXM4-80GB. The script uses four preprocessing/evaluator workers, leaving CPU capacity for unrelated activity. GPU inference is strictly one run from this session and is launched only after a fresh `nvidia-smi` free-memory check; samples are processed serially by the reference SUT.

The full run is sequential because the sole target's stages depend on one another. Environment creation and preprocessing are CPU-only. No other ready work from this reproduction remains to overlap with GPU inference.

## Expected time and reserve

The one-case preprocessing and inference measurements, combined with 43 heterogeneous volumes and serial sliding-window inference, project approximately 15–25 minutes for the clean full replay. The environment rebuild should take about 1–2 minutes from cache, preprocessing about 3–6 minutes at four workers, GPU inference about 7–15 minutes depending on volume sizes and concurrent GPU load, and evaluation about 2–4 minutes. This leaves at least a 10-minute reserve inside the one-hour task limit for `report.json`, artifact validation, and one bounded repair if required.

## Re-plan triggers

After the first few full-run samples, compare measured per-case time with the 15–25 minute projection. If projected completion exceeds the reporting reserve, reduce only CPU-stage parallel contention; do not change the fixed model, precision, dataset, scenario, preprocessing, or evaluator. On failure, repair the failure class once, rerun with the earliest affected `--from` stage, and record the changed projection here.

## Re-plan 1: early full-run rate

At about 1.6 minutes into the full GPU process, the accuracy log contained approximately 0.63 GB of an expected 2.83 GB of prediction hex data (roughly 20% of output). This projects the inference stage at about 7–9 minutes. The projection remains within the original 15–25 minute total and preserves the 10-minute reporting reserve, so the schedule and parallelism are unchanged.

## Re-plan 2: stdout artifact repair

The first clean full replay succeeded in 9:02.56 and measured `Accuracy: mean = 0.86168, kidney = 0.9347, tumor = 0.7887`. Final artifact inspection showed that `accuracy.txt` contained evaluator stderr warnings in addition to stdout because both streams had been combined. The task defines this file as evaluator stdout. The script was repaired to write stdout alone to `accuracy.txt`, retain stderr in `evaluator.stderr.log`, and combine both only in `evaluator.log`. Because the delivered script changed, repeat the clean full replay from `environment`; the measured 9:03 runtime leaves ample reserve.

The corrected clean replay completed successfully in 11:58.61 under concurrent GPU and CPU activity from an unrelated session. It reproduced the same printed values, validated all 43 records, and confirmed that `accuracy.txt` is stdout-only. No further re-plan is required.
