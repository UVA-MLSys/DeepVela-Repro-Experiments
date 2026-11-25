# Reproduction plan

Updated: 2026-11-19 00:07 EST

## Workload

One target, `3d-unet`, with four ordered stages: build environment and repair missing SUT adapter; preprocess and checksum all 43 manifest entries; run FP32 PyTorch/Offline/AccuracyOnly inference through LoadGen; run `accuracy_kits.py` and materialize the three required output files.

## Measured costs

- Environment: Python 3.11 environment resolved and installed 39 pinned packages in about 13 seconds; first CUDA/PyTorch import added about 60 seconds. Host has 128 logical CPUs and about 1.8 TiB available RAM.
- Preprocessing: representative `case_00000` produced a 141.6 MB pickle in about 5 seconds and exactly matched the published MD5 (`470ec60377be417a9c271ffd81c1d610`). Full preprocessing will use 8 workers and still run the repository checksum verifier.
- GPU inference: representative `case_00000` took 8.54 seconds through LoadGen, including model warm-up, with batch size one. Its input shape was `[1,192,384,384]`. Before the repaired preflight the A100 had 78,100 MiB free; the run needed only a small fraction of the 80 GB device.
- Evaluation: one case completed in about 6 seconds. Full evaluation is expected to remain below several minutes.

## Resources and schedule

Build and eight-way preprocessing use CPU only. After preprocessing succeeds, the script checks current free GPU memory and requires at least 12,000 MiB, then performs exactly one local GPU run. No Slurm job is submitted. There is only one target, so no other ready CPU stage remains to overlap with its GPU stage.

The 43 volumes vary in size, so the one-case rate is only a lower-bound estimate. Allow 5 minutes for build/preprocessing, 5–15 minutes for inference, and 1–3 minutes for evaluation: projected total 11–23 minutes. The working allowance is 25 minutes.

## Reserve

Hold 10 minutes before the one-hour limit for `report.json`, artifact validation, and a clean-replay check. If measured inference reaches 25 minutes, the projection remains within the 50-minute run budget; if it projects beyond that, stop starting new work and report measured evidence.

## Re-plan record

- **2026-11-19 00:21 EST — LoadGen configuration repair.** The first full run completed and measured the expected accuracy, but LoadGen's detail log contained `error_invalid_config: Multiple conf files are used`. Inspection of the supplied 6.0.17 source showed that `mlperf.conf` must be loaded with `conf_type=0`, after which the single `user.conf` is loaded normally. The generated work copy and `reproduce.sh` were repaired accordingly, then the script was resumed with `--from run`. The second run reported `No errors encountered during test.` and reproduced the same accuracy values. Projection remained below the original 25-minute allowance, so no schedule/resource change was needed.
