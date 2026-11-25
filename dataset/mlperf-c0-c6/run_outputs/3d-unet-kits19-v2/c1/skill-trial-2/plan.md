# 3D-UNet KiTS19 reproduction plan

## Workload

Target `3d-unet` runs three ordered stages: build a Python 3.11 environment and the
repository LoadGen binding; preprocess and checksum-verify the 43 cases in
`meta/inference_cases.json`; run the fp32 TorchScript model with the PyTorch backend in
Offline AccuracyOnly mode; and evaluate the LoadGen accuracy log with the repository's
`accuracy_kits.py`.

## Preflight measurements

- Setup from the mandated uv cache took 2 minutes 58 seconds during heavy shared-machine
  contention. Package installation itself took 8.60 seconds and the LoadGen build took
  59.64 seconds. Peak RSS was 527 MB.
- One representative case (`case_00000`, preprocessed shape 192x384x384) took 40.09
  seconds to preprocess and checksum-verify, with 2.56 GB peak RSS. Eight preprocessing
  workers are used. A conservative full projection is 43 / 8 * 40.09 = 216 seconds,
  plus two minutes for shared-filesystem overhead.
- The same case required 50 sliding-window subvolumes and 6.98 seconds of GPU inference.
  The complete model-load, LoadGen, and evaluator stage took 28.61 seconds with 3.08 GB
  peak RSS. The prior trial recorded 2,761 subvolumes over all 43 official cases, giving
  a projected inference time of 2,761 / 50 * 6.98 = 385 seconds, plus two minutes for
  model loading, LoadGen log I/O, and full evaluation.
- Immediately before preflight GPU work, the A100 had 68,959 MiB free. The runner checks
  memory again immediately before every GPU run and requires at least 16,000 MiB free.

## Schedule and limit

There is one target, so setup, preprocessing, GPU inference, and evaluation run in order.
No GPU work launched by this reproduction overlaps its inference stage. The measured
projection is approximately 3.0 + 5.6 + 8.4 = 17 minutes under current contention. This
fits the one-hour limit while retaining a 10-minute reserve for output validation and
`report.json`. The clean-copy full run starts with `--from setup`, which clears and
recreates every script-owned stage and runs the delivered script unchanged.

## Re-plan after early full-run progress

All 43 preprocessed files completed and passed repository checksum verification. The
later inference samples ran at roughly 0.08 to 0.09 seconds per subvolume, faster than
the preflight estimate of 0.14 seconds per subvolume. The projected completion therefore
remained inside the original 17-minute estimate, so the one-at-a-time GPU schedule and
10-minute closing reserve were unchanged.

## Actual

The unchanged clean-copy invocation completed setup, preprocessing, all 43 inference
cases, and repository evaluation in 10 minutes 36.82 seconds. Peak process-tree RSS was
11.48 GB. The A100 had 81,037 MiB free immediately before the full GPU run, all 43
official preprocessing checksums matched, and the evaluator printed its aggregate result
after processing the complete LoadGen accuracy log.
