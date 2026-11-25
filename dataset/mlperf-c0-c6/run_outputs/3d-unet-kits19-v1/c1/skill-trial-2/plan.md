# 3D-UNet KiTS19 reproduction plan

## Workload

Target `3d-unet` runs three ordered stages: build a Python 3.11 environment and the
repository LoadGen binding; preprocess and checksum-verify the 43 cases in
`meta/inference_cases.json`; run the fp32 TorchScript model with the PyTorch backend in
Offline AccuracyOnly mode and evaluate the LoadGen accuracy log with `accuracy_kits.py`.

## Preflight measurements

- Setup resolved 33 cached packages immediately, installed them in 5.64 seconds, and built
  LoadGen in 33.11 seconds. Allow 2 minutes for setup and imports on the shared filesystem.
- One representative case (`case_00000`, preprocessed shape 192x384x384) took 47.05 seconds
  to preprocess and checksum-verify, with 2.56 GB peak RSS. Eight preprocessing workers are
  used; a conservative full projection is 43 / 8 * 47.05 = 253 seconds plus 2 minutes of
  shared-filesystem overhead.
- The same case required 50 sliding-window subvolumes and 7.55 seconds of GPU inference.
  The complete run-and-evaluate stage took 58.29 seconds with 3.06 GB peak RSS. Treating
  this substantial case as the per-case rate gives 43 * 7.55 = 325 seconds, plus 2 minutes
  for model loading, LoadGen, log I/O, and evaluation.
- Immediately before preflight, the A100 had 81,037 MiB free. The runner requires at least
  16,000 MiB free before starting GPU work.

## Schedule and limit

There is one target, so setup, preprocessing, GPU inference, and evaluation run in order.
No other GPU work from this reproduction overlaps the inference stage. The measured
projection is approximately 2 + 6.2 + 7.4 = 15.6 minutes. The run therefore fits within
the one-hour limit while retaining a 10-minute reserve for artifact validation and
`report.json`.

## Re-plan after full preprocessing

The completed preprocessed NIfTI headers contain 2,761 sliding-window subvolumes across
43 cases. At the preflight rate of 7.55 seconds per 50 subvolumes, projected full GPU
inference is 417 seconds. Setup and preprocessing completed faster than the conservative
estimate, so the updated total projection is about 12 minutes. The one-at-a-time GPU
schedule and 10-minute closing reserve are unchanged.

## Actual

The unchanged clean-copy invocation completed all three stages and evaluation in 6 minutes
38 seconds. Peak process RSS was 11.46 GB. All 43 official preprocessing checksums matched,
and the evaluator completed over 43 cases plus its aggregate `mean` row.
