# Reproduction plan

## Workload

The single `3d-unet` target runs four serial stages through the delivered `reproduce.sh`: create
a Python 3.11 environment and compile local LoadGen; stage and preprocess the canonical 43 cases
with published checksum verification; run fp32 PyTorch Offline accuracy inference on the local
A100 after a free-memory check; and evaluate the LoadGen log with `accuracy_kits.py`. Success
markers are written only after a stage exits successfully.

## Measured costs and resources

The current one-case preflight measured environment/build 179 s, preprocessing 22 s for
case_00000 (shape 1x192x384x384) with four workers and a matching checksum, GPU/loadgen 24 s
(4.64 s reported model work), and evaluation 5 s. The A100-SXM4-80GB had 77,029 MiB free at the
immediate pre-run check and the model was confirmed on `cuda:0`. The filesystem has 5.1 TiB free.

The same-arm prior trial (`prior-artifacts/plan.md`, provenance in
`prior-artifacts/provenance.json`) measured a faster warm-cache environment build and projected
at most 12 minutes preprocessing, 10 minutes inference, and 3 minutes evaluation. The delivered
script uses four CPU workers and requires at least 24,000 MiB GPU memory free before inference.

## Schedule and projection

There is one target, so all GPU work is serialized. The stages depend on each other and no other
ready target exists for useful CPU/GPU overlap. The clean replay starts from `--from env`, which
clears and reconstructs the script's own full-profile environment, work tree, outputs, and
markers. Allow 3 minutes environment, 12 minutes preprocessing, 10 minutes inference, 3 minutes
evaluation, and 2 minutes filesystem variance: 30 minutes total.

Reserve 5 minutes before the one-hour limit to validate required files, check log coverage and
the required prefix, copy exact printed values to `report.json`, and syntax-check the deliverable.

## Re-plan triggers

After preprocessing and during inference, compare observed time with the 30-minute projection.
On failure, repair the responsible stage and resume with `--from <stage>`. If the projection
crosses the reporting reserve, start no new work and report only completed measured results.
