# Full-run plan: 3D U-Net KiTS19

Updated: 2026-11-18 23:25 America/New_York, after successful one-case preflight.

## Workload and dependencies

One target, `3d-unet`, runs three serial stages through the delivered
`reproduce.sh`: (1) build an isolated Python 3.11 environment and compile the
supplied LoadGen snapshot, (2) CPU-preprocess and checksum all 43 inference cases,
and (3) check free GPU memory, run fp32 TorchScript Offline accuracy on the local
A100, then run the repository evaluator on CPU. Each stage writes a success marker;
the accelerator stage depends on the complete preprocessed split.

## Measured costs and resources

- Environment: package installation used the mandated shared uv cache; LoadGen
  compilation took 32.76 s. Allow 2 minutes including imports and filesystem work.
- Preprocessing: case_00000 took 15.94 s and peaked at 2,562,396 KiB RSS using the
  repository path. The full stage uses four workers on the machine's 8 CPU cores.
  Conservative projection: 43 × 16 s / 4 plus filesystem variance = 4-6 minutes;
  allow 10 minutes and roughly 12 GiB host memory.
- Accelerator: immediately before preflight, the A100 had 81,034 MiB free. The
  largest sampled input shape was `(1,192,384,384)`; its 50 sliding windows took
  4.659 s. Whole-process elapsed time was 14.64 s and peak host RSS 3,101,596 KiB.
  Project 43 cases at 5-15 s/model case plus startup and varied volume sizes =
  4-12 minutes. GPU work remains strictly serial and starts only after another
  `nvidia-smi`/PyTorch free-memory check requiring at least 16 GiB.
- Evaluation: the one-case accuracy log was 55 MiB and evaluation completed in
  about 30 s including NIfTI writing. Project 2.4 GiB of accuracy JSON and allow
  5-10 minutes for JSON decoding, NIfTI output, and four-process Dice evaluation.
- Available resources observed at preflight: 8 CPU cores, about 1.8 TiB host memory
  available, 4.8 TiB workspace filesystem free, and one shared A100 80GB with
  81,034 MiB free.

## Schedule and expected time

Launch `./reproduce.sh --target 3d-unet --from setup` from the workspace. This
clears and rebuilds the full-run environment and all full-run outputs, satisfying
the clean-copy path while leaving preflight evidence separate. Stages are serial
because there is one target and preprocessing is an input dependency of inference.
Expected full wall time is 15-30 minutes. No second GPU job will overlap it.

Reserve 10 minutes of the one-hour task budget for evaluator completion,
`report.json`, artifact validation, and a copied-alone static replay check. If the
first several cases project the accelerator stage beyond the remaining time less
reserve, stop launching new work at reserve and report only measured output as the
task requires; no reduced split will be reported as the full target.

## Known non-fatal diagnostic

LoadGen 6.0.17 records `error_invalid_config` because the repository launcher reads
both `mlperf.conf` and `user.conf`, which it calls invalid for an official
submission. The documented command does exactly this, and this target reproduces
accuracy rather than an official submission. The preflight executed the expected
sample, emitted a complete accuracy record, and the evaluator succeeded, so the
route is retained unchanged.

## Re-plan checkpoint: accelerator first few percent

At 2026-11-18 23:31 America/New_York, 1:32 after the inference process started,
the accuracy log had reached 568 MB, GPU allocation was 6,768 MiB, and process
host RSS was about 10.0 GB. Relative to the one-case 55 MB log, the observed rate
projects roughly 6-10 minutes for the accelerator stage. This remains inside the
original 4-12 minute range and well before the 10-minute closeout reserve, so stage
order and parallelism are unchanged.
