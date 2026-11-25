# Reproduction plan

Planned: 2026-11-19 after the bounded end-to-end preflight.

## Workload and dependencies

The single `3d-unet` target runs four stages in order: environment build (`env`), exact 43-entry
inference-list preprocessing (`data`), one local A100 Offline FP32 accuracy run (`gpu`), and the
benchmark-owned Dice evaluator (`evaluate`). Each stage consumes the preceding stage's files and
writes a success marker only after all commands succeed.

## Measured costs

The one-case preflight measured 141 s for `env`, 26 s for `data`, 20 s for `gpu`, and 9 s for
`evaluate`. It verified the reduced manifest checksum, loaded the supplied TorchScript model on
`cuda:0`, produced LoadGen logs, and completed the official evaluator. The hash-verified prior
same-arm 43-entry run measured 64 s, 124 s, 287 s, and 134 s for the same four stages,
respectively (609 s total).

## Resources and schedule

Run directly on this machine, never through Slurm. The accelerator stage is serialized because
there is one target and the GPU is shared. Immediately before that stage, `reproduce.sh` records
`nvidia-smi` and requires at least 20,000 MiB free; preflight observed 78,277 MiB free on the
NVIDIA A100-SXM4-80GB. Environment setup and preprocessing use local CPU/storage only. With one
target there is no independent ready CPU work to overlap with its GPU stage.

Launch the full clean replay with `OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet
--from env`; the wrapper clears and recreates only its target-owned stages, so this exercises the
delivered script from nothing built while preserving the workspace contract files.

## Expected time and reserve

Budget 15 minutes for the full run: the prior full-scope measurement is about 10 minutes, with a
5-minute contention allowance. Hold 10 additional minutes inside the one-hour task limit for
report extraction, output validation, and a final clean-replay audit. The full run is projected to
finish comfortably before that reserve.
