# Full reproduction plan

Recorded 2026-11-19 after successful one-case preflight.

## Workload and dependencies

Target `3d-unet` runs `setup -> preprocess -> run/evaluator`. Setup creates an offline Python 3.11 environment and target-local source copy. Preprocessing consumes the fixed 43-entry manifest and read-only NIfTI inputs. The GPU run consumes all preprocessed cases and the supplied fp32 TorchScript model; `accuracy_kits.py` consumes its LoadGen accuracy JSON.

## Measured costs and resources

- One-case preflight setup: 12 s; CPU/filesystem only.
- One-case preprocessing with checksum verification: 16 s using 4 processes; CPU/filesystem only.
- One-case accelerator inference plus evaluation: about 9 s; case inference itself was 7.56 s.
- GPU memory immediately before preflight: 64,263 MiB free of 81,920 MiB; the wrapper requires at least 20,000 MiB.
- Provenance-backed prior full run: setup 26 s, 43-case preprocessing 278 s, and the 43 reported inference case times sum to roughly 163 s. This is scheduling evidence only, not a result for this trial (`prior-artifacts/provenance.json` and prior logs).
- Workspace filesystem: approximately 5.6 TiB free before launch. One-case preflight working storage is 5.4 GiB, dominated by environment and preprocessed artifacts.

## Schedule and expected time

Run `OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet --from setup`, which clears and recreates only the script's work directory and runs the delivered wrapper unchanged. There is one target, so no useful CPU target can overlap the GPU stage. Expected wall time is under 10 minutes: about 30 s setup, 5 minutes preprocessing, 3 minutes inference, and up to 2 minutes evaluation/output I/O. GPU work is serial and begins only after a fresh memory check.

Reserve 10 minutes inside the one-hour limit for output validation, exact-value extraction, `report.json`, and a deliverable replay audit. If the first several inference cases project beyond 40 minutes total, update this plan before continuing; the prior full-run measurements make that unlikely.
