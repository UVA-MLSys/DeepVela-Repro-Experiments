# Full reproduction plan

Planned 2026-11-19 after a successful one-case bounded run of the delivered `reproduce.sh`.

## Workload and dependencies

Target `3d-unet` has three stages: `setup` copies the supplied source to a private work tree, applies documented fixture repairs, creates Python 3.11 environment, and builds supplied LoadGen; `preprocess` stages the fixed 43-entry inference list (including the work-local `case_00400` duplicate) and generates/verifies preprocessed volumes; `run` checks free GPU memory, executes fp32 PyTorch Offline accuracy inference, and invokes `accuracy_kits.py`. Each stage writes a success marker consumed by the next.

## Measured costs and resources

The one-case preflight used the same wrapper and environment pins:

- setup: about 57 seconds on 8 local CPU cores with a warm mandated shared uv cache;
- preprocessing: about 17 seconds for `case_00000`, using four worker processes and ample host RAM (about 1.8 TiB available at preflight);
- inference: 8.14812 seconds for 50 sliding-window sub-volumes on the local A100;
- evaluator: under 10 seconds;
- GPU check: 62,185 MiB free immediately before the run, despite three unrelated jobs. The wrapper requires at least 20,000 MiB free.

The source input is 3.8 GiB, model input 596 MiB (the used TorchScript is 125 MB), workspace filesystem has 4.4 TiB free, and GPU total memory is 81,920 MiB.

## Schedule and estimate

Run `OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet --from setup` with `REPRO_CASE_LIMIT` unset. This explicitly clears the wrapper's own bounded-run environment, work tree, markers, and scored outputs, then rebuilds from the supplied repository. CPU setup and preprocessing precede the single GPU stage because this is the only target; no second GPU run overlaps it.

A naive case-linear projection is 12.2 minutes preprocessing plus 5.8 minutes inference, but preprocessing uses four workers and case shapes vary. Allow 20 minutes total for the full run. Reserve 10 minutes before the one-hour limit for evaluator completion, artifact validation, `report.json`, and a standalone clean-copy replay check. Re-plan after the first several full-run cases if the observed inference rate projects beyond that reserve.
