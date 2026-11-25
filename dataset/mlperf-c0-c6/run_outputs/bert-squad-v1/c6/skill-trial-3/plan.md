# BERT-SQuAD reproduction plan

## Workload

One target, `bert-squad`: build an isolated Python environment and repository
LoadGen, stage the pinned BERT source with the fixed read-only SQuAD/model/DLE
inputs, run the PyTorch fp32 Offline accuracy pass, then invoke the repository's
`accuracy-squad.py` evaluator. Each stage writes a success marker and the run
and evaluator stages depend on all preceding markers.

## Measured costs and resources

- Environment creation plus the local LoadGen build took about 50 seconds.
- Input validation/staging took less than 2 seconds. Initial full-dataset
  tokenization completed during the two-sample end-to-end preflight.
- The warm-cache 100-feature run plus evaluator took 33.81 seconds.
- The preprocessing cache contains 10,833 features for 10,570 SQuAD questions.
- Immediately before the measured GPU runs, the shared A100 80GB had 74,718 MiB
  free. The BERT process used about 1.8 GiB, so memory is not limiting; several
  unrelated concurrent BERT jobs make compute contention the main uncertainty.

## Schedule and expected time

Run the target's environment, inputs, GPU inference, and evaluator stages in
order via `reproduce.sh --from environment`. Only this target's GPU stage is
launched, satisfying the one-at-a-time local GPU rule. No other CPU target is
ready to overlap because the manifest contains one target.

Linear extrapolation of the contention-affected bounded measurement gives
10,833 / 100 * 33.81 seconds = 61.0 minutes. This overestimates slightly because
model loading and evaluation are fixed costs included in every bounded pass.
Allowing another 3 minutes for a clean environment build and preprocessing puts
the planned completion at about 64 minutes. Hold 15 minutes of the 2-hour limit
for artifact checks, `report.json`, and any evaluator-only repair.

## Re-plan triggers

After the first several percent of inference, compare elapsed time and LoadGen
progress with the 64-minute projection. If projected completion passes the
105-minute launch budget, preserve completed stage outputs, diagnose the shared
GPU rate, and update this plan before starting any replacement run.

## Re-plan: first progress sample

At 4:08 of the full GPU process, `mlperf_log_accuracy.json` had reached 20.0 MB.
The 100-feature sample produced 0.619 MB, which projects about 67 MB for 10,833
features; the full run was therefore approximately 30% complete. The observed
rate revises inference to about 14 minutes and total clean-run completion to
about 16 minutes. No schedule change is needed, and the 15-minute reporting
reserve remains intact.

## Completion

The clean-copy path completed in 627.10 seconds (10:27), substantially below
the revised projection. LoadGen emitted 10,833 accuracy records with 10,833
unique QSL indices, reported no errors, and the repository evaluator completed
over the full validation set.
