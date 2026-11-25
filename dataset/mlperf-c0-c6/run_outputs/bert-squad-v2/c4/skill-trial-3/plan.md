# Reproduction plan

## Workload

- Target `bert-squad`: create a target-local Python 3.11 environment, build the
  pinned LoadGen extension, preprocess SQuAD v1.1 dev into 384-token BERT
  features, run BERT-Large fp32 with PyTorch in Offline AccuracyOnly mode on the
  local A100, decode spans, and invoke the local `evaluate_v1.1.py` interface.
- Fixed inputs: `dev-v1.1.json` (48 articles, 10,570 questions),
  `model.pytorch`, and `vocab.txt`. Preprocessing produces 10,833 features.

## Preflight measurements and resources

- A clean three-example end-to-end run completed successfully in 115.39 seconds
  with 2,266,880 KiB peak host RSS. It built LoadGen in about 28 seconds and
  preprocessed all 10,833 features in about 45 seconds.
- The evaluator printed exact match 100.0 and F1 100.0 for the bounded subset;
  LoadGen reported no warnings or errors, and the required device line was
  `pytorch device: cuda:0`.
- The A100-SXM4-80GB had 70,774 MiB free immediately before the preflight GPU
  stage. `reproduce.sh` rechecks free memory immediately before every GPU run
  and refuses to launch below 6,000 MiB.
- This machine exposes 8 CPU cores and about 2 TiB host RAM. Other sessions are
  using the GPU and are left untouched. Only one target exists, so GPU
  concurrency is one and there is no independent ready CPU target to overlap.

## Schedule and expected time

1. Run `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from setup`.
   This uses the delivered script, clears only script-owned work, and executes
   setup, preprocessing, GPU inference, and evaluation in order.
2. The prior permitted trial of this byte-identical script and identical input
   hashes completed the clean full run in 1,263.58 seconds under heavy GPU
   contention. Current GPU utilization is already 100%, so budget 25 minutes
   for the full run rather than extrapolating three samples dominated by model
   loading.
3. Hold a 10-minute reserve to validate output scope and hashes and write
   `report.json`. The 35-minute projection is within the two-hour limit.

## Re-plan during full run

- Trigger: shortly after inference began, `nvidia-smi` showed six concurrent
  BERT processes and 100% GPU utilization. Exactly one process belongs to this
  session; it used 1,814 MiB GPU memory and remained healthy at about 99% host
  CPU. The other five processes are left untouched.
- Decision: continue the single target uninterrupted. Six-way sharing matches
  the permitted prior trial's worst contention, whose measured full duration
  was about 21 minutes. Restarting would discard valid work, no alternate GPU
  schedule exists, and the 25-minute run budget plus 10-minute reserve remains
  well inside the two-hour limit.

## Input evidence

- `dev-v1.1.json`: SHA-256
  `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`
- `model.pytorch`: SHA-256
  `71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f`
- `vocab.txt`: SHA-256
  `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`

## Completion

- The delivered script's clean full run completed in 1,240.73 seconds with
  2,264,868 KiB peak host RSS and exit status 0.
- LoadGen wrote 10,833 unique accuracy records covering QSL indices 0 through
  10,832. Its detail log records Offline, AccuracyOnly, an effective accuracy
  sample count of 10,833, and zero warning or error events.
- The evaluator printed exact match `83.68968779564806` and F1
  `90.87487229720105`; `report.json` records those exact numeric values.
