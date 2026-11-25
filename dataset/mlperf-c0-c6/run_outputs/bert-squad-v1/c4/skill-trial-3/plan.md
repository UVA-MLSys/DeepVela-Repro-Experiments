# Reproduction plan

## Workload

- Target `bert-squad`: build a target-local Python 3.11 environment and the pinned
  LoadGen extension; preprocess SQuAD v1.1 dev into 384-token BERT features; run
  BERT-Large fp32 in Offline AccuracyOnly mode on CUDA; decode spans and invoke
  `evaluate_v1.1.py` for exact match and F1.
- Inputs are the read-only dataset, vocabulary, and PyTorch checkpoint fixed by
  the prompt. The dataset has 48 articles, 10,570 questions, and preprocesses to
  10,833 features.

## Measured costs and resources

- Clean environment resolution used the shared uv cache; local LoadGen compilation
  took about 29 seconds.
- Preprocessing took about 46 seconds and is CPU-only.
- A 100-feature end-to-end bounded run took 20.33 seconds with 2,256,716 KiB peak
  host RSS. LoadGen measured 2.270 seconds for inference, about 44.1 features/s.
- The local NVIDIA A100-SXM4-80GB had 81,038 MiB free immediately before each
  bounded GPU run. The script rechecks free memory immediately before every GPU run.
- Only one GPU target exists, so GPU concurrency is one. There is no independent
  ready CPU target to overlap with its GPU stage.

## Schedule and expected time

1. Run `./reproduce.sh --target bert-squad --from setup` with
   `OUTPUT_DIR=./bert-squad/out`; this clears only script-owned target-local work,
   then rebuilds and executes all four stages.
2. The measured GPU rate projects 10,833 features to about 246 seconds. Including
   setup, preprocessing, model loading, full log decoding, and evaluation, budget
   7 minutes total.
3. Hold a 10-minute reserve for artifact validation and `report.json`, far inside
   the two-hour wall-clock limit.

## Re-plan during full run

- Trigger: after the full GPU stage began, `nvidia-smi` showed 100% utilization
  with four concurrent BERT processes. Only one is this session's process; the
  other three are separate sessions and are left untouched.
- Change: conservatively multiply the measured 246-second inference projection by
  four, yielding about 16.5 minutes for inference and about 19 minutes end to end.
  No schedule change is useful for the single target, and the revised projection
  remains well inside the two-hour limit and reporting reserve.
- Second trigger: contention later rose to five concurrent GPU processes while the
  target had run for about seven minutes. The revised conservative upper bound is
  about 21 minutes of inference; the uninterrupted single-target schedule remains
  the fastest safe option and still preserves the reserve.
- Third trigger: contention reached six concurrent processes at 11.5 minutes
  elapsed. A linear upper bound is about 25 minutes of inference. Restarting would
  discard valid work, so the run remains uninterrupted with ample deadline margin.

## Completion

- The clean full run finished in 1,263.58 seconds with 2,320,248 KiB peak host RSS.
- LoadGen wrote 10,833 accuracy records and reported Offline AccuracyOnly with an
  effective accuracy sample count of 10,833 and no errors.
- The evaluator printed exact match `83.68968779564806` and F1
  `90.87487229720105`; `report.json` records those exact values.

## Input evidence

- `dev-v1.1.json`: SHA-256
  `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`
- `model.pytorch`: SHA-256
  `71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f`
- `vocab.txt`: SHA-256
  `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`
