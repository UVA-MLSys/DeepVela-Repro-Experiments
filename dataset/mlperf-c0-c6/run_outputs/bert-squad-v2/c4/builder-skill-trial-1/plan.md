# Full reproduction plan

## Workload

Target `bert-squad` runs four serial stages: clean environment/source setup, CPU preprocessing,
one direct A100 Offline accuracy run over all 10,833 features, then EM/F1 evaluation. Each stage
depends on the preceding stage's files and success marker.

## Preflight measurements

- Setup: about 45 seconds; CPU, filesystem, and package cache; no GPU.
- Preprocessing: about 51 seconds; CPU and filesystem; produced 10,833 features.
- Eight-sample accelerator pass: LoadGen measured 4.607 seconds, or about 0.576 seconds/sample;
  GPU free memory immediately before launch was 72,486 MiB.
- Eight-sample evaluator: about 3 seconds; CPU only; printed EM/F1 successfully.

## Resources and schedule

Use the one local NVIDIA A100 80GB serially, after the script's immediate `nvidia-smi` memory
check. No other target exists, so there is no useful CPU work to overlap with the GPU stage.
Launch the delivered script with `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad
--from setup`, which clears and rebuilds all of its own state before the full run.

## Expected time and reserve

Linear projection from the bounded GPU pass is 10,833 × 0.576 seconds = about 104 minutes.
Allow 2 minutes for clean setup/preprocessing, 2 minutes for evaluation and artifact checks, and
an 8-minute reserve for `report.json` and final validation. Projected total is about 108 minutes,
within the two-hour limit including reserve. The sample inference rate is the scheduling driver.

## Actual outcome and re-plan

The full GPU interval was 873.3 seconds (14.56 minutes) for all 10,833 features, much faster than
the eight-sample linear estimate because startup/warm-up dominated that bounded pass. The run
completed before a schedule change was needed; the remaining reserve was used for evaluator,
scope, log, checksum, script-syntax, and report validation.
