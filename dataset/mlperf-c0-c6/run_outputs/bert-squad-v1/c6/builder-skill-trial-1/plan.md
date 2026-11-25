# Reproduction plan

## Workload

One target, `bert-squad`: build a Python 3.11 environment, stage the pinned source and read-only inputs, preprocess the full SQuAD v1.1 validation set, run BERT-Large FP32 with PyTorch under LoadGen Offline AccuracyOnly, and invoke the supplied SQuAD evaluator. Environment and preparation feed the single GPU run; the evaluator consumes its LoadGen accuracy log.

## Measured costs

- Clean environment resolution/install: about 11 seconds from the successful preflight log; GPU memory: none.
- Input verification and private source staging: about 2 seconds; GPU memory: none.
- First bounded run, including full validation preprocessing, model startup, eight feature inferences, and evaluation: about 103 seconds under a 100%-utilized shared GPU. The resulting feature cache represents the full dataset.
- Cached bounded rerun: about 18 seconds total. Model plus evaluator overhead accounts for most of this small run, so eight samples alone do not provide a stable linear rate under contention.
- GPU memory observed for this process: approximately 1.7 GiB. Immediately before the successful preflight, 73,864 MiB was free.

## Resources and schedule

Run directly on GPU 0 (NVIDIA A100-SXM4-80GB), never through Slurm. Only one GPU process started by this reproduction runs at a time. The script checks free memory immediately before launch and requires at least 8,192 MiB. Environment and preparation are sequential because there is one target and both are short; there is no ready independent CPU target to overlap. Unrelated sessions currently share and saturate the GPU; their processes are not modified.

The full measured run uses the delivered `reproduce.sh --target bert-squad --from environment`, which clears only script-owned artifacts, rebuilds from nothing, preprocesses, runs all 10,570 questions, evaluates, and writes the required output files.

## Projection and reserve

Budget 10 minutes for clean rebuild and preprocessing, up to 75 minutes for inference under observed contention, and 15 minutes for evaluation/output validation. Hold 20 minutes of the two-hour limit for `report.json`, artifact checks, and a launcher replay check. Re-plan after the first visible progress or on any failure; do not start a new full run once the reserve is reached.

## Re-plan at first progress

At 15:09:38–15:09:48, the live LoadGen accuracy log advanced from 307 to 394 feature records: 8.7 features/second with five observed GPU processes sharing the device. Approximately 10.8k total features therefore project to about 21 minutes of inference, plus the already completed build/preprocessing and final evaluation. This is below the original 75-minute inference allowance, so stage order and parallelism remain unchanged.

## Completion

The clean full run completed successfully in 802.18 seconds (13.37 minutes) with maximum recorded host RSS 2,296,500 KiB. LoadGen wrote 10,833 feature records and reported no warnings or errors. The supplied evaluator printed `{"exact_match": 83.68968779564806, "f1": 90.87487229720105}`; the required files and `report.json` passed final consistency checks.
