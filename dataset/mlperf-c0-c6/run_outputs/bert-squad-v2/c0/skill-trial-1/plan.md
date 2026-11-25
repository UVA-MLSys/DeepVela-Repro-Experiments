# Full reproduction plan

## Workload

Target `bert-squad` has three ordered stages: create the cached Python environment and build the repository's LoadGen extension, preprocess the complete SQuAD v1.1 validation set, then run one local-GPU Offline accuracy pass and the repository-owned evaluator. The run stage is the only accelerator work and depends on the other two stages.

The fixed inputs are `dev-v1.1.json` (SHA-256 `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`), `model.pytorch` (`71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f`), and `vocab.txt` (`07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`). Preprocessing produced 10,833 windows of length 384.

## Measured costs and resources

- Current clean bounded execution used the delivered script and completed all stages in about 2 minutes 34 seconds.
- Setup took about 71 seconds. Full preprocessing took about 58 seconds and covered the complete dataset.
- The bounded run loaded the fp32 model on `cuda:0`, evaluated 8 windows, and completed its accelerator/evaluator stage in about 24 seconds. The benchmark evaluator reported 75.0 exact match and 75.0 F1 for that bounded subset.
- Immediately before the bounded GPU launch, the A100 80 GB had 77,503 MiB free. A subsequent check showed 77,429 MiB free while two unrelated processes used about 3.6 GiB total. The script requires at least 10,000 MiB free and checks again immediately before every GPU run.
- The provenance-matched prior trial observed about 14.1 windows/second early in its full run and completed with the same script. Its steady rate is useful because the 8-window preflight is dominated by model loading and synchronization.

## Schedule and projection

Run `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad` once from an unbuilt target directory. This task has one target, so setup and preprocessing precede the serial GPU stage; there is no independent CPU work left to overlap with it.

The deliberately conservative bounded-rate projection is `10833 / 8 * 1.896 = 2,567` seconds, or about 43 minutes of inference, plus roughly 3 minutes for setup, preprocessing, loading, and evaluation. The prior full-run steady rate instead projects about 13 minutes of inference. Budget 60 minutes for the complete clean run to absorb shared-GPU variability.

Reserve the final 15 minutes of the two-hour limit for evaluator parsing, `report.json`, artifact checks, and the standalone-script contract check. Re-plan if the first few percent project beyond 105 minutes total, or if the launch-time GPU-memory gate fails.

## Re-plan at 12:22

Trigger: the first 562 of 10,833 full-run responses (5.2%) were written about 67 seconds after the GPU stage launched. This is roughly 8.4 windows/second including model and LoadGen startup, despite concurrent unrelated GPU activity.

No schedule change is needed. The observed rate projects inference completion near 12:43, followed by evaluator processing, leaving substantially more than the 15-minute reserve.
