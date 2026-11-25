# Full-run plan

## Workload

Target `bert-squad` has three resumable stages. `build` creates a Python 3.11
environment and builds the pinned repository's LoadGen. `preprocess` converts all
10,570 SQuAD v1.1 validation questions into 10,833 features at sequence length
384. `run` loads the supplied fp32 BERT-Large checkpoint, executes LoadGen in the
default Offline scenario and AccuracyOnly mode on the local A100, and invokes the
repository `accuracy-squad.py` evaluator over the resulting accuracy log. The
evaluator calls `evaluate_v1.1.py` and reports `exact_match` and `f1`.

## Measured costs

The clean eight-example preflight completed end to end in 166.43 seconds. It
resolved and installed 48 cached packages, compiled LoadGen, preprocessed the
entire validation set, loaded the model, ran one batch, and evaluated its bounded
log. The preflight confirmed 10,833 features and printed `pytorch device: cuda:0`.
Immediately before inference, the A100 had 75,120 MiB free of 81,920 MiB, so the
script selected batch 32. A provenance-verified same-arm prior trial measured a
full-run accelerator stage at approximately 7--9 minutes under contention.

## Resources and schedule

Run the sole target in dependency order through the delivered script, beginning
with `--from build` so that no preflight-built state is reused. CPU build and
preprocessing run locally. The accelerator stage begins only after its own fresh
`nvidia-smi` snapshot and runs directly on the one local A100. Batch size is 32
when at least 24 GiB is free and is reduced automatically at lower memory
headroom. There is no second target or independent CPU stage to overlap after GPU
work starts, so only one GPU run is launched.

## Expected time and reserve

Allow 2 minutes for the measured build and preprocessing work, 25 minutes for
model loading and 339 batch-32 inference steps under shared-GPU contention, and 5
minutes for full evaluation: 32 minutes total. Hold 15 minutes for validating
artifacts, writing `report.json`, and auditing the clean-copy deliverable. This is
within the two-hour wall-clock limit.

## Re-plan at first full-run progress

About 1.5 minutes into full inference, `mlperf_log_accuracy.json` had reached 9.5
MiB while the GPU still had 70,620 MiB free. Relative to the expected full log
size, this projects the GPU phase to roughly 10--12 minutes and the whole run to
about 16 minutes. This is below the original 32-minute allowance, so stage order,
single-GPU serialization, and batch size 32 remain unchanged.
