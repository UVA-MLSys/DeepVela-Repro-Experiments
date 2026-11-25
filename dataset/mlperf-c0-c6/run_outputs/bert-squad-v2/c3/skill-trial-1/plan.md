# BERT SQuAD reproduction plan

## Workload

Target `bert-squad` runs the pinned repository's PyTorch backend in Offline accuracy mode on the supplied BERT-Large fp32 checkpoint and the complete SQuAD v1.1 validation set. The dataset contains 10,570 questions and preprocessing produces 10,833 features. Stages are: build an isolated Python 3.11 environment and local LoadGen extension; prepare the complete feature cache; run LoadGen on the local GPU; run `accuracy-squad.py`, which invokes `evaluate_v1.1.py`.

## Measured costs

- This trial's clean preflight environment/source preparation and LoadGen compilation took 67.56 seconds and used 525,600 KiB maximum RSS.
- Complete CPU feature preparation took 58.91 seconds and used 883,792 KiB maximum RSS, producing 10,833 features in a 53,468,824-byte cache.
- This trial's bounded accelerator pass covered 100 features end to end in 25.56 seconds, including model loading, and used 2,324,504 KiB maximum host RSS. The run reported `pytorch device: cuda:0`; the A100 had 77,377 MiB free immediately before launch.
- This trial's bounded evaluator took 7.24 seconds and used 853,700 KiB maximum RSS.
- The provenance-verified same-arm prior full run took 1,619.10 seconds and 2,304,652 KiB maximum RSS. Its setup used the byte-identical delivered script, the same source target, cached inputs, dependency pins, and local A100 route.

## Resources and schedule

Run directly on the single local NVIDIA A100, never through Slurm. Recheck free GPU memory immediately before the accelerator stage and require at least 8 GiB. There is one target, so stages run sequentially: clean build, full CPU preprocessing, one complete GPU accuracy run, evaluator. No second GPU run overlaps it. The delivered `reproduce.sh --target bert-squad --from build` path removes its own later markers and rewritten artifacts, then recreates everything from the fixed read-only inputs.

## Expected time and reserve

The directly comparable prior full measurement drives the estimate: allow 30 minutes total, consisting of about 2 minutes for clean build and preprocessing, about 27 minutes for the complete GPU/evaluator path, and artifact checks. Hold 15 minutes before the two-hour limit for `report.json`, required-output validation, and clean-script checks. The projected completion is well inside the available budget.

## Re-plan checkpoint

At the first 5% checkpoint, the live accuracy log contained 554 of 10,833 features after approximately 85 seconds from GPU-stage launch. The observed rate of about 6.5 features/s including startup projects the accelerator pass to about 27.7 minutes, consistent with the 30-minute full-run plan. The single-GPU sequential schedule is unchanged.
