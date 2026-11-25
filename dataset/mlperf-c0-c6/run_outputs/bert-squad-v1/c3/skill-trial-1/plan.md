# BERT SQuAD reproduction plan

## Workload

Target `bert-squad` runs the pinned repository's PyTorch backend in Offline accuracy mode on the supplied BERT-Large fp32 checkpoint and the complete SQuAD v1.1 validation set. Its stages are: build an isolated Python 3.11 environment and local LoadGen wheel; prepare the complete feature cache; run LoadGen on the GPU; run `accuracy-squad.py`, which invokes `evaluate_v1.1.py`.

## Measured costs

- Environment/source preparation and LoadGen compilation: approximately 47 seconds on the successful preflight rebuild.
- Complete CPU feature preparation: approximately 57 seconds, producing 10,833 features from 10,570 questions and a 51 MiB cache.
- Accelerator: the 100-feature bounded run took 11.76 seconds inside LoadGen (8.50 features/s). Linear projection for 10,833 features is 1,274 seconds (21.2 minutes). The complete bounded run including model load and evaluation took 32.28 seconds.
- Evaluator: approximately 6 seconds for the bounded output; allow up to 2 minutes for the complete accuracy log and decoding.
- Memory: the bounded process maximum RSS was 2,299,344 KiB. Immediately before the measured GPU run, the A100 had 75,558 MiB free of 81,920 MiB while unrelated jobs occupied the remainder. The model process uses only a small fraction of this capacity.

## Resources and schedule

Run directly on the single local NVIDIA A100, never through Slurm. Recheck free GPU memory immediately before the accelerator stage and require at least 8 GiB. There is one target, so stages run sequentially: build, full CPU preprocessing, one GPU run, evaluator. No second GPU run will overlap it. The delivered `reproduce.sh --target bert-squad --from build` path clears all later markers/artifacts itself and recreates them from the fixed inputs.

## Expected time and reserve

Expected full wall time is 25–30 minutes: about 2 minutes for build and preprocessing, about 21–23 minutes for inference at the measured shared-GPU rate, and at most 2 minutes for evaluation and artifact checks. Hold 15 minutes before the two-hour limit for `report.json`, output validation, and the final clean-script check. The projected completion is well inside the available budget.

## Re-plan checkpoint

At the first 5% of the full GPU pass, the live accuracy log held 556 of 10,833 features after approximately 78 seconds from process start (about 7.1 features/s including model startup). This projects the accelerator stage to about 25.4 minutes. The measured rate is slower than the 100-feature bounded rate but remains within the plan's conservative 30-minute full-run envelope, so the single-GPU sequential schedule is unchanged.
