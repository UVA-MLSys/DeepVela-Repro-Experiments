# BERT SQuAD reproduction plan

## Workload

One target, `bert-squad`: create an isolated Python 3.11 environment; copy the pinned source; stage the supplied SQuAD v1.1 validation data, fp32 BERT-Large checkpoint, vocabulary, and runtime dependency; build pinned LoadGen; preprocess all 10,833 feature records derived from 10,570 questions; run Offline AccuracyOnly on CUDA; reconstruct predictions; and invoke the evaluator for exact match and F1.

## Measured costs and resources

- A clean 32-feature end-to-end preflight completed in 171.19 seconds with 1,955,252 KiB maximum host RSS.
- Environment installation used the required shared uv cache and completed in about 5 seconds; LoadGen compilation plus full-dataset preprocessing took about 143 seconds.
- The CUDA portion processed 32 records in 5.20 seconds (6.15 records/s) while the shared GPU reported 100% utilization from concurrent work.
- The preflight produced 32 accuracy records and evaluator output `{"exact_match": 90.625, "f1": 93.125}`.
- Immediately before inference the NVIDIA A100-SXM4-80GB had 71,882 MiB free. The delivered script checks for at least 12,000 MiB free before every GPU run.
- Execution is local on the single shared A100 and local CPUs, with no Slurm submission and only one GPU workload from this reproduction.

## Schedule and expected time

Run the delivered script's four stages in order for the one target: environment, prepare, run, and evaluate. CPU setup and preprocessing feed the only GPU run, so there is no independent target work to overlap. The measured contended inference rate projects to 29.4 minutes for 10,833 features. Allowing clean setup, model loading, evaluation, and further contention, the expected full wall time is 45 minutes.

Reserve 10 minutes before the two-hour limit for parsing evaluator output, writing `report.json`, checking required files and log lines, and verifying standalone-script invariants. If early full-run progress projects past 100 minutes, retain the fixed workload and record the revised projection; benchmark semantics will not be reduced.

## Re-plan during full run

At 456/10,833 records (4.2%), the observed shared-GPU rate was approximately 7.2--7.6 records/s while device utilization remained 100%. This revised the inference projection to about 24--25 minutes, better than the preflight's 29.4-minute estimate. No schedule or workload change is needed: there is one sequential GPU target and the projection remains comfortably inside the time limit and reserve.

The clean full run completed successfully in 18:09.81 wall time, faster than both projections as competing GPU load decreased during inference.
