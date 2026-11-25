# BERT SQuAD reproduction plan

## Workload

One target, `bert-squad`: create an isolated Python environment; copy the pinned source; stage the supplied SQuAD v1.1 validation data, fp32 BERT-Large checkpoint, vocabulary, and DLE dependency; build pinned LoadGen; preprocess all 10,833 feature records; run Offline AccuracyOnly on CUDA; reconstruct predictions and evaluate exact match and F1.

## Measured costs and resources

- Environment installation: about 16 seconds from the shared uv cache.
- LoadGen build plus full preprocessing: about 105 seconds; the feature cache is 53,468,824 bytes and contains 10,833 records.
- A 256-record CUDA preflight: 17.70 seconds in LoadGen, 43.32 seconds end to end, 14.46 records/s, and 1.93 GB maximum host RSS. It completed while another session used the shared GPU.
- GPU before that run: NVIDIA A100-SXM4-80GB, 77,395 MiB free. The script checks for at least 12,000 MiB free immediately before each run.
- Available execution: this machine's single shared A100 and local CPUs; no Slurm. This reproduction runs only one GPU job.

## Schedule and expected time

Run the four script stages in order for the one target. CPU environment setup, compilation, and preprocessing precede and feed the GPU run; there is no second target with CPU work to overlap. At the measured contended rate, full inference projects to 749 seconds (12.5 minutes). Including clean setup, model load, evaluation, and contention margin, expected wall time is 18 minutes.

Reserve 10 minutes before the two-hour limit to parse evaluator output, write `report.json`, verify required artifacts/log lines, and test standalone-script invariants. If the first few percent project beyond 100 minutes, retain the exact workload and allow the large remaining time budget rather than changing benchmark semantics.

## Re-plan during full run

At 1,065/10,833 records (9.8%) after about two GPU minutes, six other processes were sharing the A100. The observed projection moved from 12.5 to approximately 20–22 inference minutes. No schedule or workload change is needed: there is one inherently sequential GPU target, more than 72 GB GPU memory remained free, and the new projection remains well within the limit and reserve.

The full clean-copy run completed successfully in 23:27.49 wall time, consistent with the revised contention estimate.
