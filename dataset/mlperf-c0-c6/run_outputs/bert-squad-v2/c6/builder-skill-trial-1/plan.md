# Reproduction plan

## Contract

- Target: `bert-squad` only.
- Route: `python3 run.py --backend=pytorch --accuracy`, default Offline scenario, BERT-Large FP32 checkpoint, complete SQuAD v1.1 validation split, followed by `accuracy-squad.py` / `evaluate_v1.1.py`.
- Required outputs: `mlperf_log_accuracy.json`, `mlperf_log_detail.txt`, and evaluator stdout in `accuracy.txt` under `bert-squad/out`.
- Required results: `exact_match` and `f1`, copied exactly from the full evaluator JSON into `report.json`.
- Compute: local shared NVIDIA A100 80GB, no Slurm, one GPU run at a time, with free memory checked immediately before launch.

## Measured costs

- Bounded preflight (8 examples) ran end to end from 2026-11-23 13:26:18 to approximately 13:28:19: about 121 seconds total.
- Environment resolution/install: approximately 17 seconds; CPU staging/input checks: approximately 2 seconds.
- Run/evaluation stage: approximately 102 seconds. It built the complete 53,468,824-byte feature cache, loaded the 1.34 GB checkpoint, ran 8 examples, and evaluated them.
- Preflight result: `exact_match=75.0`, `f1=75.0`; mandatory log line was `pytorch device: cuda:0`.
- Launch-time GPU availability was 73,324 MiB free. Observed process GPU allocation was about 1.7 GiB during the bounded run.
- Comparable clean full run from the read-only same-arm prior campaign took 802.18 seconds and reported peak host RSS 2,296,500 KiB.

## Resources and schedule

- Host: 8 visible CPU cores, about 1.9 TiB available RAM, one shared A100 80GB.
- There is one target, so no cross-target CPU/GPU overlap is possible. Environment, staging, GPU run, and evaluator execute sequentially through the delivered script.
- Launch the first full run with `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from environment`. This clears and rebuilds the script-owned environment, work copy, markers, and required outputs, satisfying the clean-run contract while preserving the separately named preflight logs.
- The run stage performs its own immediate `nvidia-smi` memory check and refuses to launch below 8,192 MiB free.

## Expected time and reserve

- Expected full replay: 15 minutes, driven by the prior 802.18-second clean full run plus shared-GPU contention margin.
- Reporting and integrity-check reserve: 20 minutes.
- Total projection is comfortably below the 2-hour wall-clock limit.

## Re-plan triggers

- After early inference evidence appears, compare elapsed time with the 15-minute projection.
- On failure, use the first failing log line to repair once and rerun with the appropriate `--from` stage; never alter the fixed backend, precision, dataset, scenario, preprocessing, or evaluator semantics.
- If projected completion approaches the reserve, stop launching new work and write `report.json` from only fully measured evaluator output.

## Re-plan log

- 2026-11-23 approximately 13:33 EST: the full feature cache contains 10,833 features, and the growing LoadGen accuracy log advanced from 3.3 MB to 6.9 MB in about 72 seconds under shared-GPU contention. Projection changed from 15 minutes to approximately 20 minutes. Stage order and parallelism remain unchanged because there is only one target and the new projection still leaves more than one hour before the reporting reserve.
- 2026-11-23 approximately 13:39 EST: direct counting found 4,518 of 10,833 feature records (41.7%) after about 9.7 minutes of GPU-process time. Projection changed from 20 to approximately 24 minutes. No schedule change is useful for a single fixed GPU run, and the projection remains safely inside the two-hour limit.
