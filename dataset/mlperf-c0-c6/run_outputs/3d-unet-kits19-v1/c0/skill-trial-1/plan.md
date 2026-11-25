# Full reproduction plan

## Workload

- Target: `3d-unet` only, in the order specified by `tasks.json`.
- Setup: offline uv environment from `UV_CACHE_DIR`, PyTorch fp32 TorchScript model, and LoadGen 6.0.17 built from the supplied source snapshot.
- Preprocess: all 43 cases in `meta/inference_cases.json`; `case_00400` is a workspace-local symlink to the documented duplicate `case_00185`. The repository checksum verifier must pass.
- Run: repository `run.py --backend=pytorch --scenario=Offline --accuracy`, then repository `accuracy_kits.py` over the complete LoadGen accuracy log.

## Measured costs and resources

- Environment setup: 63 seconds elapsed, CPU-only; the local LoadGen extension build was about 31 seconds.
- One-case bounded preprocessing and checksum verification: 18 seconds using 8 CPU processes.
- One-case bounded GPU inference: 4.19 seconds for 50 sliding-window subvolumes after warmup (the first attempt measured 7.80 seconds).
- One-case complete run plus evaluator: about 21 seconds after preprocessing.
- GPU immediately before both bounded runs: 81,037 MiB free on the NVIDIA A100 80GB.
- The bounded evaluator completed and printed all requested metric classes. LoadGen reports the snapshot's known multiple-config-file submission-validity error because documented `run.py` loads both `mlperf.conf` and `user.conf`; this does not prevent accuracy logging or evaluation.

## Schedule and projection

1. Reuse the successful setup marker.
2. Run full preprocessing with 8 CPU workers. Linear one-case scaling is 12.9 minutes; multiprocessing should make this a conservative estimate.
3. Check free GPU memory, then run the sole GPU workload serially. Linear inference scaling from the slower cold measurement is 5.6 minutes; allow 15 minutes for larger volumes.
4. Run the CPU evaluator with 8 workers and verify artifacts and metric text.

Projected total after launch: 30 minutes conservatively. No other target or independent ready CPU workload exists to overlap with the single GPU run. Reserve 10 minutes before the one-hour limit for `report.json`, artifact checks, and a clean-copy wrapper review.

## Re-plan after first measured full pass

- Trigger: the full 43-case pass completed in 7 minutes 18 seconds and artifact checks passed, but it reused the bounded preflight's successful setup marker.
- Change: rerun with `./reproduce.sh --from setup` so the accepted result exercises the delivered wrapper from nothing built, as required by the reproduction workflow.
- Projection: another 8 minutes based on the measured pass, leaving more than 30 minutes before the limit and preserving the 10-minute reporting reserve.
