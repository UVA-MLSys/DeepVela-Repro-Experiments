# Full reproduction plan

- Workload: one `3d-unet` target with four stages: clean local Python/LoadGen setup; CPU preprocessing and repository checksum verification for all 43 inference cases; one local A100 fp32 PyTorch AccuracyOnly/Offline run; and the repository `accuracy_kits.py` evaluator.
- Inputs: the read-only raw KiTS19 cache contains the 42 unique source cases needed by the 43-case manifest. The working copy maps documented duplicate `case_00400` to `case_00185`. The supplied model is `3dunet_kits19_pytorch.ptc` (124,917,156 bytes).
- Preflight: setup completed in about 84 s; preprocessing and checksum verification for `case_00000` took 14 s; inference startup and run took 14 s, including 7.88 s for 50 sliding-window sub-volumes; evaluation took 5 s. The one-case accuracy log is 55 MB. The GPU check immediately before inference reported 62,497 MiB free, and the run logged `pytorch device: cuda:0`.
- Resources: this machine provides 8 CPU cores, about 2 TiB RAM, and one shared A100 80GB. Preprocessing and evaluation use 8 workers. Exactly one GPU run is launched, directly rather than through Slurm, after a fresh memory check; the script requires at least 16,000 MiB free.
- Schedule: run the delivered `./reproduce.sh --target 3d-unet --from setup` from a clean full-run work directory. Setup, preprocessing, inference, and evaluation are dependent and therefore serial. There is no second target or independent ready CPU stage to overlap with inference.
- Projection: setup 2 min; full preprocessing 5 min (conservative versus 14 s for one case with 8 workers); full inference 10 min (conservative versus the measured one-case rate and prior same-arm artifact); evaluation 8 min; total about 25 min.
- Reserve: retain 10 min of the one-hour limit for `report.json`, required-output checks, evaluator parsing, and clean-deliverable validation.
- Compatibility repair: LoadGen 6.0.17 rejects the legacy harness's second config load. In AccuracyOnly, `user.conf` contains only performance target-QPS fields, so the copied harness skips that load without changing dataset, model, backend, precision, scenario, preprocessing, predictions, or evaluator.

## Re-plan after full-run progress

- Trigger: after 2 min 17 s of inference, the LoadGen accuracy log had reached about 1.1 GB and the A100 was at 100% utilization.
- Observed resources: the run launched with 81,037 MiB free; its process initially used about 4.0 GiB and remained healthy as log output grew.
- Decision: no schedule change. The observed growth is consistent with a roughly 7–8 min inference stage, below the 10 min stage allowance and well inside the total projection and reserve boundary.
