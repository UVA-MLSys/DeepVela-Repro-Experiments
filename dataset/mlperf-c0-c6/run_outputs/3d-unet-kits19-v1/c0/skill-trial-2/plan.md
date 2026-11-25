# Full reproduction plan

- Workload: one `3d-unet` target. Stages are local Python/LoadGen setup, CPU preprocessing plus repository checksum verification for all 43 manifest cases, one local A100 fp32 TorchScript AccuracyOnly run, and the repository `accuracy_kits.py` evaluator.
- Inputs: read-only raw KiTS19 cache (42 unique inference cases), working-copy `case_00400` link to documented duplicate `case_00185`, and `3dunet_kits19_pytorch.ptc` from the supplied model cache.
- Preflight measurements: environment setup 107 s from marker timestamps; preprocessing/checksum 16 s for `case_00000`; inference 5.98 s for its 50 sliding-window sub-volumes, launched with 81,037 MiB GPU memory free; evaluator 6 s. The one-case accuracy log was 56.6 MB.
- Resources: 8 CPU preprocessing/evaluator workers; one shared A100 run at a time; GPU launch requires at least 16,000 MiB free and records `nvidia-smi` state in `run.log`. No Slurm work is used.
- Schedule: build and preprocess serially, then check current GPU memory and run the single accelerator stage, then evaluate. There are no other targets whose CPU work can overlap this GPU run.
- Projection: setup 2 min; preprocessing 6 min (conservative versus 16 s/case with 8 workers); inference 10 min (conservative versus 0.12 s/sub-volume in preflight); evaluation 8 min; total about 26 min.
- Reserve: retain at least 10 min of the one-hour limit for `report.json`, output validation, and a clean-deliverable check.
- Compatibility repair: LoadGen 6.0.17 flags the legacy harness's second config load. In AccuracyOnly, `user.conf` contains only performance target-QPS fields, so the copied harness skips that second config load without changing dataset, model, backend, precision, scenario, preprocessing, predictions, or evaluator.

## Re-plan after full-run progress

- Trigger: the first several cases completed. Preprocessing plus checksum verification took about 2 min, and after 76 s of inference the accuracy log was 420 MB (about seven one-case log equivalents).
- GPU state: 65,041 MiB free at launch; this process uses about 6.8 GiB while two unrelated jobs share the device.
- Decision: no schedule change. The observed rate projects roughly 7–8 min for inference and remains below the original 26 min total projection and 10 min reserve boundary.
