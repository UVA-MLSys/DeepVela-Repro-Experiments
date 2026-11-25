# BERT-SQuAD reproduction plan

## Workload

One target, `bert-squad`: build a Python 3.11 environment and this source revision's LoadGen; stage the pinned BERT source and read-only inputs; preprocess all SQuAD v1.1 validation examples into 10,833 features; run PyTorch fp32 BERT-Large in LoadGen Offline AccuracyOnly mode on the local A100; run the repository-owned `accuracy-squad.py` / `evaluate_v1.1.py` evaluator.

## Preflight measurements

- Environment and LoadGen build: 71 seconds, CPU only, approximately 5.1 GiB staged environment.
- Full input preprocessing: 61 seconds, CPU only, 10,833 features from 10,570 questions.
- Bounded accelerator run: 17 seconds including model construction/loading; LoadGen measured 0.825 seconds for eight samples (about 9.7 samples/s) while the shared GPU reported 100% utilization from concurrent jobs. Peak availability immediately before launch was 75,666 MiB.
- Bounded evaluator: 6 seconds.

## Resources and schedule

The machine provides 8 visible CPU cores, about 1.9 TiB available host memory, and one shared NVIDIA A100-SXM4-80GB. The single target runs stages in dependency order. GPU work is serialized; no other process started by this reproduction uses the GPU. The script checks free memory immediately before each accelerator run and requires at least 8 GiB. There is no second ready target whose CPU stages could overlap the GPU run.

## Expected time and reserve

Scaling the measured shared-GPU rate gives about 18.6 minutes for 10,833 samples. Allowing for model load, filesystem logging, shared-GPU contention, and full evaluator postprocessing gives a conservative 25-minute full-run projection. A 15-minute reserve is held for evaluator validation, `report.json`, artifact checks, and clean-copy script verification. This fits within the two-hour wall-clock limit.

## Launch

Run the delivered script with no preflight bound using `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from environment`. This recreates the environment and every downstream artifact through the script's own clean path.

## Re-plan at 4.9% of full inference

The first 531 of 10,833 samples logged in approximately 52 seconds (about 10.2 samples/s) while aggregate GPU utilization remained 100%. This projects about 17.7 minutes for full inference from its launch, slightly faster than the bounded estimate. The schedule, 25-minute total projection, and 15-minute closeout reserve remain unchanged.

## Clean-path audit and final run

The first artifact audit found that `--from environment` retained the preflight copy's private `loadgen-src/build` directory. The reset path was repaired to remove both the virtual environment and private LoadGen source/build tree. The final accepted run then recreated all packages, compiled LoadGen from a fresh source copy, regenerated all 10,833 features, checked 79,197 MiB free immediately before accelerator launch, completed all 10,833 accuracy samples, and ran the evaluator. Reduced GPU contention made final inference substantially faster than the conservative plan; the evaluator printed the same metrics as the earlier measurement.
