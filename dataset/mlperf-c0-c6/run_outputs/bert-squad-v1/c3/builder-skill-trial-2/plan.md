# Reproduction plan

Updated: 2026-11-22 18:36 America/New_York, after successful bounded preflight.

## Workload and dependencies

One target, `bert-squad`, runs the stages built into `reproduce.sh` in this order:

1. `prepare`: copy the read-only source into `bert-squad/work/repo`, stage the fixed dataset/model/vocabulary and pinned DeepLearningExamples import tree, and add the required device diagnostic in the disposable copy.
2. `environment`: create a workspace-local uv Python 3.11 environment from the mandated cache and compile the pinned source tree's LoadGen extension.
3. `preprocess`: convert all SQuAD v1.1 validation examples into 10,833 features on CPU.
4. `run`: inspect current GPU memory, require at least 6,000 MiB free, then execute the exact full command `python3 run.py --backend=pytorch --accuracy` in Offline AccuracyOnly mode on the local GPU.
5. `evaluate`: run `accuracy-squad.py` over the full LoadGen accuracy log and retain its stdout and scored outputs.

Each stage reads the preceding stage's workspace files. There is no second target with independent CPU work to overlap with the GPU stage.

## Measured preflight costs

- Environment resolution/install plus LoadGen compilation: approximately 35 seconds wall time; 8 host CPUs available. The package install was cache-only.
- Full dataset preprocessing: approximately 51 seconds and 10,833 generated features. This preprocessing was not reduced for the bounded check.
- GPU preflight: 2 features completed in 385.8 ms of measured LoadGen test time, or about 193 ms/feature. Model startup and imports added under one minute. Peak memory was not separately sampled, but the one-sample-at-a-time reference implementation passed with far more than its 6,000 MiB admission threshold.
- Bounded evaluator: approximately 6 seconds.
- Pre-run resource observation: NVIDIA A100-SXM4-80GB, 81,920 MiB total, 74,324 MiB free; host memory 1.9 TiB available; 8 CPUs.

## Full schedule and expected time

The delivered no-stage invocation performs a clean rebuild: preparation removes only its disposable work copy, and environment creation removes only its target-local virtual environment. The stages then run serially because they depend on one another and there is only one target/GPU workload.

The measured GPU rate projects `10,833 * 0.193 s = 2,091 s`, approximately 35 minutes. Allowing two minutes for clean preparation/environment/preprocessing, two minutes for model startup and output handling, and one minute for full post-processing/evaluation gives a conservative 40-minute projection. GPU work runs alone and directly on this host.

## Reserve and re-plan triggers

Reserve 15 minutes before the two-hour wall-clock limit for `report.json`, artifact checks, exact-value extraction, and a final deliverable inspection. After the first several hundred full-run features, compare observed elapsed time with the 35-minute inference projection. Re-plan if the projection exceeds 90 minutes total, if free GPU memory falls below the wrapper's 6,000 MiB gate, or if another process materially changes the observed rate. No new work starts once the 15-minute reserve is reached.

## Re-plan 1: first full-run progress

At 18:41:12 EST, 3,269 of 10,833 accuracy records had been written after 181 seconds of the inference process. The GPU was at 100% utilization with 75,684 MiB free while two unrelated sessions also held small GPU allocations. The observed rate projects roughly 10 minutes total for inference, comfortably below the original 35-minute projection. The stage order and one-run GPU concurrency are unchanged; the saved time is added to the reporting and artifact-validation reserve. Processes owned by the other sessions remain untouched.
