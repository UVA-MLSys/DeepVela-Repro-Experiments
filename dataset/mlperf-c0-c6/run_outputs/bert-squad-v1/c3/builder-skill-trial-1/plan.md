# Full reproduction plan

Planned 2026-11-22 after the bounded end-to-end preflight required by
`baseline-reproduce`.

## Workload and dependencies

The sole target is `bert-squad`. Its ordered stages are:

1. `source`: copy the pinned source snapshot into `bert-squad/src`, stage the fixed read-only
   SQuAD, vocabulary, checkpoint, and DeepLearningExamples inputs, and add required device
   logging.
2. `env`: create the Python 3.12 environment from the mandated offline uv cache and compile the
   checked-in LoadGen.
3. `preprocess`: convert all 10,570 validation questions into 10,833 fixed-length features.
4. `run`: check free GPU memory, run LoadGen Offline AccuracyOnly over all 10,833 features on
   GPU, run `accuracy-squad.py`, validate the declared outputs, and mark completion.

Each stage reads the prior stage's success marker and persistent outputs. The full run reuses the
successful source, environment, and full preprocessing artifacts, then replaces the bounded run
outputs through `reproduce.sh --from run`.

## Measured costs and resources

- Environment: 48 cached packages plus LoadGen compilation completed in approximately 2 minutes;
  the environment occupies about 5.2 GB on the shared filesystem. This is CPU/storage work.
- Preprocessing: 10,833 features completed in approximately 65 seconds, used about one CPU core
  and 0.7 GB resident memory, and wrote a 51 MB cache.
- Bounded GPU run: 8 features completed inside LoadGen in 1.728 seconds, or 0.216 seconds/feature.
  The end-to-end run stage took 38 seconds including checkpoint loading and evaluator startup.
- GPU state immediately before the successful preflight: NVIDIA A100-SXM4-80GB with 66,565 MiB
  free. Unrelated processes were using the rest; they are not modified. The measured GPU
  utilization was 100%, so the rate already reflects substantial sharing contention.
- The script requires at least 8,192 MiB free immediately before a run. Only one reproduction GPU
  process is launched at a time.

## Schedule and expected time

There is only one target, so no GPU overlap is possible or useful. The source, environment, and
preprocessing stages have succeeded. The full run starts by invalidating only the bounded `run`
marker and its declared outputs, checks GPU memory again, runs all 10,833 features, evaluates,
then writes `report.json`.

The measured feature rate projects `10833 * 0.216 = 2,340` seconds (39 minutes) of inference.
Allowing 5 minutes for checkpoint loading, post-processing, and filesystem variance gives a
44-minute expected remaining run. A conservative contention envelope is 60 minutes. The 2-hour
task budget therefore leaves at least a 15-minute reserve for report generation, output checks,
and a clean-copy replay audit. If the first few percent project beyond 100 minutes, the plan will
be updated; model/backend/precision/data/scenario/evaluator semantics will not be changed.

## Re-plan at first few percent

At 2026-11-22 14:26 EST, 66 seconds after stage launch, the accuracy log held approximately 420
records (2.60 MB using the measured bounded-record size). Excluding checkpoint startup, this is
about 13.5 features/second and projects 13–15 minutes for inference. The revised end-to-end
projection is 18 minutes including evaluation and filesystem variance. No scheduling change is
needed: the sole GPU run remains serialized, and the larger closing reserve is retained.

## Full-run completion

The shared GPU became more contended after the first checkpoint. Full inference completed without
LoadGen warnings or errors in approximately 30 minutes, still within the original conservative
60-minute envelope. The evaluator printed exact match `83.68968779564806` and F1
`90.87487229720105`; these values were recorded exactly in `report.json`. The remaining closing
work is a clean-copy replay and artifact audit.

## Clean-copy replay completion

A separate `clean-replay` directory was initialized with only the delivered `reproduce.sh` and a
fresh copy of `repo`. Running `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad`
rebuilt the environment and LoadGen, regenerated all 10,833 features, checked 73,322 MiB free
immediately before GPU work, completed LoadGen without warnings or errors, and independently
printed exact match `83.68968779564806` and F1 `90.87487229720105`. The replay therefore matches
the primary measured result exactly.
