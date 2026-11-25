# Reproduction plan

## Workload and dependencies

One target, `bert-squad`, runs four serial stages in `reproduce.sh`: `environment` builds Python 3.11 and LoadGen; `prepare` validates hashes, copies the repository, installs the repair, generates all feature windows, and serializes them; `gpu` checks current A100 memory then runs the exact PyTorch AccuracyOnly launcher over every feature; `evaluate` uses the benchmark postprocessor and evaluator over the full LoadGen log. Each stage writes a marker only after success.

## Preflight measurements

- Environment: about 45 seconds; LoadGen compilation dominated; peak build RSS observed earlier was about 0.53 GiB.
- Preparation: 60.0 seconds, 10,570 SQuAD question examples to 10,833 windows of length 384; peak RSS 0.79 GiB.
- Bounded GPU stage: 83.2 seconds for model import/load, repeated full feature construction, and 8 queries; peak host RSS 1.94 GiB. The run log confirmed `pytorch device: cuda:0`.
- Bounded evaluation: completed in about 1 second after serialized feature preparation and printed both target metrics.
- GPU availability immediately before bounded run: 72,468 MiB free of 81,920 MiB on the NVIDIA A100-SXM4-80GB.

## Resources and schedule

The task permits direct local execution and requires one GPU run at a time. With only one target, environment and preparation run first on CPU, followed by one GPU stage, then CPU evaluation. There is no other ready target whose CPU work can overlap the GPU stage. The script checks free GPU memory immediately before the full run and does not interact with processes it did not start.

The bounded timing contains roughly 60 seconds of fixed feature construction plus model load; extrapolating the remaining per-query work to 10,833 windows gives an intentionally broad 10–20 minute full-run estimate. This fits the two-hour limit. Hold 15 minutes in reserve for report extraction, artifact validation, and standalone replay checks. If the first 500–1,000 completed samples project beyond 90 minutes, re-plan before the reserve; LoadGen's detail log is the progress/rate evidence.

## Full run

Run the delivered script unchanged and without `--max-examples`:

```bash
OUTPUT_DIR="$PWD/bert-squad/out" ./reproduce.sh --target bert-squad --from environment
```

`--from environment` removes only script-owned downstream markers and output files, clears and recreates the isolated environment, recreates the repository work copy, and executes all later stages. On completion, parse the final evaluator JSON exactly into `report.json`, validate required filenames and log prefix, then perform a clean standalone-copy replay or equivalent copy-alone validation within the remaining reserve.

## Re-plan: early full-run rate

Trigger: the streamed accuracy log reached 207 of 10,833 results (1.9%) after roughly 90 seconds of inference. The measured rate was about 2.3 samples/s while unrelated sessions also occupied the shared A100. Projected accelerator duration is about 79 minutes; adding fixed stages and evaluation remains below the 105-minute execution allowance before the 15-minute reporting reserve. No schedule or concurrency change is warranted: continue the one owned GPU process, do not touch unrelated processes, and reassess if the measured rate falls below about 1.9 samples/s.
