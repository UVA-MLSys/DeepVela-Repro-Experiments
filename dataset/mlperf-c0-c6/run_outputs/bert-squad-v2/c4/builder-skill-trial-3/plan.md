# Full reproduction plan

Written after the successful bounded preflight on 2026-11-23.

## Workload and dependencies

The single `bert-squad` target has three serial stages in the delivered `reproduce.sh`:

1. `build` copies the metadata-free source, creates a target-local Python 3.11 environment from the mandated offline uv cache, and compiles LoadGen.
2. `prepare` connects the fixed read-only dataset, checkpoint, vocabulary, and tokenizer source; installs the benchmark-owned evaluator; instruments the required device line; and validates hashes plus all 10,570 unique question IDs.
3. `run` checks free A100 memory, preprocesses 10,833 BERT feature windows, runs BERT-Large fp32 with LoadGen Offline AccuracyOnly, evaluates all 10,570 questions, and copies the three declared outputs.

The acceptance run uses `./reproduce.sh --target bert-squad --from build`, which clears every script-owned derived artifact and starts from nothing built.

## Measured costs

- Current bounded preflight build: about 50 seconds, CPU-only; all packages resolved offline from `UV_CACHE_DIR` and LoadGen compiled successfully.
- Current input preparation: about 1 second, CPU-only; dataset, model, and vocabulary hashes were recorded and 10,570 unique question IDs validated.
- Current two-record run: about 91 seconds including fixed model loading and full feature preprocessing. The feature cache is 53,469,661 bytes and contains 10,833 windows (confirmed by the permitted prior trial and the current identical cache size).
- Current bounded evaluation: about 8 seconds.
- Permitted prior-trial evidence projects the full inference at about 69 minutes from the short-run rate, while that trial's live full-run rate improved to about 11.7 records/s after 16% completion, projecting roughly 15 minutes of steady-state inference. Use the conservative 72-minute estimate for launch.

## Resources and schedule

This machine exposes 8 CPU cores, 2.0 TiB RAM (1.9 TiB currently available), and one shared NVIDIA A100-SXM4-80GB. The preflight GPU check found 70,684 MiB free; the model fit and ran successfully. Only one GPU workload will run. There is no second target with ready CPU work to overlap.

Run build -> prepare -> run. The script checks `nvidia-smi` immediately before inference and refuses to start below 12,000 MiB free. Expected clean completion is at most 72 minutes. Hold a 15-minute reserve for evaluation, `report.json`, artifact checks, and clean-deliverable verification, within the two-hour limit.

## Re-plan triggers

After the first several percent of records, compare accuracy-log growth with the conservative projection. If completion moves beyond the time limit minus the 15-minute reserve, record the observed rate and changed schedule here. A failure returns to the earliest affected stage through `--from`; no script-owned output is removed manually.

## Re-plan 1: live full-run rate

At 640/10,833 records (5.9%), the run had sustained about 10.7 records/s during its first minute of inference. The remaining 10,193 records project to roughly 16 minutes, substantially faster than the conservative 72-minute allowance and consistent with the permitted prior full run. The GPU was at full utilization while shared unrelated processes remained present. No schedule change is needed; keep the single GPU run active and retain the 15-minute reporting reserve.
