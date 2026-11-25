# Full reproduction plan

## Workload and dependencies

Target `bert-squad` runs stages `prepare -> environment -> preprocess -> run -> evaluate`. Preparation creates a workspace-local source copy and stages read-only inputs; environment builds a local virtual environment and LoadGen; preprocessing creates all 10,833 validation features; the single GPU run consumes those features; the official evaluator consumes the resulting LoadGen accuracy log.

## Measured costs

- End-to-end 8-example preflight: 2:05.91 wall time, 2,338,764 KiB peak host RSS. This includes a cold environment build and full preprocessing.
- Environment evidence: Python 3.11.16, PyTorch 2.5.1+cu124, Transformers 4.46.3, LoadGen 6.0.17.
- Preprocessing output: 10,833 features from the full SQuAD v1.1 validation input.
- 256-example `run -> evaluate` measurement: 41.25 seconds, 2,330,776 KiB peak host RSS. GPU free memory immediately before launch was 76,125 MiB.
- The 8-example preflight saw 73,066 MiB free immediately before launch. Both bounded runs logged `pytorch device: cuda:0` and completed the official evaluator.

## Resources and schedule

Use the local shared NVIDIA A100 80GB directly, with one GPU run at a time and a fresh `nvidia-smi` memory check in the run stage. The wrapper requires at least 6,000 MiB free before launch. CPU stages run serially because there is one target; no independent ready work remains to overlap with its GPU stage. Package resolution uses only the fixed shared uv cache and cutoff.

Run the delivered entry point unchanged from nothing built:

```bash
OUTPUT_DIR="$PWD/bert-squad/out" ./reproduce.sh --target bert-squad
```

## Projection and reserve

The 256-example measurement contains about 25 seconds of fixed startup/evaluation overhead; its remaining rate projects the 10,833-feature inference near 12 minutes. Budget 15 minutes for the complete cold replay and 20 minutes afterward for evaluator/report validation and artifact checks. This is comfortably inside the two-hour limit.

Re-plan if the first progress evidence or a failure projects beyond 100 minutes total, GPU free memory falls below 6,000 MiB, or a required output is absent. In that event, record the trigger and revised projection here before any repaired rerun.
