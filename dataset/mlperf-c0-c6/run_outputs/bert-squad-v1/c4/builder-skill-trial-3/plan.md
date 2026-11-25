# Full reproduction plan

Written after the successful bounded preflight on 2026-11-23.

## Workload and dependencies

One target, `bert-squad`, runs three serial stages from the delivered `reproduce.sh`:

1. `build`: copy the metadata-free source into `bert-squad/work`, create a Python 3.11
   environment from the mandated offline uv cache, and compile LoadGen.
2. `prepare`: connect the three fixed inputs and separately staged tokenizer tree, generate the
   benchmark-owned evaluator, add the required device instrumentation, validate checksums and
   all 10,570 unique question IDs, and preprocess 10,833 BERT feature windows.
3. `run`: check GPU memory, run all 10,833 feature windows through BERT-Large fp32 in LoadGen
   Offline AccuracyOnly mode, evaluate all 10,570 questions, and copy the three declared outputs.

The full acceptance run starts with `--from build`, so it clears and recreates every script-owned
derived artifact and tests the delivered script from nothing built.

## Measured costs

- Environment resolution plus LoadGen compilation: about 50 seconds on the successful preflight;
  CPU-only, with package payloads reused from `UV_CACHE_DIR`.
- Input validation plus full preprocessing: about 46 seconds; CPU-only. The resulting feature
  cache contains 10,833 records and is 53,469,661 bytes.
- GPU inference: LoadGen measured 3.071 seconds for eight records, or 0.384 seconds/record, on the
  shared A100. Linear projection for 10,833 records is 4,158 seconds (69.3 minutes).
- Evaluator: nine seconds on the bounded output; allow two minutes for full log parsing and scoring.

## Resources and schedule

The machine exposes 8 CPU cores, 2.0 TiB RAM (1.9 TiB available at planning time), and one shared
NVIDIA A100-SXM4-80GB. At planning time the GPU had 75,685 MiB free. The model preflight fit with
more than 70 GiB headroom. Only one GPU run will execute. There is no second target or ready
CPU-only target work to overlap with it.

Stages run build -> prepare -> run. The script checks `nvidia-smi` again immediately before the GPU
stage and refuses to start below 12,000 MiB free. Estimated clean completion is 72 minutes. A
15-minute reserve is held for `report.json`, artifact checks, and a final clean-deliverable audit,
keeping the projected total below the two-hour limit.

## Re-plan triggers

After the first several percent of LoadGen records, compare observed progress with 0.384
seconds/record. If projected completion exceeds the limit minus the 15-minute reserve, record the
new projection here before changing anything. Any failure returns to the earliest affected script
stage via `--from`; completed unrelated stages are retained as directed by the reproduction skill.

## Re-plan 1: live full-run rate

At 1,757/10,833 records (16.2%), 852 additional records arrived in 73 seconds: 11.7 records/s or
0.086 seconds/record. The short preflight was dominated by fixed LoadGen completion overhead and
therefore overestimated the linear per-record cost. With about 9,076 records left, inference now
projects roughly 13 minutes remaining. The GPU is shared by three unrelated sessions, but this run
is healthy and its accuracy log is growing. The schedule remains unchanged and the reporting
reserve increases substantially.
