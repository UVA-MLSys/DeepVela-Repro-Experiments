# Reproduction plan

## Workload

One target, `3d-unet`, runs three ordered stages through the delivered `reproduce.sh`:

1. `environment`: create a Python 3.11 environment, resolve only through the mandated uv cache and
   cutoff, copy the application privately, repair the missing SUT modules, and build supplied
   LoadGen source.
2. `preprocess`: materialize the 43-case manifest with `case_00400` linked to its documented source
   `case_00185`, run canonical preprocessing, and verify published checksums.
3. `run`: check current GPU free memory, run FP32 TorchScript Offline accuracy through LoadGen on
   the A100, then run `accuracy_kits.py` on all 43 records.

## Preflight measurements

- One representative case (`case_00000`, preprocessed shape `1x192x384x384`) completed end to end.
- Environment installation used cached packages in about 5 seconds; local LoadGen compilation took
  about 40 seconds; the complete environment stage was about 55 seconds.
- One-case preprocessing completed and its published MD5 matched in seconds.
- LoadGen recorded 6.7 seconds through the end of the one-case GPU accuracy run.
- Evaluator postprocessing and scoring took about 34 seconds.
- One-case accuracy JSON size was 55 MiB, implying a likely multi-gigabyte full accuracy log.
- Preflight evaluator result (not the report result): mean 0.90515, kidney 0.9700, tumor 0.8403.
- Preflight GPU check: A100 80GB, 73,287 MiB free immediately before the repaired run.

## Resources and schedule

The machine exposes 8 CPU cores, approximately 1.8 TiB available host memory, 4.5 TiB free
workspace storage, and one shared A100 80GB. The stages run sequentially because there is only one
target and preprocessing is a prerequisite for inference. Preprocessing uses four worker processes;
GPU work is one run at a time. No Slurm work is submitted. The run stage starts only after another
`nvidia-smi` check shows at least 12 GiB free.

The one-case GPU rate is not assumed uniform because CT shapes vary. Allow 1 minute for environment,
6 minutes for 43-case preprocessing/checksums, 8 minutes for volume-dependent inference, and 10
minutes for serializing/loading the multi-gigabyte accuracy log plus NIfTI evaluation: 25 minutes
expected. Hold 8 additional minutes for `report.json`, artifact validation, and a replay-contract
check, staying within the one-hour limit.

## Full-run rule

Launch the unchanged delivered script from its no-stage path with no built target state:

```bash
OUTPUT_DIR="$PWD/3d-unet/out" ./reproduce.sh --target 3d-unet
```

The script itself clears and recreates every stage-owned work product. Re-plan if the first several
cases project the run beyond 52 minutes total, leaving the 8-minute reserve intact.

## Re-plan checkpoint

After about 1.5 minutes of the full GPU stage, the accuracy log had reached 536 MiB, the A100 was at
100% utilization, and this run used about 6.5 GiB of device memory. The observed rate projects the
GPU stage to roughly 7–9 minutes and the whole run remains below the original 25-minute estimate.
Schedule and 8-minute reserve are unchanged.
