# Full reproduction plan

## Workload

Target `3d-unet` has four ordered stages: create a Python 3.11 environment and build the supplied LoadGen source; preprocess and checksum all 43 inference cases; run the supplied FP32 TorchScript model in LoadGen Offline accuracy mode on the local A100; run the repository `accuracy_kits.py` evaluator over the resulting accuracy log.

## Measured costs and resources

- Setup: the fresh cache-backed preflight environment and LoadGen build took about 1 minute. The full run starts with no environment and may reuse only the mandated UV cache.
- Preprocessing: two representative cases, including duplicate-mapped `case_00400`, took 32.3 seconds and 2.44 GiB peak host RSS with four processes. Both matched repository MD5 checksums. A conservative linear projection is 11.6 minutes for 43 cases; the provenance-verified prior full run of the same source and route completed this stage in 3 minutes 11 seconds.
- GPU inference: the two cases took 17.7 seconds of model time and 34.0 seconds end to end, with 5.32 GiB peak host RSS. The large case exercised 125 sub-volumes. A conservative linear projection is 12.2 minutes; the provenance-verified comparable full run completed in 5 minutes 3 seconds. The preflight GPU check found 71,493 MiB free; the script rechecks immediately before the full GPU stage and requires 20,000 MiB free.
- Evaluation: the bounded evaluator completed in about 15 seconds. Allow 5 minutes for all 43 cases.
- Available resources: one shared NVIDIA A100-SXM4-80GB used serially, 8 host CPUs, about 1.8 TiB available host RAM, and 5.4 TiB free workspace storage at preflight.

## Schedule and expected time

There is one target, so setup and preprocessing run first on CPU, then exactly one GPU run, then CPU evaluation. Four preprocessing/evaluation workers avoid oversubscribing the 8 visible CPUs. The conservative measured projection is about 30 minutes; the comparable full measurement projects under 15 minutes. The preprocessing and inference measurements drive those estimates. Reserve at least 10 minutes of the one-hour limit for `report.json` and final artifact/script checks.

## Re-plan after full-run progress

Full preprocessing completed in 4 minutes 26 seconds with all 43 checksums matching. The first three GPU cases (7% of the set) used 11.8 seconds of model time, or 3.93 seconds per case, projecting about 2 minutes 49 seconds of model time for all 43 cases and a sub-5-minute inference stage including setup and cleanup. The one-GPU serial schedule is unchanged. Revised total full-run projection remains under 15 minutes, preserving the reporting reserve.
