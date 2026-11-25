# Full reproduction plan

## Workload

Target `3d-unet` runs four ordered stages: create a Python 3.11 environment and build the supplied LoadGen source; preprocess and checksum all 43 inference cases; run the supplied FP32 TorchScript model in LoadGen Offline accuracy mode on the local A100; run the repository `accuracy_kits.py` evaluator over the resulting accuracy log.

## Measured costs and resources

- Setup: the initial uncached environment plus LoadGen build took about 4 minutes; the repaired cached preflight setup took about 1 minute. The full run may reuse only the mandated UV package cache, not the preflight environment.
- Preprocessing: two representative cases, including duplicate-mapped `case_00400`, took 29.8 seconds and 2.56 GiB peak host RSS with four processes. Both matched repository MD5 checksums. A linear projection is 10.7 minutes for 43 cases; allow 15 minutes for size variation and shared-filesystem contention.
- GPU inference: the two cases required 12.2 seconds of model time and 30.6 seconds end to end, with 5.58 GiB peak host RSS. The larger case used 125 sub-volumes. A conservative full-set allowance is 12 minutes. The preflight GPU checks observed 72,975 and 78,276 MiB free; the script rechecks immediately before the full GPU stage and requires 20,000 MiB free.
- Evaluation: the two-case evaluator completed in under 30 seconds. Allow 5 minutes for 43 cases and the larger accuracy log.
- Available resources: one shared NVIDIA A100-SXM4-80GB used serially, 8 host CPUs, about 1.8 TiB available host RAM, and 4.7 TiB free workspace storage at preflight.

## Schedule and expected time

There is one target, so setup and preprocessing run first on CPU, then exactly one GPU run, then CPU evaluation. Four preprocessing/evaluation workers avoid oversubscribing the 8 visible CPUs under concurrent jobs. Expected full wall time is 25–35 minutes, driven by the measured preprocessing rate and bounded GPU sub-volume rate. A 10-minute reserve remains within the one-hour limit for report generation and artifact checks.

## Re-plan after full-run progress

Full preprocessing finished in 3 minutes 11 seconds rather than the 10.7-minute linear estimate, with every checksum matching. The first four GPU cases used 21.3 seconds of model time, or about 5.3 seconds per case, projecting roughly 4 minutes for all 43 cases. The schedule is unchanged because the reference SUT processes cases serially on the one permitted GPU. Revised total full-run projection: under 15 minutes, leaving more than the planned reserve.
