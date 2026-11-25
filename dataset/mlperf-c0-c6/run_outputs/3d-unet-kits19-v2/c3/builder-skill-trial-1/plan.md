# Full reproduction plan

Written 2026-11-19 after a fresh bounded end-to-end preflight.

## Workload

One target, `3d-unet`, with sequential stages implemented by the delivered `reproduce.sh`:

1. `setup` creates a Python 3.10 environment from the required offline uv cache, copies `repo/`
   into target-local writable work, and creates a read-only symlink view of raw cases with the
   documented `case_00400 -> case_00185` mapping.
2. `preprocess` transforms and checksum-verifies all 43 inference-list entries on CPU.
3. `run` records current GPU memory, performs one local A100 Offline accuracy run, and invokes the
   benchmark evaluator on CPU.

The run stage depends on setup and preprocessing success markers. There is no second target or
independent ready CPU workload to overlap with the only GPU run.

## Measured costs

Fresh one-case preflight (`3d-unet/preflight`):

- setup: compatible environment built successfully from the offline cache; Torch 2.2.0+cu121;
- preprocessing: 21.880 seconds, checksum verified;
- GPU run: 51.423 seconds total, 5.520 seconds model processing for the case;
- evaluator: 36.416 seconds, terminal evaluator output produced;
- GPU state immediately before launch: A100 80GB, 81,920 MiB total, 13,373 MiB used,
  67,781 MiB free.

The provenance-identified prior same-arm full run supplies a stronger full-scope estimate:

- preprocessing: 2m33.454s;
- GPU run: 4m07.559s;
- evaluator: 2m12.932s;
- result: all 43 cases and evaluator completed.

## Resources and schedule

This machine currently exposes 8 CPU cores, about 1.8 TiB available host memory, 5.5 TiB free
filesystem space, and one shared NVIDIA A100 80GB. The full run uses four preprocessing/evaluator
workers and exactly one GPU process. `reproduce.sh --target 3d-unet --from setup` first clears only
its own full-target products, rebuilds from nothing, then runs all stages in order. It checks GPU
memory immediately before inference and logs it. GPU work is serialized by construction.

## Projection and reserve

The comparable measured full stages total about 8m54s. Allowing environment rebuild and shared
filesystem/GPU contention, projected wall time is 10-15 minutes. Reserve 10 minutes before the
one-hour limit for exact result extraction, `report.json`, required-output checks, and a final
fresh-copy/deliverable audit. If the first several cases project beyond 40 minutes total, retain
the running single process (no safe accuracy-preserving concurrency change exists) and begin final
artifact checks as soon as evaluation finishes.
