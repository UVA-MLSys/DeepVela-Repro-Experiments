# Full-run plan

## Workload and dependencies

Target `3d-unet` runs one serialized chain: `environment -> data -> preflight -> run -> evaluate`. The environment copies the supplied source and builds in-tree LoadGen; data creates a read-only 43-case view and checksum-verified preprocessing; preflight checks the GPU/model/evaluator; run performs the full PyTorch fp32 Offline accuracy pass; evaluate invokes the benchmark evaluator and writes `accuracy.txt`.

## Measured costs and resources

Current-trial preflight stages measured 89 s for environment construction, 203 s for four-worker CPU preprocessing, and 9 s for GPU/model/evaluator preflight (`3d-unet/out/{environment,data,preflight}.log`). The same-arm prior snapshot measured 253 s for full GPU inference and 88 s for evaluation (`prior-artifacts/3d-unet/out/{run,evaluate}.log`).

The host exposes 8 CPU cores, about 2 TiB RAM, one shared NVIDIA A100-SXM4-80GB, and 5.2 TiB free workspace storage. Immediately before preflight, `nvidia-smi` reported 81,037 MiB free (`3d-unet/out/preflight.log`). The accelerator run is single-process and no other work from this session uses the GPU concurrently.

## Schedule, estimate, and reserve

Launch the delivered script unchanged with `--from environment`; the script itself clears and rebuilds every stage and output it owns. With one target there is no useful cross-target overlap. Four preprocessing/evaluator workers fit the 8-core host while leaving headroom. The projected clean replay is `89 + 203 + 9 + 253 + 88 = 642 s` (about 11 minutes). Reserve at least 10 minutes of the one-hour limit for artifact verification, parsing exact result strings, writing `report.json`, and diagnosing one repair cycle if necessary.

At early run progress, compare logged per-case inference times against the ~253 s prior run. If projected GPU time exceeds 20 minutes, retain the existing correctness-first schedule because no validated throughput/batching semantic change is available; the remaining wall-clock reserve still fits the limit.
