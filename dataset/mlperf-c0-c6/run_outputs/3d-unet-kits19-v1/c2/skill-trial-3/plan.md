# Full reproduction plan

## Workload

Target `3d-unet` runs three ordered stages: build an offline Python environment and
LoadGen from the pinned source, preprocess all 43 manifest cases (mapping documented
duplicate `case_00400` to `case_00185`), then run the fp32 TorchScript model in
LoadGen Offline accuracy mode and invoke `accuracy_kits.py` over its accuracy log.

## Measured costs and resources

- Preflight environment setup/build: about 35 seconds.
- Preflight preprocessing: about 10 seconds for `case_00000`, producing shape
  `(1, 192, 384, 384)`. Full preprocessing uses four of the machine's 8 CPU cores.
- Preflight inference: 50 sliding windows in 6.39853 seconds on `cuda:0`, or
  0.128 seconds/window. Header-derived full scope is 2,761 windows, projecting
  353 seconds of model execution.
- Preflight end-to-end evaluation succeeded. It reported a one-case result only and
  is not used as the benchmark result.
- Available resources before launch: 8 CPU cores, 2.0 TiB RAM (1.8 TiB available),
  and one NVIDIA A100-SXM4-80GB with 81,037 MiB free. The script rechecks GPU memory
  immediately before inference and requires 12,000 MiB free.
- Workspace has 4.2 TiB free. The one-case preflight occupies 5.6 GiB.

## Schedule and expected time

Run the delivered default command once from its fresh `3d-unet/work` directory.
Setup precedes four-process CPU preprocessing; the single GPU stage begins only
after preprocessing completes. There is one target, so no other ready CPU stage can
overlap the GPU stage. Estimated full duration is 15--25 minutes: under one minute
for setup, roughly 5--12 minutes for preprocessing, about 6 minutes for inference,
and several minutes for postprocessing/evaluation.

Reserve 10 minutes of the one-hour limit for artifact validation and `report.json`.
The projected run fits with more than 25 minutes of margin.

## Re-plan during full run

Trigger: the first 10 of 43 inference cases completed. Their measured throughput was
about 0.045 seconds per sliding window, versus the conservative preflight rate of
0.128 seconds/window. The full 2,761-window GPU stage therefore projected to about
124 seconds and completed without requiring a schedule change. Preprocessing also
completed in roughly two minutes. The revised end-to-end projection is under 10
minutes plus evaluator time, so the 10-minute reporting reserve remains intact.
