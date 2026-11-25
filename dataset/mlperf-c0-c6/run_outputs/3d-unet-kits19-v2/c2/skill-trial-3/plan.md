# Full reproduction plan

## Workload

Target `3d-unet` runs three ordered stages: build an offline Python environment and
LoadGen from the pinned source, preprocess all 43 manifest cases (mapping documented
duplicate `case_00400` to `case_00185`), then run the fp32 TorchScript model in
LoadGen Offline accuracy mode and invoke `accuracy_kits.py` over its accuracy log.

## Measured costs and resources

- Preflight environment setup/build: 64 seconds.
- Preflight preprocessing: 76 seconds wall time for `case_00000`, including current
  shared-filesystem wait, and 14 seconds CPU time. It produced shape
  `(1, 192, 384, 384)`. The comparable prior full preprocessing completed in about
  two minutes; the full stage uses four of this machine's 8 CPU cores.
- Preflight inference: 50 sliding windows in 8.18899 seconds on `cuda:0`, or
  0.164 seconds/window. The prior artifact records 2,761 windows for the full scope,
  projecting about 452 seconds of model execution at today's bounded rate.
- Preflight end-to-end evaluation succeeded. Its one-case result is sizing evidence
  only and is not the benchmark result.
- Available resources before preflight launch: 8 CPU cores, 2.0 TiB RAM (1.8 TiB
  available), and one NVIDIA A100-SXM4-80GB with 74,261 MiB free immediately before
  inference. The script rechecks GPU memory and requires 12,000 MiB free.
- Workspace has 5.0 TiB free. The preflight occupies 5.6 GiB. All 43 logical input
  cases were checked for both image and segmentation files; none are missing.

## Schedule and expected time

Run the delivered default command once from its fresh `3d-unet/work` directory.
Setup precedes four-process CPU preprocessing; the single GPU stage begins only
after preprocessing completes. There is one target, so no other ready CPU stage can
overlap the GPU stage.

Estimated full duration is at most 25 minutes: about one minute for setup, up to
10 minutes for preprocessing under filesystem contention, about 8 minutes for
inference, and up to 6 minutes for postprocessing/evaluation. Reserve 10 minutes of
the one-hour limit for artifact validation and `report.json`; the projected run fits
with about 25 minutes of additional margin.

## Re-plan trigger

After the first several full-scope cases, compare observed windows/second and
preprocessing duration with this projection. If the projection exceeds the
50-minute execution budget, preserve the completed artifacts, record the revised
projection here, and begin final reporting at the 10-minute reserve.

## Full-run re-plan outcome

Trigger: all 43 preprocessing outputs and the first several inference cases became
available. Preprocessing completed in about four minutes, and the full 2,761-window
inference timings summed to about 189 seconds, versus the planned 452 seconds. The
projection therefore moved earlier, no schedule change was needed, and the complete
run plus evaluator finished successfully in 10 minutes 35 seconds. The 10-minute
reporting reserve remained intact.
