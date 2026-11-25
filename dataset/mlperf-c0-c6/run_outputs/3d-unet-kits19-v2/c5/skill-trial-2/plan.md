# Reproduction plan

## Workload

One target, `3d-unet`: build a clean Python/LoadGen environment; preprocess all 43
manifest entries (materializing documented duplicate `case_00400` from `case_00185`);
run PyTorch FP32 Offline accuracy on the local A100; run the repository evaluator;
then report its printed mean, kidney, and tumor Dice values.

## Measured costs and resources

- Today's clean offline UV environment and repository LoadGen build: 88 seconds.
- Today's checksum-verified preprocessing of representative `case_00000`: 31 seconds
  with four of the machine's eight CPU cores.
- Today's GPU stage for that case: 59 seconds including model/QSL startup, with 9.00
  seconds reported for inference.
- Today's one-case postprocessing/evaluation: 8 seconds on four CPU processes.
- GPU immediately before today's preflight: 74,261 MiB free of 81,920 MiB.
- Prior trial 2's provenance-backed clean full run on the same target: 11 minutes 29
  seconds (51 seconds environment, 3 minutes 9 seconds preprocessing, 5 minutes 50
  seconds run, 1 minute 39 seconds evaluation).
- Host: 8 CPU cores, about 1.8 TiB available RAM, and 5.2 TiB filesystem free.

## Schedule and projection

Run the delivered script unchanged from its first stage. Environment, preprocessing,
GPU inference, and evaluation are serial dependencies for this sole target. Use four
CPU preprocessing/evaluation workers, leaving headroom for the system. Check free GPU
memory immediately before the sole GPU stage and require at least 12 GiB. The prior
full measurement is the strongest estimate; today's slower bounded run adds
contention margin, giving a conservative 12--18 minute projection. Reserve at least
10 minutes for artifact checks and `report.json`.

## Completion

The clean full run finished in 11 minutes 1 second: environment 1 minute 27
seconds, preprocessing 2 minutes 40 seconds, inference (including QSL loading and
accuracy-log writing) 5 minutes 38 seconds, and evaluation 1 minute 16 seconds. All
43 preprocessing case checksums passed, LoadGen reported 43 total and 43 accuracy
samples, the accuracy log contained 43 unique response indices, and all four stage
success markers were written. No re-plan was needed because the run remained inside
the 12--18 minute projection.
