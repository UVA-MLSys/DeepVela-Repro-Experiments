# Reproduction plan

## Workload

One target, `3d-unet`: build a clean Python/LoadGen environment; preprocess all 43
manifest entries (materializing documented duplicate `case_00400` from `case_00185`);
run PyTorch FP32 Offline accuracy on the local A100; run the repository evaluator;
then report its printed mean, kidney, and tumor Dice values.

## Measured costs and resources

- Clean offline UV environment and repository LoadGen build: 110 seconds.
- Checksum-verified preprocessing of representative `case_00000`: 18 seconds with
  four of the machine's eight CPU cores.
- GPU inference for that case (preprocessed shape 1x192x384x384): 4.93 seconds.
- One-case postprocessing/evaluation: 6 seconds on four CPU processes.
- GPU immediately before the successful preflight: 78,276 MiB free of 81,920 MiB.
- Host: 8 CPU cores, about 1.8 TiB available RAM, 4.5 TiB filesystem free.

## Schedule and projection

Run the delivered script unchanged from its first stage. Environment, preprocessing,
GPU inference, and evaluation are serial dependencies for this sole target. Use four
CPU preprocessing/evaluation workers, leaving headroom for the system. Check free GPU
memory immediately before the sole GPU stage and require at least 12 GiB. The measured
representative case implies about 4 minutes of inference; allowing case-size variance,
preprocessing, environment setup, and evaluation gives a conservative 10--15 minute
projection. Reserve at least 10 minutes for artifact checks and `report.json`.

## Completion

The clean full run finished in 11 minutes 29 seconds: environment 51 seconds,
preprocessing 3 minutes 9 seconds, inference (including QSL loading and log writing)
5 minutes 50 seconds, and evaluation 1 minute 39 seconds. All 43 preprocessing
checksums and all 43 unique LoadGen response indices were verified. No re-plan was
needed because the run remained inside the 10--15 minute projection.
