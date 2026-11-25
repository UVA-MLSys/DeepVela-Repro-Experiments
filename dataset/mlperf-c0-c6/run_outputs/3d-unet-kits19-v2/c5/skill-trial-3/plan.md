# Reproduction plan

## Workload

Target `3d-unet` has three stages: offline environment/work-copy setup, CPU
preprocessing of all 43 cases in `meta/inference_cases.json`, then one serial
A100 LoadGen Offline accuracy run followed by the repository evaluator. The
inputs contain 42 distinct supplied case directories; the setup stage maps the
documented duplicate `case_00400` to read-only `case_00185`.

## Measured costs and resources

- Clean bounded setup took 6.7 seconds and used at most 77 MiB RSS. Packages
  resolved offline from the required shared uv cache.
- Bounded preprocessing of `case_00000` took about 43 seconds with 16 CPU
  workers; its resulting image shape was `1x192x384x384`.
- Immediately before bounded inference, the local A100 80GB had 78,259 MiB
  free. Inference plus evaluator took about 43 seconds under observed shared-GPU
  contention and produced a complete LoadGen accuracy log and Dice result.
- The prior campaign's hash-verified full run measured about 9--10 seconds of
  inference for the same bounded case and approximately 2.8 GiB GPU use. Its
  full workload completed with all 43 cases and the same package set.

## Schedule and projection

Run `OUTPUT_DIR=./3d-unet/out ./reproduce.sh --target 3d-unet` once with no
stage option. Setup, preprocessing, one GPU run, and evaluation execute in
dependency order. GPU work is serial. The bounded inference rate projects about
7--23 minutes for 43 cases (uncontended versus current contention). Allowing 10
minutes for parallel preprocessing and setup/evaluation gives a conservative
projection below 35 minutes.

Reserve 10 minutes for extracting evaluator values into `report.json`, checking
the required output files and log prefix, and testing the runner's completed
stage behavior.
