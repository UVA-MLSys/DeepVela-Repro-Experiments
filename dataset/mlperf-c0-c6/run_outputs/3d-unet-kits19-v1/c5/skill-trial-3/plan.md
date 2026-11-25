# Reproduction plan

## Workload

Target `3d-unet` has three stages: offline environment/work-copy setup, CPU preprocessing of all 43 cases in `meta/inference_cases.json`, then one serial A100 LoadGen accuracy run followed by the repository evaluator. Setup produces the environment and repaired work copy; preprocessing consumes the 42 supplied cases plus the documented `case_00400 -> case_00185` duplicate; inference consumes all preprocessed cases.

## Measured costs and resources

- Bounded setup from an empty target directory: about 40 seconds; shared-filesystem writes dominate while installing 4.3 GiB of cached CUDA/PyTorch packages.
- Bounded preprocessing: about 24 seconds for `case_00000` at shape `1x192x384x384`, using 16 of the machine's 128 CPU cores. Host memory available before launch was about 1.8 TiB.
- Bounded inference: about 9-10 seconds on the local NVIDIA A100 80GB; 74,260 MiB was free immediately before the run and the process used approximately 2.8 GiB.
- Bounded evaluator: about 8 seconds with 16 processes.

## Schedule and projection

Run `./reproduce.sh --target 3d-unet` once with no stage option and `OUTPUT_DIR=./3d-unet/out`. Stages run setup, preprocessing, inference, evaluation in dependency order. GPU work remains one run at a time. The one-case GPU rate projects 43 cases at about 7 minutes. Budgeting 10 minutes for parallel preprocessing, 2 minutes for setup/evaluation, and a 3x filesystem/contention allowance projects under 30 minutes total.

Reserve 10 minutes for extracting exact evaluator values into `report.json`, validating required files and log prefixes, and checking the standalone runner.
