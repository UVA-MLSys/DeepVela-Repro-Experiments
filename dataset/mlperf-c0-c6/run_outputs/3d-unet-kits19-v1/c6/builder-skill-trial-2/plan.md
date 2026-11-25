# Reproduction plan

## Workload

One target, `3d-unet`, runs four serial stages through the delivered `reproduce.sh`: (1) create a
Python 3.11 environment and compile the supplied LoadGen, (2) stage and preprocess the canonical
43-case inference set while verifying published checksums, (3) run fp32 PyTorch Offline accuracy
inference on the local A100 after checking free memory, and (4) evaluate the resulting LoadGen
accuracy log with `accuracy_kits.py`. Each stage reads the prior stage's files and writes a success
marker only after all its commands succeed.

## Measured costs and resources

The one-case end-to-end preflight measured:

- Environment plus LoadGen build: 53 seconds, CPU only, using the mandated uv cache.
- Preprocessing: 15–16 seconds for case_00000 (shape 1×192×384×384), using four of eight CPU cores;
  its repository checksum matched exactly.
- GPU inference: 4.8 seconds for 50 sub-volumes, with approximately 74 GiB free before launch.
- Evaluation: about 5 seconds for prediction reconstruction, NIfTI output, and Dice scoring.

The machine exposed 8 CPU cores, about 1.8 TiB available RAM, and an NVIDIA A100-SXM4-80GB. The
model load and bounded inference confirmed `cuda:0`. Other jobs share the GPU, so the script
requires at least 24,000 MiB free immediately before inference.

## Schedule and projection

There is one target, so GPU work is serialized. Environment build and preprocessing precede and
do not overlap inference because the latter consumes their files. Four preprocessing/evaluator
workers leave capacity for the system and other sessions. Extrapolating the measured large case
linearly and allowing for case-size variation gives up to 12 minutes for preprocessing, up to
10 minutes for inference, 3 minutes for evaluation, and 2 minutes for environment setup: a
conservative 27-minute workload projection. Allow 3 minutes for filesystem variance, for a
30-minute execution projection.

Reserve 5 minutes before the one-hour limit to validate all required files, parse exact printed
values into `report.json`, and test the deliverable script syntax. The projected 35 minutes
including reserve fits the remaining budget at launch.

## Re-plan triggers

After preprocessing and after the first several inference cases, compare observed time with the
projection. If the projection exceeds the reporting reserve, preserve measured outputs and write
`report.json` from the completed evaluator as required. A failed stage is repaired and rerun via
`./reproduce.sh --from <stage>`; successful earlier stages are retained unless the repaired stage
reads their outputs.
