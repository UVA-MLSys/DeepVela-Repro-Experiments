# Full reproduction plan

## Workload

One target, `bert-squad`, runs three ordered stages through the delivered `reproduce.sh`:

1. `environment`: verify the three fixed input hashes, create a Python 3.11 virtual environment,
   install pinned runtime packages through the prescribed uv cache, and build the pinned local
   LoadGen source;
2. `preprocess`: link the allowed staged DeepLearningExamples dependency, apply the local QSL/device
   logging repair, and transform all 10,570 SQuAD questions into 10,833 BERT feature windows;
3. `run`: inspect GPU free memory, run all 10,833 windows through LoadGen in Offline AccuracyOnly
   mode on the A100, then invoke `accuracy-squad.py` and its repository evaluator.

Each stage depends on the preceding stage's files and writes a success marker only after all of its
commands exit successfully.

## Measured costs

- Environment setup: approximately 46 seconds on the first preflight, including the local LoadGen
  C++ extension build. Re-resolving pinned packages from the shared cache took under 2 seconds;
  LoadGen compilation took approximately 30 seconds.
- Full CPU preprocessing: approximately 20 seconds and produced 10,833 windows of length 384.
- Eight-window end-to-end preflight: passed LoadGen, required device logging, and the evaluator.
- 256-window run: 46.61 seconds end to end. `mlperf_log_detail.txt` measured 26.237 seconds from
  query issue through completion, or about 102.5 ms per window. Its accuracy evaluation also
  completed successfully.
- Model plus runtime GPU memory fits with a large margin: the two preflight launch checks observed
  74,324 MiB and 75,576 MiB free, respectively, on the 81,920 MiB A100.

## Resources and schedule

The machine currently exposes 8 CPU cores, about 1.9 TiB available system memory, and one shared
NVIDIA A100-SXM4-80GB GPU. Other jobs are using about 5.6 GiB of GPU memory at planning time. The
script runs CPU stages first and only one GPU run; there is no second target whose CPU work can
overlap it. It rejects a GPU launch with less than 8 GiB free and records `nvidia-smi` immediately
before inference.

The clean replay runs all stages sequentially from an unbuilt target directory. No Slurm job is
submitted. The 256-window rate projects the full inference to approximately 1,111 seconds (18.5
minutes). Adding environment construction, preprocessing, model load, evaluator work, filesystem
variance, and shared-GPU contention gives a conservative expected completion time of 25–35 minutes.

## Reserve

Reserve 20 minutes before the two-hour wall-clock limit for evaluator/report parsing, required-file
validation, and a final inspection of the standalone script and artifacts. The schedule therefore
has at least 65 minutes of additional margin beyond the conservative 35-minute projection.

## Execution outcome

The clean full replay completed all stages in 1,328.76 seconds (22.15 minutes), inside the expected
25–35 minute range. The full evaluator produced exact match 83.68968779564806 and F1
90.87487229720105.
