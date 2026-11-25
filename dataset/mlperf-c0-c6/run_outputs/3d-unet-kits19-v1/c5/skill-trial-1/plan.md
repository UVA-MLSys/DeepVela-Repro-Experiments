# Reproduction plan

## Workload

- Target `3d-unet`: build an isolated Python/PyTorch and repository LoadGen environment; preprocess and checksum-verify all 43 cases in `meta/inference_cases.json`; run the fp32 TorchScript model through `run.py --backend=pytorch --scenario=Offline --accuracy`; evaluate its LoadGen accuracy log with `accuracy_kits.py`.
- The build produces the environment and repaired documented PyTorch adapter. Preprocessing depends on that environment and the fixed raw-data cache. The GPU run depends on all preprocessing and the fixed model. Evaluation depends on the full LoadGen accuracy log.

## Measured costs

- Bounded case: `case_00005`, the largest compressed inference input (313,477,305 bytes), preprocessed shape `(1, 256, 448, 448)`.
- Clean preflight build: about 2m17s (the 760 MB PyTorch wheel is now in the mandated shared uv cache); CPU only.
- One-case preprocessing plus checksum verification: about 1m21s using 8 processes; CPU only.
- One-case LoadGen inference: 13.35s; about 4 GiB attributable GPU memory, with 67,487 MiB free immediately before launch.
- One-case official evaluation: about 17s; CPU only.

## Resources and schedule

- Local host only; no Slurm. One NVIDIA A100 80GB, checked immediately before every GPU run. Current competing jobs leave over 60 GiB free, while preflight used about 4 GiB.
- Use 8 CPU preprocessing/evaluation processes to limit contention and memory pressure. There is one target, so no independent CPU stage remains to overlap once its GPU stage starts.
- Run the delivered script from `--from build`, which clears its own preflight work and outputs, then executes build, preprocess, GPU inference, and evaluation in order. Stage markers allow safe resumption without repeating successful work.

## Expected time and reserve

- Project 3–7 minutes for parallel full preprocessing, 10–15 minutes for 43-case GPU inference using the measured conservative case, and 2–4 minutes for evaluation and log copies. With the cached build, total projection is 15–25 minutes.
- Hold 8 minutes before the one-hour limit for `report.json`, artifact checks, and runner verification.
