# Reproduction plan

Written after the successful bounded preflight on 2026-11-22.

## Workload and dependencies

There is one target, `bert-squad`, with 10,833 tokenized validation features derived from
10,570 SQuAD v1.1 questions. Its stages are: (1) stage the source and pinned
DeepLearningExamples tree, create Python 3.12 environment, and compile source-tree LoadGen;
(2) check current GPU free memory; (3) run the complete PyTorch fp32 Offline AccuracyOnly
pass serially on the local A100; (4) run `accuracy-squad.py`/`evaluate_v1.1.py` and preserve
the three required outputs plus logs.

## Measured costs and resources

- Clean environment setup: approximately 60 seconds, including a 28-31 second LoadGen
  extension build. It used CPU only and 8 available logical cores.
- Bounded end-to-end preflight: 8 LoadGen features, full feature-cache construction, model
  load, inference, and official evaluator completed in 78 seconds. Evaluator output was
  `{"exact_match": 75.0, "f1": 75.0}`; this is evidence only, not the result.
- LoadGen's bounded GPU interval was about 1.5 seconds for 8 serial samples, approximately
  0.188 seconds/feature including its fixed interval. The full feature cache contains
  10,833 features.
- Immediately before the successful bounded GPU run, `nvidia-smi` reported 77,503 MiB free
  of 81,920 MiB on the shared NVIDIA A100-SXM4-80GB. The wrapper requires at least 4,096
  MiB free and records a fresh check for the full run.
- Host memory is about 2 TiB with about 1.9 TiB available at planning time; preprocessing is
  CPU-side and does not contend materially with the GPU run.

## Schedule and projection

Run the delivered `./reproduce.sh --target bert-squad` once from a clean target directory.
It performs the CPU build first, then the single GPU run, then CPU evaluation. No other GPU
work from this session will overlap it. With only one target, there is no ready independent
CPU target to overlap during inference.

The conservative full inference projection is `10,833 * 0.188 s = 2,036 s` (34 minutes).
Allowing 2 minutes for clean setup, 2 minutes for preprocessing/model load, 5 minutes for
postprocessing/evaluation, and shared-GPU variance gives an expected total of 43 minutes.
Reserve 15 minutes before the two-hour limit for `report.json`, artifact validation, and a
final clean-script check. The projected completion is therefore comfortably before the
105-minute launch-to-reserve boundary.

## Re-plan triggers

After early full-run progress is visible, compare the observed feature rate with 0.188
seconds/feature. If projection exceeds 105 minutes total, record the new projection here;
do not change model, backend, precision, dataset, scenario, preprocessing, or evaluator.
On failure, group by the first causal log line, repair the delivered script, and rerun from
the affected script stage using its `--from` mechanism.

## Re-plan checkpoint: early full run

At 2026-11-22T23:50:10-05:00, the accuracy log had reached 4.2 MiB, approximately 6% of
the projected full log size, after roughly two minutes of LoadGen inference. Several
unrelated sessions were sharing the GPU; `nvidia-smi pmon` attributed about 49% SM to this
run in that sample, with 68,847 MiB GPU memory still free overall. The resulting 32-35
minute inference projection agrees with the original plan, so stage order, parallelism,
43-minute total estimate, and 15-minute reserve remain unchanged.

## Completion checkpoint

The clean full run finished at 2026-11-23T00:08:45-05:00. From the build-log completion
timestamp to the run success marker was about 21 minutes, faster than the 43-minute total
plan despite concurrent unrelated GPU jobs. LoadGen recorded all 10,833 unique QSL indices
and reported no warnings or errors. The official evaluator printed exact match
`83.68968779564806` and F1 `90.87487229720105`; no re-plan or semantic departure was
required.
