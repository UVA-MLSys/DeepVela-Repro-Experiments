# BERT SQuAD reproduction plan

## Workload

Target `bert-squad` runs three ordered stages: (1) copy the pinned benchmark and staged
NVIDIA BERT support code and build a uv environment plus LoadGen, (2) generate the complete
SQuAD v1.1 feature cache on CPU, and (3) run PyTorch fp32 AccuracyOnly in Offline mode on the
local A100 and invoke `accuracy-squad.py` / `evaluate_v1.1.py`.

Inputs are the supplied 10,570-question SQuAD v1.1 file, BERT-Large `model.pytorch`,
`vocab.txt`, and DeepLearningExamples commit tree. Preprocessing produces 10,833 features.

## Measured costs and resources

- Clean 8-feature setup-to-evaluator preflight: 2m25.20s elapsed, 2.49 GB peak host RSS.
- Warm 200-feature run plus evaluator: 37.03s elapsed, 2.34 GB peak host RSS.
- GPU checks immediately before bounded runs reported 79,215 MiB free of 81,920 MiB.
- Available route: one shared local NVIDIA A100-SXM4-80GB. GPU stages run serially. Setup,
  compilation, preprocessing, and evaluation are CPU work.

## Schedule and expected time

There is one target, so no cross-target overlap is useful. Run the delivered no-stage script from
nothing built: setup, complete preprocessing, one full GPU run, then evaluator. The 200-feature
measurement includes large fixed import/checkpoint/evaluator costs; even conservatively scaling
its entire 37.03s by 10,833/200 gives 33m26s. Adding the measured clean setup/preprocessing cost
still projects under 36 minutes. The expected actual duration is substantially lower because fixed
costs do not scale.

Reserve 15 minutes before the two-hour wall-clock limit for one repair/rerun if necessary,
`report.json`, artifact checks, and a copied-alone script validation. No new GPU work begins in
that reserve.

## Re-plan during full run

At 3m39s of full inference, GPU utilization was 100%, the incrementally written accuracy log was
54 MB, GPU memory usage was 5,988 MiB, and 75,166 MiB remained free. Observed progress projected
the inference stage to roughly 4–6 minutes, well ahead of the conservative plan, so stage order,
parallelism, and the 15-minute reserve were unchanged.
