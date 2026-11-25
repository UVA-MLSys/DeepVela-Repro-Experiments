# Reproduction plan: bert-squad

Written after successful bounded end-to-end preflight on 2026-11-22.

## Workload and dependencies

One target, `bert-squad`, runs four ordered stages through the delivered `reproduce.sh`: `env` (stage source and build the isolated Python/LoadGen environment) → `preprocess` (create all 10,833 SQuAD features) → `gpu` (BERT-Large FP32 Offline AccuracyOnly over all features) → `evaluate` (official postprocessing and SQuAD v1.1 EM/F1 evaluator). Each stage writes a success marker; `--from` invalidates that stage and all dependent outputs.

## Measured costs

| Stage | Bounded measurement | Resources |
|---|---:|---|
| Environment | 44 s (`17:49:36`–`17:50:20`) | CPU/build, 8 host cores available; shared UV cache |
| Preprocess | 57 s (`17:52:52`–`17:53:49`) | about one CPU core; all 10,833 features, not a subset |
| GPU | 17 s total startup + 8 samples (`17:53:49`–`17:54:06`) | one A100; 73,738 MiB free immediately before run; another session was using about 7.4 GiB and GPU utilization was 100% |
| LoadGen query execution | about 1.10 s for 8 samples from trace | observed rate about 7.27 features/s under contention |
| Evaluate | 7 s (`17:54:06`–`17:54:13`) | CPU; 8-question bounded output |

The bounded evaluator printed `{"exact_match": 75.0, "f1": 75.0}`. This is preflight evidence only, not a reported full result. The repaired run loaded the preprocessing cache at the launcher's expected fixed filename and emitted `pytorch device: cuda:0`.

## Resources and schedule

This machine exposes 8 CPU cores, about 1.9 TiB available host memory, and one shared NVIDIA A100-SXM4-80GB. GPU stages run locally and serially within this session, never via Slurm. There is only one target, so no independent CPU work remains to overlap with its GPU stage. The wrapper checks GPU free memory immediately before inference and requires at least 8,192 MiB.

Launch one clean full replay with `OUTPUT_DIR=./bert-squad/out ./reproduce.sh --target bert-squad --from env`. This deliberately rebuilds every stage using the delivered script. At the bounded rate, full query execution projects to about 1,490 s (25 min); adding environment, preprocessing, checkpoint startup, full postprocessing, and shared-GPU variance gives a conservative 40-minute projection.

## Time budget and reserve

The task limit is 2 hours. Reserve the final 20 minutes for evaluator completion, exact-value extraction into `report.json`, artifact/log checks, and a standalone-script audit. Start no additional experimental work once the reserve is reached. The projected 40-minute run plus 20-minute reserve fits with about 60 minutes of contingency.

## Re-plan triggers

After roughly the first few percent of full queries, use the live accuracy-log growth and GPU log/process time to compare against the 40-minute projection. Re-plan if the projected finish exceeds 100 minutes from full-run launch, if free GPU memory falls below the wrapper threshold, or if any stage fails. Repairs resume via `reproduce.sh --from <stage>`; successful earlier stages are not repeated unless the delivered clean-replay condition requires it.

## Re-plan record

- **2026-11-22 17:58 EST, trigger: first 5%.** The live accuracy log contained 567/10,833 records 97 seconds after the GPU-stage timestamp, including checkpoint startup. The observed aggregate rate was about 5.85 features/s, projecting GPU completion about 31 minutes after launch. This remains inside the conservative 40-minute run estimate, so stage order and parallelism are unchanged; expected total remains 35–40 minutes with the 20-minute reporting reserve intact.
