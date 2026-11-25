# Reproduction plan

Recorded 2026-11-23 after the successful bounded preflight.

## Workload and dependencies

There is one target, `bert-squad`. Its setup stage copies the pinned source,
hash-checks and symlinks the immutable SQuAD/model/vocabulary inputs, creates a
Python 3.11 uv environment, and builds the supplied LoadGen. Its accelerator
stage preprocesses the validation set to 10,833 features, loads BERT-Large
fp32, runs every feature in Offline AccuracyOnly mode on the local A100, and
then invokes `accuracy-squad.py` on CPU. The evaluator depends on the complete
LoadGen accuracy log. GPU work is serial and there is no second target whose
CPU work can overlap it.

## Measured costs and resources

- Setup completed in about 41 seconds, including cached package installation
  and a clean local LoadGen build.
- The 8-feature end-to-end preflight completed in 77.64 seconds wall time,
  used 66.57 seconds user CPU, 6.34 seconds system CPU, and peaked at
  2,270,376 KiB host RSS. This includes model load, first feature conversion,
  GPU inference, and evaluator startup.
- LoadGen recorded 0.720 seconds for the 8-sample GPU section, or about 0.090
  seconds per feature. The resulting feature cache is 51 MiB.
- GPU preflight immediately before the bounded run recorded 77,377 MiB free
  on the A100. The current process environment exposes 8 CPU cores and about
  1.9 TiB available host memory.
- The read-only prior same-arm trial reports a 22m17s clean end-to-end replay,
  providing a conservative comparable-run check on the short preflight
  extrapolation (`prior-artifacts/plan.md`).

## Schedule, projection, and reserve

Preserve the bounded evidence separately, then invoke the delivered script's
no-stage path from a clean `./bert-squad/` target directory. It will redo setup,
check current free GPU memory, run the single full GPU workload, and evaluate
it. The measured per-feature rate projects 10,833 * 0.090 = 975 seconds (16.2
minutes) for inference. Allow 25 minutes end to end based on the measured
setup/preprocessing overhead and the prior clean replay. Hold 10 further
minutes for exact result parsing, `report.json`, artifact validation, and the
copied-alone script check. This fits inside the two-hour limit.

## Re-plan triggers

After the first few percent of responses appear, compare their rate with the
16.2-minute inference projection. If the projected finish exceeds 100 minutes,
preserve measured outputs and enter the reporting reserve rather than launch
another full run. On any failure, classify the first failing log line, repair
through the script, return to bounded preflight, and do not alter processes
started by other sessions.

## Re-plan record

- **15:30 on 2026-11-23, 3.8% complete:** the accuracy log contained 416 of
  10,833 responses. From 124 to 416 it grew by 292 responses in 48 seconds, or
  about 6.1 samples/s while the shared A100 was at 100% utilization. This
  projects about 29 minutes for the remaining inference and roughly 32 minutes
  end to end. The projection remains far below the 100-minute trigger, so the
  serial single-run schedule is unchanged.
- **15:51 on 2026-11-23, completed:** the clean no-stage invocation exited 0
  after 24m02.97s, within the 25-minute plan. GPU contention eased after the
  midpoint; the complete accuracy log contains all 10,833 unique features and
  the evaluator completed inside the reporting reserve.
