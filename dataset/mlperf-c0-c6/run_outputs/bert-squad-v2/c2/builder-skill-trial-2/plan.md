# Full-run plan

## Workload

Target `bert-squad` has four serial stages: environment build, full validation preprocessing, one local A100 GPU accuracy run, and benchmark-owned evaluation. Each stage consumes the preceding stage's files. There is one target, so no other GPU work is eligible to overlap.

## Measured costs and resources

- Bounded preflight from a clean environment took 3:14 wall clock and at most 1,953,112 KiB host RSS before its output-path check exposed a driver issue.
- After the path repair, GPU plus evaluation for eight examples took 1:21 wall clock and at most 1,941,240 KiB host RSS.
- Full preprocessing produced 10,570 examples and 10,833 windows of length 384.
- The GPU check immediately before preflight showed an NVIDIA A100-SXM4-80GB with 76,125 MiB free.
- Allowed prior replay evidence (`prior-artifacts/bert-squad/full-run-driver.log`) completed the same full workload in 14:49.61 with at most 1,943,456 KiB host RSS.

## Schedule and projection

Run `./reproduce.sh --target bert-squad --from environment` once with no sample bound. This clears and recreates the environment and every later script-owned output, copies a fresh work tree from `repo`, preprocesses all inputs, checks current GPU memory, runs GPU work serially, and evaluates all questions. No CPU-only target is waiting, so there is no useful independent overlap.

Projection is 20 minutes: the same-arm prior full measurement is 14:50, plus roughly five minutes for shared-GPU/filesystem variability. Hold 15 additional minutes for artifact checks and `report.json`, far below the two-hour limit. If the full run's first observable progress or a failure moves the projection past 1:45 elapsed, re-plan; otherwise complete the single serial run.

## Re-plan 1

Trigger: full inference reached 18:57, exceeding the original same-arm inference profile because several unrelated sessions simultaneously saturated the shared A100. A read-only status check showed device memory use falling from 13.6 GiB to 6.8 GiB as competing jobs completed, with 74.3 GiB free. Keep the canonical run unchanged and revise expected total wall time to 30 minutes. This remains far below the two-hour limit and preserves the 15-minute reporting reserve.
