# Reproduction plan

Recorded 2026-11-23 after bounded preflight and before the full run.

## Workload and dependencies

The sole target is `bert-squad`. `reproduce.sh` runs three ordered stages: `environment`
(fresh Python 3.11 environment and repository LoadGen build), `prepare` (CPU-only deterministic
feature records from the fixed validation JSON and vocabulary), and `run` (one A100 Offline
AccuracyOnly pass followed by `accuracy-squad.py` and `evaluate_v1.1.py`). Each stage writes a
success marker only after all of its commands complete.

## Measured costs

| Stage | Measurement | Resource use |
|---|---:|---|
| Environment | about 45 seconds from an empty target directory | 8 host CPUs available, shared package-cache reads and C++ compilation, no GPU |
| Prepare | about 55 seconds for 10,570 examples / 10,833 windows | CPU-only; 384 tokens, stride 128, max query 64 |
| Bounded run/evaluator | about 22 seconds for 8 windows | one A100; about 75.5 GiB free immediately before launch |

The permitted prior-trial `prior-artifacts/plan.md` records two complete clean replays of the
same source/input contract: 1,332.406 seconds under contention and 629.685 seconds with less
contention. Those full-run observations are stronger estimates than extrapolating eight startup-
dominated samples.

## Resources and schedule

The host exposes 8 CPU cores, about 1.9 TiB available RAM, and one shared NVIDIA
A100-SXM4-80GB. At preflight, other sessions used roughly 5.6 GiB in total and about 75.5 GiB was
free. Only this target uses the GPU, so there is no independent ready CPU target to overlap. The
schedule is environment -> prepare -> GPU run/evaluation. `reproduce.sh` checks free memory again
immediately before the accelerator stage and does not touch other sessions' processes.

## Expected time and reserve

Budget 25 minutes for the full clean replay, using the slower prior replay plus margin for current
contention, and 5 minutes for metric extraction, `report.json`, output validation, and script
audit. This 30-minute projection is well inside the 120-minute limit. If early progress or a
failure changes the projection materially, update this file before starting new work.

The completed clean replay stayed within this plan: LoadGen finished without warnings or errors,
the evaluator printed both required metrics, and no re-plan was necessary.
