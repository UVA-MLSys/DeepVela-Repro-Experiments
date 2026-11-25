---
name: benchmark-runner
description: Extend a successful unprofiled benchmark reproduction into an optional, environment-neutral workload runner and profiling workflow. Use when the user wants repeatable plain runs, profiling, or a runbook for an HPC or ML workload. The workload and profiling backend are configured by the user; this skill assumes no vendor, model, scheduler, cluster, or filesystem.
---

# Run and profile a benchmark workload

This skill starts after the workload has a successful, unprofiled reproduction command. It produces
an explicit runbook and optional profiling artifacts; it does not silently change the benchmark's
backend, precision, inputs, evaluator, or generation semantics.

## 1. Establish the plain run

Read the benchmark card, `reproduce.sh`, environment instructions, scheduler configuration, input
counts, expected output, and timing line. Confirm the source revision, workload size, device count,
launch layout, correctness check, and output directory. If the command has not succeeded
unprofiled, hand back to `baseline-reproduce` and stop here.

Copy `scripts/workload.conf.template` to a workload-local configuration and fill in:

- required files and input staging;
- environment setup through the `setup_env` hook;
- the application command as a bash array;
- the application-owned timing matcher;
- an optional `profile_cmd` that wraps the same application command with the selected profiler.

Do not put account names, partitions, reservations, cache paths, model paths, scheduler settings,
or host-specific defaults in the skill. Supply those through the local configuration or launch
command.

## 2. Generate and validate the runbook

Run:

```bash
bash scripts/make_runbook.sh path/to/workload.conf
DRY_RUN=1 bash scripts/run_plain.sh path/to/workload.conf
DRY_RUN=1 bash scripts/run_profile.sh path/to/workload.conf
```

The dry run must show the actual application and profile commands without launching them. The user
launches the scripts with the process manager or scheduler appropriate to the environment; the skill
does not submit jobs. Stage inputs into a per-run directory so the source tree remains unchanged.

## 3. Run the least expensive characterization first

Run or prepare these stages independently:

1. plain workload: correctness and reference timing;
2. optional profiler wrapper: counters, traces, telemetry, or another user-selected profile;
3. generic CSV summary with `scripts/summarize_profile.py`, when the profiler emits CSV.

The supplied scripts do not interpret vendor-specific counters or invent hardware-derived metrics.
If a profiler needs a counter plan, keep that plan in the workload directory and validate it with
`scripts/check_counters.py` against that profiler's own available-counter listing. Backend-specific
collection and post-processing belong to the backend configuration or a separately scoped skill.

## 4. Keep profiling out of the canonical result

The canonical reproduction remains the unprofiled command and its evaluator output. Profilers,
counter collection, sampling wrappers, and their overhead belong in separate run directories and
reports. Never replace the canonical timing or quality result with a profiled run. Record profiler
version, configuration, hardware identity, sampling settings, and known overhead beside the
profiling artifacts.

## Hand-off

```text
successful unprofiled reproduction -> benchmark-runner -> separate profiling report
```

If profiling reveals a route change, propose it under `baseline-reproduce`'s approval rule. Do not
edit or relaunch a benchmark route merely to make profiling easier.
