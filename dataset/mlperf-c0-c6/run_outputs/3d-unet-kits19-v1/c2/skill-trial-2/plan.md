# Full reproduction plan

## Workload

- Target `3d-unet`: build a Python 3.11 environment and the supplied LoadGen extension; preprocess all 43 cases from `meta/inference_cases.json`; run the fp32 TorchScript model in LoadGen Offline AccuracyOnly mode on the local A100; run the repository Dice evaluator.
- The raw data and model remain read-only. `case_00400` is represented in target-owned raw-data links by the documented duplicate `case_00185`.

## Measured costs and resources

- Environment plus one complete bounded case took 2:01 wall clock; LoadGen compilation was about 46 seconds.
- Bounded preprocessing and checksum verification used less than 10 seconds for `case_00061`; peak host RSS for the complete bounded run was 2.94 GB.
- Its 36 sliding windows took 5.80 seconds in the slower of two checks (6.20 windows/s). NIfTI headers imply 2,761 windows for the full 43-case set, projecting 445 seconds (7.4 minutes) of model inference.
- Header-derived preprocessed image/label payload is about 6.6 GiB before NIfTI/compression overhead. The workspace has 4.6 TB free.
- The A100 has 80 GiB. Checks immediately before the bounded runs found 74.2 GiB free; the wrapper requires at least 20 GiB free before starting GPU work.

## Schedule

1. Run `reproduce.sh --target 3d-unet --from environment` as the clean full run.
2. Use eight CPU preprocessing/evaluator workers. There is one target, so no other GPU work is launched concurrently by this session.
3. Run inference only after preprocessing and checksum verification succeed, then evaluate the complete 43-case accuracy log.
4. After the first cases, compare observed inference seconds/window with the conservative 6.20 windows/s projection and update this plan if completion would exceed the reserve.

## Expected time and reserve

- Environment: 1–2 minutes.
- Full preprocessing and verification: conservatively 8–12 minutes based on the bounded case and 6.6 GiB derived payload.
- Inference: 7.4 minutes projected; allow 10 minutes for shared-GPU contention.
- Evaluation and final checks: 2–3 minutes.
- Expected total: 20–25 minutes. Hold 10 minutes of the one-hour limit for reporting and artifact verification.

## Re-plan: concurrent workspace collision

- Trigger: 15 preprocessed files appeared under `3d-unet/work`, while process inspection showed two reproductions started by other sessions already using the same directory and `OUTPUT_DIR`.
- Change: interrupted only this session's full-run PID and changed private intermediates to `3d-unet/.codex-reproduce-work-3fbc329`. No process from another session was signaled or modified.
- Projection: a clean restart remains 20–25 minutes, within the original reserve. GPU inference will not start until the shared output race has cleared.

### Collision audit correction

- Resolving `/proc/<pid>/cwd` showed that the similarly named processes run in sibling `c1`, `c2`, and `c3` workspaces, not this workspace. The partial files were from this session's interrupted run; no foreign process or artifact was modified.
- Keep the unique private intermediate path as a harmless isolation improvement and restart the clean full run immediately. Projection remains 20–25 minutes.

## Completion

- Clean full run wall time: 10:02; peak host RSS: 11.54 GB.
- Preprocessing produced and verified all 43 case files. Inference produced 43 unique QSL results and LoadGen reported no errors.
- Official evaluator result: `Accuracy: mean = 0.86173, kidney = 0.9347, tumor = 0.7888`.
- Required outputs and all stage logs passed the final artifact audit.
