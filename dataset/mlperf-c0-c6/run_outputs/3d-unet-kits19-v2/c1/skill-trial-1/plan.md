# Full reproduction plan

- Workload: `3d-unet` environment build, 43-case verified preprocessing, one local
  A100 Offline fp32 accuracy run, then the repository evaluator.
- Preflight evidence: the provenance-tracked prior snapshot from the same arm and
  source contains a successful bounded end-to-end run. Representative
  `case_00000` used 50 sub-volumes and took 5.47 seconds for inference and about
  1.1 seconds for evaluation. GPU free memory was 78,276 MiB.
- Input checks: all 43 manifest entries have imaging and segmentation inputs;
  `case_00400` maps to the documented duplicate `case_00185`. The model SHA-256
  is `a0d14529e0a6dd0d0ec78c936502dc643d2fe601191d82cd89e66c51570ce1d4`.
- Resources now: 8 CPU cores, about 1.8 TiB available host memory, and 81,037 MiB
  of 81,920 MiB GPU memory free. Preprocessing/evaluation use 4 CPU workers; GPU
  work is serialized.
- Schedule: run the delivered script once from its unbuilt state. It builds the
  environment, preprocesses and checksum-verifies all cases, checks free GPU
  memory immediately before inference, performs one complete GPU run, then runs
  the official evaluator.
- Expected time: about 5 minutes for preprocessing, 6--9 minutes for inference,
  and 1--2 minutes for evaluation; the warm shared package cache should keep the
  environment build under 10 minutes. Total projection is under 30 minutes.
- Reserve: 5 minutes for report extraction, artifact validation, and checking the
  delivered script.

## Actual / re-plan

- The environment build completed in under one minute from the warm mandated
  cache. Full preprocessing completed for all 43 cases and repository checksum
  verification passed.
- Immediately before the full run, `nvidia-smi` reported 72,647 MiB free and
  PyTorch reported 70,617 MiB free. LoadGen completed all 43 cases in about 262
  seconds, within the original projection, so no schedule change was needed.
- The repository evaluator completed successfully. The accuracy log contains 43
  records with 43 unique QSL indices spanning 0 through 42.
- LoadGen's single detail error is its official-submission check that multiple
  configuration files were loaded. It does not indicate a failed query or
  evaluator, and the task's documented `run.py` route loads both `mlperf_conf`
  and `user_conf`.
