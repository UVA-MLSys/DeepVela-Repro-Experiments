#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --from) FROM="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "3d-unet" ]] || { echo "--target must be 3d-unet" >&2; exit 2; }
[[ -z "$STAGE" || -z "$FROM" ]] || { echo "use only one of --stage and --from" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$ROOT_DIR/repo"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
KITS19_RAW_DIR="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR="$CACHE_ROOT/3d-unet-kits19/model"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/$TARGET/out}"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
[[ "$OUTPUT_DIR" != / && "$OUTPUT_DIR" != "$ROOT_DIR" ]] || { echo "unsafe OUTPUT_DIR" >&2; exit 2; }
WORK_DIR="${OUTPUT_DIR}.work"
SRC_DIR="$WORK_DIR/src"
VENV_DIR="$WORK_DIR/venv"
PREP_DIR="$WORK_DIR/preprocessed_data"
RAW_VIEW="$WORK_DIR/raw_data"
POST_DIR="$WORK_DIR/postprocessed_data"

stage_env() {
  mkdir -p "$OUTPUT_DIR" "$WORK_DIR"
  rm -rf "$SRC_DIR" "$VENV_DIR"
  cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" "$SRC_DIR"
  cp "$REPO_DIR/loadgen/mlperf.conf" "$SRC_DIR/mlperf.conf"
  : > "$SRC_DIR/audit.conf"
  SRC_DIR="$SRC_DIR" python3 - <<'PY'
from pathlib import Path
import os
p = Path(os.environ["SRC_DIR"])
g = p / "global_vars.py"
s = g.read_text()
assert "TARGET_SPACING = None" in s
g.write_text(s.replace("TARGET_SPACING = None", "TARGET_SPACING = [1.6, 1.2, 1.2]"))
q = p / "pytorch_SUT.py"
s = q.read_text()
needle = "        self.model.eval()\n"
assert needle in s
q.write_text(s.replace(needle, needle + '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n'))
u = p / "inference_utils.py"
s = u.read_text()
u.write_text(s.replace("signal.gaussian(n, std)", "signal.windows.gaussian(n, std)"))
v = p / "preprocess.py"
s = v.read_text()
old = '''    if args.mode == "preprocess":
        preprocess_with_multiproc(args)
        verify_dataset(args)'''
new = '''    if args.mode == "preprocess":
        preprocess_with_multiproc(args)
        checksum_file = CHECKSUM_CALIB_FILE if args.calibration else CHECKSUM_INFER_FILE
        if checksum_file.is_file():
            verify_dataset(args)
        else:
            print("Checksum manifest absent in source snapshot; preprocessing completed without hash verification.")'''
assert old in s
v.write_text(s.replace(old, new))
PY
  if [[ -n "${PREFLIGHT_CASE:-}" ]]; then
    SRC_DIR="$SRC_DIR" PREFLIGHT_CASE="$PREFLIGHT_CASE" python3 - <<'PY'
from pathlib import Path
import json, os
p = Path(os.environ["SRC_DIR"])
case = os.environ["PREFLIGHT_CASE"]
cases = json.loads((p / "meta/inference_cases.json").read_text())
assert case in cases
(p / "meta/inference_cases.json").write_text(json.dumps([case], indent=2) + "\n")
c = p / "mlperf.conf"
s = c.read_text()
assert "3d-unet.Offline.min_query_count = 43" in s
c.write_text(s.replace("3d-unet.Offline.min_query_count = 43", "3d-unet.Offline.min_query_count = 1"))
PY
  fi
  uv venv --python 3.10 "$VENV_DIR"
  uv pip install --offline --python "$VENV_DIR/bin/python" \
    'torch==2.4.1' 'numpy==1.26.4' 'scipy==1.10.1' 'nibabel==5.3.2' 'pandas==1.5.3' 'pybind11==2.11.1' 'setuptools<76' wheel
  uv pip install --offline --python "$VENV_DIR/bin/python" "$REPO_DIR/loadgen"
  "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "nibabel", nibabel.__version__)
PY
  touch "$WORK_DIR/env.success"
}

stage_preprocess() {
  [[ -f "$WORK_DIR/env.success" ]] || { echo "environment stage missing" >&2; exit 1; }
  rm -rf "$RAW_VIEW" "$PREP_DIR" "$POST_DIR"
  mkdir -p "$RAW_VIEW" "$PREP_DIR" "$POST_DIR"
  SRC_DIR="$SRC_DIR" KITS19_RAW_DIR="$KITS19_RAW_DIR" RAW_VIEW="$RAW_VIEW" python3 - <<'PY'
from pathlib import Path
import json, os, shutil
src = Path(os.environ["SRC_DIR"])
raw = Path(os.environ["KITS19_RAW_DIR"])
view = Path(os.environ["RAW_VIEW"])
cases = json.loads((src / "meta/inference_cases.json").read_text())
for case in cases:
    if case == "case_00400":
        duplicate = view / case
        duplicate.mkdir()
        for name in ("imaging.nii.gz", "segmentation.nii.gz"):
            shutil.copy2(raw / "case_00185" / name, duplicate / name)
    else:
        (view / case).symlink_to(raw / case, target_is_directory=True)
print("prepared raw view for", len(cases), "cases")
PY
  (
    cd "$SRC_DIR"
    "$VENV_DIR/bin/python" preprocess.py --raw_data_dir "$RAW_VIEW" --results_dir "$PREP_DIR" --mode preprocess --num_proc 4
  )
  touch "$WORK_DIR/preprocess.success"
}

stage_run() {
  [[ -f "$WORK_DIR/preprocess.success" ]] || { echo "preprocess stage missing" >&2; exit 1; }
  rm -rf "$POST_DIR"
  mkdir -p "$POST_DIR"
  rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  [[ "$free_mib" =~ ^[0-9]+$ && "$free_mib" -ge 20000 ]] || { echo "need at least 20000 MiB free GPU memory, found $free_mib" >&2; exit 1; }
  (
    cd "$SRC_DIR"
    LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 "$VENV_DIR/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
      --preprocessed_data_dir="$PREP_DIR" \
      --mlperf_conf="$SRC_DIR/mlperf.conf" \
      --user_conf="$SRC_DIR/user.conf" \
      --audit_conf="$SRC_DIR/audit.conf"
  )
  (
    cd "$SRC_DIR"
    "$VENV_DIR/bin/python" accuracy_kits.py \
      --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="$PREP_DIR" \
      --postprocessed_data_dir="$POST_DIR" --num_proc 4
  ) | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  touch "$WORK_DIR/run.success"
}

run_logged() {
  local name="$1"
  shift
  "$@" 2>&1 | tee "$OUTPUT_DIR/$name.log"
}

stages=(env preprocess run)
if [[ -n "$FROM" ]]; then
  found=0
  for s in "${stages[@]}"; do
    if [[ "$s" == "$FROM" ]]; then found=1; fi
    if [[ $found -eq 1 ]]; then
      case "$s" in
        env) rm -f "$WORK_DIR/env.success" "$WORK_DIR/preprocess.success" "$WORK_DIR/run.success" ;;
        preprocess) rm -f "$WORK_DIR/preprocess.success" "$WORK_DIR/run.success" ;;
        run) rm -f "$WORK_DIR/run.success" ;;
      esac
    fi
  done
  [[ $found -eq 1 ]] || { echo "unknown --from stage: $FROM" >&2; exit 2; }
fi

for s in "${stages[@]}"; do
  if [[ -n "$STAGE" && "$s" != "$STAGE" ]]; then continue; fi
  if [[ -n "$FROM" ]]; then
    [[ "$s" == "$FROM" ]] && FROM=""
    [[ -z "$FROM" ]] || continue
  fi
  case "$s" in
    env) [[ -f "$WORK_DIR/env.success" ]] || run_logged build stage_env ;;
    preprocess) [[ -f "$WORK_DIR/preprocess.success" ]] || run_logged preprocess stage_preprocess ;;
    run) [[ -f "$WORK_DIR/run.success" ]] || run_logged run stage_run ;;
  esac
done
