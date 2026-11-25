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

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "supported target: 3d-unet" >&2
  exit 2
fi
if [[ -n "$STAGE" && -n "$FROM" ]]; then
  echo "use only one of --stage and --from" >&2
  exit 2
fi

WORKSPACE=$(pwd -P)
TARGET_DIR="$WORKSPACE/$TARGET"
OUTPUT_DIR=${OUTPUT_DIR:-"$TARGET_DIR/out"}
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR=$(realpath -m "$WORKSPACE/$OUTPUT_DIR")
fi
CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
KITS19_RAW_DIR=${KITS19_RAW_DIR:-"$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"}
MODEL_DIR=${MODEL_DIR:-"$CACHE_ROOT/3d-unet-kits19/model"}
UV_CACHE_DIR=${UV_CACHE_DIR:-"$CACHE_ROOT/uv"}
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

WORK_DIR="$TARGET_DIR/work"
SOURCE_DIR="$WORK_DIR/repo"
APP_DIR="$SOURCE_DIR/vision/medical_imaging/3d-unet-kits19"
VENV_DIR="$WORK_DIR/venv"
RAW_STAGE="$WORK_DIR/raw_data"
PREPROCESSED_DIR="$WORK_DIR/preprocessed_data"
POSTPROCESSED_DIR="$WORK_DIR/postprocessed_data"
MARKER_DIR="$TARGET_DIR/markers"
MODEL_FILE="$MODEL_DIR/3dunet_kits19_pytorch.ptc"

mkdir -p "$TARGET_DIR" "$MARKER_DIR"

validate_inputs() {
  [[ -d "$WORKSPACE/repo" ]] || { echo "missing fresh repository at $WORKSPACE/repo" >&2; exit 1; }
  [[ -d "$KITS19_RAW_DIR" ]] || { echo "missing raw data: $KITS19_RAW_DIR" >&2; exit 1; }
  [[ -s "$MODEL_FILE" ]] || { echo "missing model: $MODEL_FILE" >&2; exit 1; }
  [[ "$OUTPUT_DIR" == "$WORKSPACE"/* ]] || { echo "OUTPUT_DIR must be inside the workspace" >&2; exit 1; }
  [[ "$TARGET_DIR" == "$WORKSPACE/3d-unet" ]] || exit 1
}

reset_from() {
  case "$1" in
    environment)
      rm -rf "$WORK_DIR" "$OUTPUT_DIR" "$MARKER_DIR"
      mkdir -p "$MARKER_DIR"
      ;;
    preprocess)
      rm -rf "$RAW_STAGE" "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR" "$OUTPUT_DIR"
      rm -f "$MARKER_DIR/preprocess.ok" "$MARKER_DIR/run.ok"
      ;;
    run)
      rm -rf "$POSTPROCESSED_DIR" "$OUTPUT_DIR"
      rm -f "$MARKER_DIR/run.ok"
      ;;
    *) echo "invalid stage: $1" >&2; exit 2 ;;
  esac
}

stage_environment() {
  [[ -f "$MARKER_DIR/environment.ok" ]] && return
  rm -rf "$SOURCE_DIR" "$VENV_DIR"
  mkdir -p "$WORK_DIR"
  cp -a "$WORKSPACE/repo" "$SOURCE_DIR"
  {
    date -Ins
    echo "source snapshot task commit: 3fbc329939999c13d0a7b5e67fb2092287e06047"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    python311=$(command -v python3.11)
    [[ -n "$python311" ]] || { echo "Python 3.11 is required" >&2; exit 1; }
    "$python311" -c 'import os,sysconfig; assert os.path.isfile(os.path.join(sysconfig.get_path("include"), "Python.h")), "Python.h is required"'
    uv venv --python "$python311" "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" \
      'numpy==1.26.4' 'scipy==1.10.1' 'pandas==1.5.3' \
      'nibabel==5.2.1' 'torch==2.6.0' 'pybind11==2.11.1' \
      'setuptools==75.8.0' 'wheel==0.45.1'
    uv pip install --python "$VENV_DIR/bin/python" "$SOURCE_DIR/loadgen"
    "$VENV_DIR/bin/python" - <<'PY'
import nibabel, numpy, pandas, scipy, torch
import mlperf_loadgen
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("pandas", pandas.__version__)
print("nibabel", nibabel.__version__)
print("torch", torch.__version__)
print("torch CUDA build", torch.version.cuda)
print("loadgen import OK", mlperf_loadgen.__file__)
PY
  } >"$TARGET_DIR/build.log" 2>&1
  touch "$MARKER_DIR/environment.ok"
}

stage_preprocess() {
  [[ -f "$MARKER_DIR/preprocess.ok" ]] && return
  [[ -f "$MARKER_DIR/environment.ok" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
  rm -rf "$RAW_STAGE" "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR"
  mkdir -p "$RAW_STAGE" "$PREPROCESSED_DIR"
  local cases_file="$APP_DIR/meta/inference_cases.json"
  local checksum_file="$APP_DIR/meta/checksum_inference.json"
  if [[ -n "${REPRO_CASE_LIMIT:-}" ]]; then
    "$VENV_DIR/bin/python" - "$cases_file" "$checksum_file" "$REPRO_CASE_LIMIT" <<'PY'
import json, sys
cases_path, sums_path, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(cases_path) as f:
    cases = json.load(f)[:limit]
with open(sums_path) as f:
    sums = json.load(f)
with open(cases_path, "w") as f:
    json.dump(cases, f, indent=4)
with open(sums_path, "w") as f:
    json.dump({f"{c}.pkl": sums[f"{c}.pkl"] for c in cases}, f, indent=4, sort_keys=True)
PY
  fi
  mapfile -t cases < <("$VENV_DIR/bin/python" - "$cases_file" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(*json.load(f), sep="\n")
PY
  )
  for case_name in "${cases[@]}"; do
    if [[ "$case_name" == "case_00400" ]]; then
      ln -s "$KITS19_RAW_DIR/case_00185" "$RAW_STAGE/$case_name"
    else
      [[ -d "$KITS19_RAW_DIR/$case_name" ]] || { echo "missing raw case $case_name" >&2; exit 1; }
      ln -s "$KITS19_RAW_DIR/$case_name" "$RAW_STAGE/$case_name"
    fi
  done
  {
    date -Ins
    echo "preprocess case count: ${#cases[@]}"
    echo "case_00400 mapping: case_00185"
    cd "$APP_DIR"
    "$VENV_DIR/bin/python" preprocess.py \
      --raw_data_dir "$RAW_STAGE" --results_dir "$PREPROCESSED_DIR" \
      --mode preprocess --num_proc 8
  } >"$TARGET_DIR/preprocess.log" 2>&1
  touch "$MARKER_DIR/preprocess.ok"
}

stage_run() {
  [[ -f "$MARKER_DIR/run.ok" ]] && return
  [[ -f "$MARKER_DIR/preprocess.ok" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  rm -rf "$POSTPROCESSED_DIR" "$OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR" "$POSTPROCESSED_DIR"
  {
    date -Ins
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version \
      --format=csv,noheader
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 24000 )); then
      echo "need at least 24000 MiB free GPU memory; observed ${free_mib:-unknown}" >&2
      exit 1
    fi
    cuda_ok=$("$VENV_DIR/bin/python" -c 'import torch; print(int(torch.cuda.is_available()))')
    [[ "$cuda_ok" == 1 ]] || { echo "PyTorch CUDA is unavailable" >&2; exit 1; }
    echo "pytorch device: cuda:0"
    cd "$APP_DIR"
    LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 \
      "$VENV_DIR/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_FILE" --preprocessed_data_dir="$PREPROCESSED_DIR" \
      --mlperf_conf="$SOURCE_DIR/loadgen/mlperf.conf" --user_conf=user.conf
  } >"$TARGET_DIR/run.log" 2>&1
  {
    date -Ins
    cd "$APP_DIR"
    "$VENV_DIR/bin/python" accuracy_kits.py \
      --preprocessed_data_dir="$PREPROCESSED_DIR" \
      --postprocessed_data_dir="$POSTPROCESSED_DIR" \
      --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" --num_proc=1
  } 2>&1 | tee "$TARGET_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" ]]
  [[ -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
  [[ -s "$OUTPUT_DIR/accuracy.txt" ]]
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  grep -q '^pytorch device: ' "$TARGET_DIR/run.log"
  touch "$MARKER_DIR/run.ok"
}

validate_inputs
if [[ -n "$FROM" ]]; then
  reset_from "$FROM"
fi

run_one() {
  case "$1" in
    environment) stage_environment ;;
    preprocess) stage_preprocess ;;
    run) stage_run ;;
    *) echo "invalid stage: $1" >&2; exit 2 ;;
  esac
}

if [[ -n "$STAGE" ]]; then
  run_one "$STAGE"
else
  start="${FROM:-environment}"
  enabled=0
  for stage_name in environment preprocess run; do
    [[ "$stage_name" == "$start" ]] && enabled=1
    (( enabled == 0 )) || run_one "$stage_name"
  done
fi

