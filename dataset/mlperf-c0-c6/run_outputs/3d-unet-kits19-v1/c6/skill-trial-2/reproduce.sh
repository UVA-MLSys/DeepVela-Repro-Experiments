#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=""
STAGE=""
FROM=""

usage() {
  echo "usage: $0 --target 3d-unet [--stage environment|preflight|preprocess|run|evaluate] [--from environment|preprocess|run|evaluate]" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --target) TARGET=${2:-}; shift 2 ;;
    --stage) STAGE=${2:-}; shift 2 ;;
    --from) FROM=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$TARGET" == "3d-unet" ]] || usage
[[ -z "$STAGE" || -z "$FROM" ]] || usage

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_INPUT=$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data
MODEL_INPUT=$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc
UV_CACHE_DIR=$CACHE_ROOT/uv
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

TARGET_DIR=$ROOT/$TARGET
WORK=$TARGET_DIR/work
OUT=${OUTPUT_DIR:-$TARGET_DIR/out}
VENV=$TARGET_DIR/venv
SRC=$WORK/source
LOADGEN_SRC=$WORK/loadgen-source
RAW_VIEW=$WORK/raw_data
PREPROCESSED=$WORK/preprocessed_data
POSTPROCESSED=$WORK/postprocessed_data
PREFLIGHT=$WORK/preflight
REPO_BENCH=$ROOT/repo/vision/medical_imaging/3d-unet-kits19

mkdir -p "$TARGET_DIR" "$WORK" "$OUT"

marker() { echo "$WORK/.$1.success"; }

clear_from() {
  case "$1" in
    environment)
      rm -rf "$VENV" "$SRC" "$LOADGEN_SRC" "$RAW_VIEW" "$PREPROCESSED" "$POSTPROCESSED" "$PREFLIGHT"
      rm -f "$(marker environment)" "$(marker preflight)" "$(marker preprocess)" "$(marker run)" "$(marker evaluate)"
      rm -f "$OUT/build.log" "$OUT/preprocess.log" "$OUT/run.log" "$OUT/evaluator.log" \
        "$OUT/accuracy.txt" "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
        "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json"
      ;;
    preprocess)
      rm -rf "$PREPROCESSED" "$POSTPROCESSED"
      rm -f "$(marker preprocess)" "$(marker run)" "$(marker evaluate)"
      rm -f "$OUT/preprocess.log" "$OUT/run.log" "$OUT/evaluator.log" "$OUT/accuracy.txt" \
        "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
        "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json"
      ;;
    run)
      rm -rf "$POSTPROCESSED"
      rm -f "$(marker run)" "$(marker evaluate)"
      rm -f "$OUT/run.log" "$OUT/evaluator.log" "$OUT/accuracy.txt" \
        "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
        "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json"
      ;;
    evaluate)
      rm -rf "$POSTPROCESSED"
      rm -f "$(marker evaluate)" "$OUT/evaluator.log" "$OUT/accuracy.txt"
      ;;
    *) usage ;;
  esac
}

preflight() {
  environment
  [[ -f "$(marker preflight)" ]] && return
  rm -rf "$PREFLIGHT"
  mkdir -p "$PREFLIGHT"
  cp -a "$SRC" "$PREFLIGHT/source"
  "$VENV/bin/python" - "$PREFLIGHT/source" <<'PY'
import json, sys
from pathlib import Path
source = Path(sys.argv[1])
case = "case_00084"
(source / "meta/inference_cases.json").write_text(json.dumps([case], indent=4) + "\n")
checks = json.loads((source / "meta/checksum_inference.json").read_text())
(source / "meta/checksum_inference.json").write_text(
    json.dumps({case + ".pkl": checks[case + ".pkl"]}, indent=4) + "\n"
)
PY
  local pf_free_mib
  pf_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
  {
    echo "Preflight case: case_00084 (largest estimated resampled volume)"
    echo "GPU memory free before preflight run: ${pf_free_mib} MiB"
    if (( pf_free_mib < 16000 )); then
      echo "Insufficient free GPU memory: need at least 16000 MiB" >&2
      exit 1
    fi
    cd "$PREFLIGHT/source"
    "$VENV/bin/python" preprocess.py --raw_data_dir "$RAW_VIEW" \
      --results_dir "$PREFLIGHT/preprocessed" --mode preprocess --num_proc 1
    CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print("pytorch device: cuda:0")
PY
    CUDA_VISIBLE_DEVICES=0 LOG_PATH="$PREFLIGHT/logs" SKIP_VERIFY_ACCURACY=1 \
      "$VENV/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_INPUT" --preprocessed_data_dir="$PREFLIGHT/preprocessed" \
      --mlperf_conf="$ROOT/repo/loadgen/mlperf.conf" --user_conf="$PREFLIGHT/source/user.conf"
    "$VENV/bin/python" accuracy_kits.py \
      --log_file "$PREFLIGHT/logs/mlperf_log_accuracy.json" \
      --preprocessed_data_dir "$PREFLIGHT/preprocessed" \
      --postprocessed_data_dir "$PREFLIGHT/postprocessed" --num_proc 1
  } 2>&1 | tee "$OUT/preflight.log"
  grep -q '^Accuracy: mean = ' "$OUT/preflight.log"
  touch "$(marker preflight)"
}

environment() {
  [[ -f "$(marker environment)" ]] && return
  # A retry after an interrupted build starts this stage from its own clean state.
  rm -rf "$VENV" "$SRC" "$LOADGEN_SRC" "$RAW_VIEW"
  {
    echo "Building Python environment with UV_CACHE_DIR=$UV_CACHE_DIR"
    test -d "$ROOT/repo"
    test -d "$RAW_INPUT"
    test -f "$MODEL_INPUT"
    uv venv --python python3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'torch==2.5.1' 'numpy==1.26.4' 'scipy==1.11.4' 'nibabel==5.2.1' \
      'pandas==1.5.3' 'pybind11==2.13.6' 'setuptools==75.3.0' wheel
    cp -a "$REPO_BENCH" "$SRC"
    cp -a "$ROOT/repo/loadgen" "$LOADGEN_SRC"
    uv pip install --python "$VENV/bin/python" --no-build-isolation "$LOADGEN_SRC"
    mkdir -p "$RAW_VIEW"
    while IFS= read -r case_dir; do
      ln -s "$case_dir" "$RAW_VIEW/$(basename "$case_dir")"
    done < <(find "$RAW_INPUT" -mindepth 1 -maxdepth 1 -type d | sort)
    if [[ ! -e "$RAW_VIEW/case_00400" ]]; then
      ln -s "$RAW_INPUT/case_00185" "$RAW_VIEW/case_00400"
    fi
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("nibabel", nibabel.__version__, "pandas", pandas.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
  } 2>&1 | tee "$OUT/build.log"
  touch "$(marker environment)"
}

preprocess() {
  [[ -f "$(marker preprocess)" ]] && return
  environment
  mkdir -p "$PREPROCESSED"
  (
    cd "$SRC"
    "$VENV/bin/python" preprocess.py \
      --raw_data_dir "$RAW_VIEW" --results_dir "$PREPROCESSED" \
      --mode preprocess --num_proc 4
  ) 2>&1 | tee "$OUT/preprocess.log"
  "$VENV/bin/python" - "$PREPROCESSED" <<'PY'
import pickle, sys
from pathlib import Path
p = Path(sys.argv[1])
with (p / "preprocessed_files.pkl").open("rb") as f:
    data = pickle.load(f)
assert len(data["file_list"]) == 43, len(data["file_list"])
assert set(data["file_list"]) == set(data["cases"]), "case metadata mismatch"
assert len(list(p.glob("*.pkl"))) == 44, "expected 43 volumes plus metadata"
print("validated 43 preprocessed inference cases")
PY
  touch "$(marker preprocess)"
}

run_benchmark() {
  [[ -f "$(marker run)" ]] && return
  preprocess
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
  {
    echo "GPU memory free before run: ${free_mib} MiB"
    if (( free_mib < 16000 )); then
      echo "Insufficient free GPU memory: need at least 16000 MiB" >&2
      exit 1
    fi
    CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print("pytorch device: cuda:0")
print("GPU:", torch.cuda.get_device_name(0))
PY
    cd "$SRC"
    CUDA_VISIBLE_DEVICES=0 LOG_PATH="$OUT" SKIP_VERIFY_ACCURACY=1 \
      "$VENV/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_INPUT" --preprocessed_data_dir="$PREPROCESSED" \
      --mlperf_conf="$ROOT/repo/loadgen/mlperf.conf" --user_conf="$SRC/user.conf"
  } 2>&1 | tee "$OUT/run.log"
  test -s "$OUT/mlperf_log_accuracy.json"
  test -s "$OUT/mlperf_log_detail.txt"
  touch "$(marker run)"
}

evaluate() {
  [[ -f "$(marker evaluate)" ]] && return
  run_benchmark
  mkdir -p "$POSTPROCESSED"
  (
    cd "$SRC"
    "$VENV/bin/python" accuracy_kits.py \
      --log_file "$OUT/mlperf_log_accuracy.json" \
      --preprocessed_data_dir "$PREPROCESSED" \
      --postprocessed_data_dir "$POSTPROCESSED" --num_proc 4
  ) 2>&1 | tee "$OUT/accuracy.txt" "$OUT/evaluator.log"
  grep -q '^Accuracy: mean = ' "$OUT/accuracy.txt"
  touch "$(marker evaluate)"
}

if [[ -n "$FROM" ]]; then
  clear_from "$FROM"
fi

if [[ -n "$FROM" ]]; then
  SELECTED=all
else
  SELECTED=${STAGE:-all}
fi

case "$SELECTED" in
  environment) environment ;;
  preflight) preflight ;;
  preprocess) preprocess ;;
  run) run_benchmark ;;
  evaluate|all) evaluate ;;
  *) usage ;;
esac
