#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet [--stage env|prepare|run] [--from env|prepare|run]" >&2
  exit 2
}

TARGET=""
MODE="all"
START="env"
while (($#)); do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE="stage"; START="${2:-}"; shift 2 ;;
    --from) MODE="from"; START="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$TARGET" == "3d-unet" ]] || usage
[[ "$START" =~ ^(env|prepare|run)$ ]] || usage

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="$ROOT/repo"
TASK_DIR="$ROOT/$TARGET"
OUTPUT_DIR=${OUTPUT_DIR:-"$TASK_DIR/out"}
WORK_DIR="$TASK_DIR/work"
SRC_DIR="$WORK_DIR/src"
VENV="$WORK_DIR/venv"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$OUTPUT_DIR" "$WORK_DIR/markers"

stage_num() {
  case "$1" in env) echo 1;; prepare) echo 2;; run) echo 3;; esac
}

invalidate_from() {
  local n
  n=$(stage_num "$1")
  if ((n <= 1)); then
    rm -f "$WORK_DIR/markers/env.ok"
  fi
  if ((n <= 2)); then
    rm -f "$WORK_DIR/markers/prepare.ok"
    rm -rf "$SRC_DIR" "$WORK_DIR/raw_data" "$WORK_DIR/preprocessed_data"
  fi
  if ((n <= 3)); then
    rm -f "$WORK_DIR/markers/run.ok"
    rm -rf "$WORK_DIR/postprocessed_data"
    rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/run.log" \
      "$OUTPUT_DIR/evaluator.log"
  fi
}

run_env() {
  [[ -f "$WORK_DIR/markers/env.ok" ]] && return
  {
    echo "Creating Python environment"
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'numpy<2' 'scipy<1.13' 'nibabel<6' 'pandas<3' 'torch<2.8' 'pybind11==2.11.1' setuptools wheel
    rm -rf "$WORK_DIR/loadgen-src"
    cp -a "$REPO/loadgen" "$WORK_DIR/loadgen-src"
    uv pip install --python "$VENV/bin/python" "$WORK_DIR/loadgen-src"
    "$VENV/bin/python" -c 'import mlperf_loadgen, torch, nibabel, scipy, pandas; print("imports ok; torch", torch.__version__)'
  } 2>&1 | tee "$OUTPUT_DIR/build.log"
  touch "$WORK_DIR/markers/env.ok"
}

run_prepare() {
  [[ -f "$WORK_DIR/markers/prepare.ok" ]] && return
  {
    test -d "$RAW_INPUT"
    test -s "$MODEL_INPUT"
    mkdir -p "$SRC_DIR" "$WORK_DIR/raw_data"
    cp -a "$REPO/vision/medical_imaging/3d-unet-kits19/." "$SRC_DIR/"
    sed -i 's/df = df\.append(/df = df._append(/' "$SRC_DIR/accuracy_kits.py"
    cp "$REPO/mlperf.conf" "$WORK_DIR/mlperf.conf"
    cp -al "$RAW_INPUT"/. "$WORK_DIR/raw_data/"
    cp -a "$RAW_INPUT/case_00185" "$WORK_DIR/raw_data/case_00400"
    cp "$MODEL_INPUT" "$WORK_DIR/3dunet_kits19_pytorch.ptc"
    cd "$SRC_DIR"
    "$VENV/bin/python" preprocess.py \
      --raw_data_dir "$WORK_DIR/raw_data" \
      --results_dir "$WORK_DIR/preprocessed_data" \
      --mode preprocess --num_proc 4
  } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  touch "$WORK_DIR/markers/prepare.ok"
}

run_benchmark() {
  [[ -f "$WORK_DIR/markers/run.ok" ]] && return
  mkdir -p "$WORK_DIR/postprocessed_data"
  {
    echo "GPU memory immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    "$VENV/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
free, total = torch.cuda.mem_get_info(0)
print(f"pytorch device: cuda:0 (free={free // 2**20} MiB, total={total // 2**20} MiB)")
if free < 8 * 2**30:
    raise SystemExit("Less than 8 GiB GPU memory is free")
PY
    cd "$SRC_DIR"
    export LOG_PATH="$OUTPUT_DIR"
    export SKIP_VERIFY_ACCURACY=1
    "$VENV/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model="$WORK_DIR/3dunet_kits19_pytorch.ptc" \
      --preprocessed_data_dir="$WORK_DIR/preprocessed_data" \
      --mlperf_conf="$WORK_DIR/mlperf.conf"
  } 2>&1 | tee "$OUTPUT_DIR/run.log"
  {
    cd "$SRC_DIR"
    "$VENV/bin/python" accuracy_kits.py \
      --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="$WORK_DIR/preprocessed_data" \
      --postprocessed_data_dir="$WORK_DIR/postprocessed_data" \
      --num_proc=4
  } 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  touch "$WORK_DIR/markers/run.ok"
}

if [[ "$MODE" == "from" ]]; then
  invalidate_from "$START"
fi

start_n=$(stage_num "$START")
for stage in env prepare run; do
  n=$(stage_num "$stage")
  if [[ "$MODE" == "stage" ]]; then
    ((n == start_n)) || continue
  else
    ((n >= start_n)) || continue
  fi
  case "$stage" in
    env) run_env;;
    prepare) run_prepare;;
    run) run_benchmark;;
  esac
done
