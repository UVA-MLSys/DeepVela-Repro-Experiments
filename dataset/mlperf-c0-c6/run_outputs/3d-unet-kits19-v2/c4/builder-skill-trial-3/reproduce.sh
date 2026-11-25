#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START_STAGE=""
CASE_LIMIT="${REPRO_CASE_LIMIT:-0}"

usage() {
  echo "usage: $0 --target 3d-unet [--stage environment|preprocess|run] [--from environment|preprocess|run]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE="stage"; START_STAGE="${2:-}"; shift 2 ;;
    --from) MODE="from"; START_STAGE="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$TARGET" == "3d-unet" ]] || usage
if [[ "$MODE" != "all" ]]; then
  [[ "$START_STAGE" =~ ^(environment|preprocess|run)$ ]] || usage
fi
[[ "$CASE_LIMIT" =~ ^[0-9]+$ ]] || { echo "REPRO_CASE_LIMIT must be a nonnegative integer" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="$ROOT/repo"
TARGET_DIR="$ROOT/$TARGET"
OUTPUT_DIR="${OUTPUT_DIR:-$TARGET_DIR/out}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
WORK="$TARGET_DIR/work"
APP="$WORK/app"
LOADGEN="$WORK/loadgen"
VENV="$TARGET_DIR/python"
STATE="$TARGET_DIR/state"

CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
KITS19_RAW_DIR="${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_EXCLUDE_NEWER=2026-11-10

stages=(environment preprocess run)
stage_index() {
  case "$1" in environment) echo 0;; preprocess) echo 1;; run) echo 2;; esac
}

if [[ "$MODE" == "from" ]]; then
  start=$(stage_index "$START_STAGE")
  for ((i=start; i<${#stages[@]}; i++)); do
    rm -f "$STATE/${stages[$i]}.done"
  done
  if [[ "$START_STAGE" == "environment" ]]; then
    rm -rf "$WORK" "$VENV"
  elif [[ "$START_STAGE" == "preprocess" ]]; then
    rm -rf "$APP/build/preprocessed_data" "$APP/build/raw_data"
  fi
  if (( start <= 2 )); then
    rm -rf "$APP/build/postprocessed_data"
    rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt"
  fi
fi
mkdir -p "$STATE"

run_environment() {
  if [[ -f "$STATE/environment.done" ]]; then return; fi
  : > "$OUTPUT_DIR/build.log"
  {
    echo "stage: environment"
    echo "source: $REPO"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    test -f "$REPO/vision/medical_imaging/3d-unet-kits19/run.py"
    test -f "$REPO/loadgen/setup.py"
    test -f "$MODEL_DIR/3dunet_kits19_pytorch.ptc"
    test -d "$KITS19_RAW_DIR"
    mkdir -p "$WORK"
    rm -rf "$APP" "$LOADGEN"
    cp -a "$REPO/vision/medical_imaging/3d-unet-kits19" "$APP"
    cp -a "$REPO/loadgen" "$LOADGEN"
    sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$APP/pytorch_SUT.py"
    sed -i '/df.set_index("case", inplace=True)/a\    df.loc["mean"] = df.mean(numeric_only=True)' "$APP/accuracy_kits.py"
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'torch==2.4.1' 'numpy==1.26.4' 'scipy==1.11.4' 'nibabel==5.2.1' \
      'pandas==1.5.3' 'pybind11==2.13.6' 'setuptools==75.6.0' 'wheel==0.45.1'
    uv pip install --python "$VENV/bin/python" --no-build-isolation "$LOADGEN"
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "nibabel", nibabel.__version__)
PY
    mkdir -p "$APP/build/model"
    ln -sfn "$MODEL_DIR/3dunet_kits19_pytorch.ptc" "$APP/build/model/3dunet_kits19_pytorch.ptc"
    cp "$REPO/loadgen/mlperf.conf" "$APP/build/mlperf.conf"
  } 2>&1 | tee -a "$OUTPUT_DIR/build.log"
  touch "$STATE/environment.done"
}

run_preprocess() {
  if [[ -f "$STATE/preprocess.done" ]]; then return; fi
  test -f "$STATE/environment.done"
  : > "$OUTPUT_DIR/preprocess.log"
  {
    echo "stage: preprocess"
    rm -rf "$APP/build/raw_data" "$APP/build/preprocessed_data"
    mkdir -p "$APP/build/raw_data" "$APP/build/preprocessed_data"
    while IFS= read -r case_name; do
      if [[ "$case_name" == "case_00400" ]]; then
        ln -s "$KITS19_RAW_DIR/case_00185" "$APP/build/raw_data/case_00400"
      else
        test -d "$KITS19_RAW_DIR/$case_name"
        ln -s "$KITS19_RAW_DIR/$case_name" "$APP/build/raw_data/$case_name"
      fi
    done < <("$VENV/bin/python" - "$APP/meta/inference_cases.json" "$CASE_LIMIT" <<'PY'
import json, sys
cases = json.load(open(sys.argv[1]))
limit = int(sys.argv[2])
if limit:
    cases = cases[:limit]
for case in cases:
    print(case)
PY
    )
    if (( CASE_LIMIT > 0 )); then
      "$VENV/bin/python" - "$APP/meta/inference_cases.json" "$APP/meta/checksum_inference.json" "$CASE_LIMIT" <<'PY'
import json, sys
manifest_path, checksum_path, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
cases = json.load(open(manifest_path))[:limit]
checksums = json.load(open(checksum_path))
with open(manifest_path, "w") as f:
    json.dump(cases, f, indent=4)
with open(checksum_path, "w") as f:
    json.dump({case + ".pkl": checksums[case + ".pkl"] for case in cases}, f, indent=4, sort_keys=True)
print("bounded preflight cases:", cases)
PY
    fi
    cd "$APP"
    "$VENV/bin/python" preprocess.py --raw_data_dir build/raw_data \
      --results_dir build/preprocessed_data --mode preprocess --num_proc 4
  } 2>&1 | tee -a "$OUTPUT_DIR/preprocess.log"
  touch "$STATE/preprocess.done"
}

run_gpu() {
  if [[ -f "$STATE/run.done" ]]; then return; fi
  test -f "$STATE/preprocess.done"
  : > "$OUTPUT_DIR/gpu_memory.log"
  nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader | tee "$OUTPUT_DIR/gpu_memory.log"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if (( free_mib < 12000 )); then
    echo "Insufficient free GPU memory: ${free_mib} MiB; need at least 12000 MiB" | tee -a "$OUTPUT_DIR/gpu_memory.log" >&2
    exit 1
  fi
  rm -rf "$APP/build/postprocessed_data"
  mkdir -p "$APP/build/postprocessed_data"
  : > "$OUTPUT_DIR/run.log"
  cd "$APP"
  export LOG_PATH="$OUTPUT_DIR"
  export SKIP_VERIFY_ACCURACY=1
  "$VENV/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
    --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
    --preprocessed_data_dir="$APP/build/preprocessed_data" 2>&1 | tee "$OUTPUT_DIR/run.log"
  : > "$OUTPUT_DIR/evaluator.log"
  "$VENV/bin/python" accuracy_kits.py \
    --preprocessed_data_dir="$APP/build/preprocessed_data" \
    --postprocessed_data_dir="$APP/build/postprocessed_data" \
    --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" 2>&1 \
    | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  touch "$STATE/run.done"
}

run_named_stage() {
  case "$1" in
    environment) run_environment ;;
    preprocess) run_preprocess ;;
    run) run_gpu ;;
  esac
}

if [[ "$MODE" == "stage" ]]; then
  run_named_stage "$START_STAGE"
elif [[ "$MODE" == "from" ]]; then
  start=$(stage_index "$START_STAGE")
  for ((i=start; i<${#stages[@]}; i++)); do run_named_stage "${stages[$i]}"; done
else
  for stage in "${stages[@]}"; do run_named_stage "$stage"; done
fi
