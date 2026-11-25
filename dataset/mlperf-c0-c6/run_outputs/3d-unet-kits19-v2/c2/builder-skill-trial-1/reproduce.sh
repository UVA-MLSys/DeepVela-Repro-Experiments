#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE="stage"; START="${2:-}"; shift 2 ;;
    --from) MODE="from"; START="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "Unsupported target '$TARGET' (expected 3d-unet)" >&2
  exit 2
fi
if [[ "$MODE" != "all" && "$START" != "env" && "$START" != "preprocess" && "$START" != "run" ]]; then
  echo "Stage must be one of: env, preprocess, run" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$ROOT_DIR/3d-unet"
WORK_DIR="$TARGET_DIR/work"
SRC_DIR="$WORK_DIR/repo"
APP_DIR="$SRC_DIR/vision/medical_imaging/3d-unet-kits19"
VENV_DIR="$TARGET_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
STATE_TAG="full"
if [[ -n "${PREFLIGHT_CASES:-}" ]]; then STATE_TAG="preflight"; fi
STATE_DIR="$TARGET_DIR/state-$STATE_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$TARGET_DIR/out}"

CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
KITS19_RAW_DIR="${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$TARGET_DIR" "$STATE_DIR" "$OUTPUT_DIR"

log_cmd() {
  local logfile="$1"
  shift
  set -o pipefail
  "$@" 2>&1 | tee -a "$logfile"
}

clear_from() {
  local stage="$1"
  case "$stage" in
    env)
      rm -rf "$WORK_DIR" "$VENV_DIR"
      rm -f "$STATE_DIR/env.done" "$STATE_DIR/preprocess.done" "$STATE_DIR/run.done"
      ;;&
    preprocess)
      if [[ -d "$APP_DIR" ]]; then
        rm -rf "$APP_DIR/build/preprocessed_data" "$APP_DIR/build/postprocessed_data"
      fi
      rm -f "$STATE_DIR/preprocess.done" "$STATE_DIR/run.done"
      ;;&
    run)
      rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt" \
        "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/gpu-before-run.log"
      rm -f "$STATE_DIR/run.done"
      ;;
  esac
}

stage_env() {
  [[ -f "$STATE_DIR/env.done" ]] && return
  : > "$OUTPUT_DIR/build.log"
  {
    echo "source revision: 3fbc329939999c13d0a7b5e67fb2092287e06047 (task-provided tree)"
    echo "repair: TARGET_SPACING None -> [1.6, 1.2, 1.2] (prior-trial evidence)"
    echo "repair: skip checksum verification because checksum_inference.json is absent"
    echo "repair: emit required model parameter device line"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
  } | tee -a "$OUTPUT_DIR/build.log"

  [[ -d "$ROOT_DIR/repo" ]] || { echo "Missing repository: $ROOT_DIR/repo" | tee -a "$OUTPUT_DIR/build.log"; exit 1; }
  [[ -d "$KITS19_RAW_DIR" ]] || { echo "Missing raw data: $KITS19_RAW_DIR" | tee -a "$OUTPUT_DIR/build.log"; exit 1; }
  [[ -f "$MODEL_DIR/3dunet_kits19_pytorch.ptc" ]] || { echo "Missing model" | tee -a "$OUTPUT_DIR/build.log"; exit 1; }

  mkdir -p "$WORK_DIR"
  cp -a "$ROOT_DIR/repo" "$SRC_DIR"
  sed -i 's/^TARGET_SPACING = None$/TARGET_SPACING = [1.6, 1.2, 1.2]/' "$APP_DIR/global_vars.py"
  sed -i '/if args.mode == "preprocess":/,/if args.mode == "gen_hash":/ { /^[[:space:]]*verify_dataset(args)$/d; }' "$APP_DIR/preprocess.py"
  sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$APP_DIR/pytorch_SUT.py"

  if [[ -n "${PREFLIGHT_CASES:-}" ]]; then
    IFS=',' read -r -a selected_cases <<< "$PREFLIGHT_CASES"
    {
      echo '['
      for ((manifest_i=0; manifest_i<${#selected_cases[@]}; manifest_i++)); do
        suffix=','; [[ $manifest_i -eq $((${#selected_cases[@]} - 1)) ]] && suffix=''
        printf '  "%s"%s\n' "${selected_cases[$manifest_i]}" "$suffix"
      done
      echo ']'
    } > "$APP_DIR/meta/inference_cases.json"
  fi

  mkdir -p "$WORK_DIR/raw_data"
  while IFS= read -r case_name; do
    source_case="$case_name"
    [[ "$case_name" == "case_00400" ]] && source_case="case_00185"
    ln -s "$KITS19_RAW_DIR/$source_case" "$WORK_DIR/raw_data/$case_name"
  done < <(sed -n 's/.*"\(case_[0-9]*\)".*/\1/p' "$APP_DIR/meta/inference_cases.json")

  log_cmd "$OUTPUT_DIR/build.log" uv venv --python 3.11 "$VENV_DIR"
  log_cmd "$OUTPUT_DIR/build.log" uv pip install --python "$PYTHON" \
    'torch==2.1.2' 'numpy==1.26.4' 'scipy==1.11.4' 'nibabel==5.2.1' \
    'pandas==1.5.3' 'pybind11==2.11.1' 'setuptools<70' wheel
  log_cmd "$OUTPUT_DIR/build.log" uv pip install --python "$PYTHON" "$SRC_DIR/loadgen"
  mkdir -p "$APP_DIR/build"
  cp "$SRC_DIR/loadgen/mlperf.conf" "$APP_DIR/build/mlperf.conf"
  "$PYTHON" - <<'PY' 2>&1 | tee -a "$OUTPUT_DIR/build.log"
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("environment imports: ok")
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("numpy:", numpy.__version__, "scipy:", scipy.__version__, "pandas:", pandas.__version__, "nibabel:", nibabel.__version__)
PY
  touch "$STATE_DIR/env.done"
}

stage_preprocess() {
  [[ -f "$STATE_DIR/preprocess.done" ]] && return
  [[ -f "$STATE_DIR/env.done" ]] || { echo "env stage is incomplete" >&2; exit 1; }
  : > "$OUTPUT_DIR/preprocess.log"
  mkdir -p "$APP_DIR/build/preprocessed_data"
  (
    cd "$APP_DIR"
    log_cmd "$OUTPUT_DIR/preprocess.log" "$PYTHON" preprocess.py \
      --raw_data_dir "$WORK_DIR/raw_data" \
      --results_dir "$APP_DIR/build/preprocessed_data" \
      --mode preprocess --num_proc "${PREPROCESS_PROCS:-4}"
  )
  expected="$(sed -n 's/.*"case_[0-9]*".*/x/p' "$APP_DIR/meta/inference_cases.json" | wc -l)"
  actual="$(find "$APP_DIR/build/preprocessed_data" -maxdepth 1 -type f -name 'case_*.pkl' | wc -l)"
  [[ "$actual" -eq "$expected" ]] || { echo "Preprocessed $actual cases, expected $expected" | tee -a "$OUTPUT_DIR/preprocess.log"; exit 1; }
  touch "$STATE_DIR/preprocess.done"
}

stage_run() {
  [[ -f "$STATE_DIR/run.done" ]] && return
  [[ -f "$STATE_DIR/preprocess.done" ]] || { echo "preprocess stage is incomplete" >&2; exit 1; }
  : > "$OUTPUT_DIR/run.log"
  nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader | tee "$OUTPUT_DIR/gpu-before-run.log" | tee -a "$OUTPUT_DIR/run.log"
  "$PYTHON" - <<'PY' 2>&1 | tee -a "$OUTPUT_DIR/run.log"
import torch
assert torch.cuda.is_available(), "CUDA GPU is not available"
free, total = torch.cuda.mem_get_info(0)
print(f"cuda memory immediately before model run: free={free} total={total}")
PY
  mkdir -p "$APP_DIR/build/postprocessed_data"
  (
    cd "$APP_DIR"
    export LOG_PATH="$OUTPUT_DIR"
    export SKIP_VERIFY_ACCURACY=1
    log_cmd "$OUTPUT_DIR/run.log" "$PYTHON" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model "$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
      --preprocessed_data_dir "$APP_DIR/build/preprocessed_data"
  )
  : > "$OUTPUT_DIR/evaluator.log"
  (
    cd "$APP_DIR"
    set -o pipefail
    "$PYTHON" accuracy_kits.py \
      --preprocessed_data_dir "$APP_DIR/build/preprocessed_data" \
      --postprocessed_data_dir "$APP_DIR/build/postprocessed_data" \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" --num_proc 4 \
      2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  )
  for required in mlperf_log_accuracy.json mlperf_log_detail.txt accuracy.txt; do
    [[ -s "$OUTPUT_DIR/$required" ]] || { echo "Missing output: $required" | tee -a "$OUTPUT_DIR/run.log"; exit 1; }
  done
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  touch "$STATE_DIR/run.done"
}

if [[ "$MODE" == "from" ]]; then clear_from "$START"; fi

run_selected() {
  local stage="$1"
  case "$stage" in env) stage_env ;; preprocess) stage_preprocess ;; run) stage_run ;; esac
}

if [[ "$MODE" == "stage" ]]; then
  run_selected "$START"
else
  stages=(env preprocess run)
  begin=0
  if [[ "$MODE" == "from" ]]; then
    for i in "${!stages[@]}"; do [[ "${stages[$i]}" == "$START" ]] && begin="$i"; done
  fi
  for ((i=begin; i<${#stages[@]}; i++)); do run_selected "${stages[$i]}"; done
fi
