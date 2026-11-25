#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet [--stage env|prepare|run | --from env|prepare|run]" >&2
  exit 2
}

TARGET=""
MODE="all"
START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; TARGET="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; MODE="stage"; START_STAGE="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; MODE="from"; START_STAGE="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$TARGET" == "3d-unet" ]] || usage
[[ "$MODE" == "all" || "$START_STAGE" =~ ^(env|prepare|run)$ ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
TARGET_DIR="$SCRIPT_DIR/$TARGET"
WORK_DIR="$TARGET_DIR/work"
VENV_DIR="$TARGET_DIR/.venv"
APP_DIR="$WORK_DIR/app"
RAW_LINK_DIR="$WORK_DIR/raw_data"
PREPROCESSED_DIR="$WORK_DIR/preprocessed_data"
POSTPROCESSED_DIR="$WORK_DIR/postprocessed_data"
CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
KITS19_RAW_DIR="${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
UV_EXCLUDE_NEWER="2026-11-10"
export UV_CACHE_DIR UV_EXCLUDE_NEWER

mkdir -p "$TARGET_DIR" "$WORK_DIR"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  mkdir -p "$OUTPUT_DIR"
  OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
else
  OUTPUT_DIR="$TARGET_DIR/out"
  mkdir -p "$OUTPUT_DIR"
fi
export OUTPUT_DIR

ENV_MARKER="$TARGET_DIR/.stage_env.done"
PREP_MARKER="$TARGET_DIR/.stage_prepare.done"
RUN_MARKER="$TARGET_DIR/.stage_run.done"

validate_layout() {
  [[ -d "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" ]] || {
    echo "missing supplied repository at $REPO_DIR" >&2; exit 1;
  }
  [[ -d "$REPO_DIR/loadgen" ]] || { echo "missing LoadGen source" >&2; exit 1; }
  [[ -d "$KITS19_RAW_DIR" ]] || { echo "missing KITS19_RAW_DIR" >&2; exit 1; }
  [[ -f "$MODEL_DIR/3dunet_kits19_pytorch.ptc" ]] || { echo "missing fp32 model" >&2; exit 1; }
  [[ -d "$UV_CACHE_DIR" ]] || { echo "missing UV_CACHE_DIR" >&2; exit 1; }
}

clear_from() {
  case "$1" in
    env)
      rm -rf "$VENV_DIR"
      rm -f "$ENV_MARKER"
      ;&
    prepare)
      rm -rf "$APP_DIR" "$RAW_LINK_DIR" "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR"
      rm -f "$PREP_MARKER"
      ;&
    run)
      rm -f "$RUN_MARKER" \
        "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" \
        "$OUTPUT_DIR/accuracy.txt" \
        "$OUTPUT_DIR/run.log" \
        "$OUTPUT_DIR/evaluator.log" \
        "$OUTPUT_DIR/gpu_memory.log"
      rm -rf "$POSTPROCESSED_DIR"
      ;;
  esac
}

stage_env() {
  [[ -f "$ENV_MARKER" ]] && return
  mkdir -p "$OUTPUT_DIR"
  {
    echo "Python environment build started: $(date --iso-8601=seconds)"
    uv venv --python 3.11 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" \
      'torch==2.4.1' 'numpy==1.26.4' 'scipy==1.10.1' \
      'pandas==1.5.3' 'nibabel==5.2.1' "$REPO_DIR/loadgen"
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("pandas", pandas.__version__)
print("nibabel", nibabel.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
    echo "Python environment build finished: $(date --iso-8601=seconds)"
  } 2>&1 | tee "$OUTPUT_DIR/build.log"
  touch "$ENV_MARKER"
}

stage_prepare() {
  [[ -f "$PREP_MARKER" ]] && return
  [[ -f "$ENV_MARKER" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
  mkdir -p "$APP_DIR" "$RAW_LINK_DIR" "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR"
  cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19/." "$APP_DIR/"
  mkdir -p "$APP_DIR/build"
  cp "$REPO_DIR/loadgen/mlperf.conf" "$APP_DIR/build/mlperf.conf"

  local cases_json="$APP_DIR/meta/inference_cases.json"
  if [[ -n "${REPRO_CASE_LIMIT:-}" ]]; then
    [[ "$REPRO_CASE_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid REPRO_CASE_LIMIT" >&2; exit 1; }
    "$VENV_DIR/bin/python" - "$cases_json" "$APP_DIR/meta/checksum_inference.json" "$REPRO_CASE_LIMIT" <<'PY'
import json, sys
cases_path, checksums_path, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(cases_path) as f: cases = json.load(f)[:limit]
with open(checksums_path) as f: sums = json.load(f)
with open(cases_path, "w") as f: json.dump(cases, f, indent=4)
keep = {f"{case}.pkl": sums[f"{case}.pkl"] for case in cases}
with open(checksums_path, "w") as f: json.dump(keep, f, indent=4, sort_keys=True)
PY
  fi

  "$VENV_DIR/bin/python" - "$cases_json" "$KITS19_RAW_DIR" "$RAW_LINK_DIR" <<'PY'
import json, os, pathlib, sys
cases_path, source, dest = sys.argv[1:]
with open(cases_path) as f: cases = json.load(f)
for case in cases:
    actual = "case_00185" if case == "case_00400" else case
    src = pathlib.Path(source, actual)
    if not src.is_dir(): raise SystemExit(f"missing raw case: {src}")
    os.symlink(src, pathlib.Path(dest, case), target_is_directory=True)
print(f"Prepared {len(cases)} target-local raw case links")
PY

  {
    echo "Preprocessing started: $(date --iso-8601=seconds)"
    cd "$APP_DIR"
    "$VENV_DIR/bin/python" preprocess.py \
      --raw_data_dir "$RAW_LINK_DIR" \
      --results_dir "$PREPROCESSED_DIR" \
      --mode preprocess --num_proc 4
    echo "Preprocessing finished: $(date --iso-8601=seconds)"
  } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  touch "$PREP_MARKER"
}

stage_run() {
  [[ -f "$RUN_MARKER" ]] && return
  [[ -f "$PREP_MARKER" ]] || { echo "prepare stage has not succeeded" >&2; exit 1; }
  mkdir -p "$POSTPROCESSED_DIR" "$OUTPUT_DIR" "$WORK_DIR/cache" "$WORK_DIR/torch"
  export LOG_PATH="$OUTPUT_DIR"
  export SKIP_VERIFY_ACCURACY=1
  export XDG_CACHE_HOME="$WORK_DIR/cache"
  export TORCH_HOME="$WORK_DIR/torch"

  nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free \
    --format=csv,noheader | tee "$OUTPUT_DIR/gpu_memory.log"
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "could not determine free GPU memory" >&2; exit 1; }
  (( free_mib >= 12000 )) || { echo "insufficient free GPU memory: ${free_mib} MiB" >&2; exit 1; }

  {
    echo "GPU inference started: $(date --iso-8601=seconds)"
    echo "pytorch device: cuda:0"
    cd "$APP_DIR"
    "$VENV_DIR/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
      --preprocessed_data_dir="$PREPROCESSED_DIR" \
      --mlperf_conf="$APP_DIR/build/mlperf.conf" \
      --user_conf="$APP_DIR/user.conf"
    echo "GPU inference finished: $(date --iso-8601=seconds)"
  } 2>&1 | tee "$OUTPUT_DIR/run.log"

  {
    echo "Evaluator started: $(date --iso-8601=seconds)"
    cd "$APP_DIR"
    "$VENV_DIR/bin/python" accuracy_kits.py \
      --preprocessed_data_dir="$PREPROCESSED_DIR" \
      --postprocessed_data_dir="$POSTPROCESSED_DIR" \
      --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --num_proc=4
    echo "Evaluator finished: $(date --iso-8601=seconds)"
  } 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"

  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" ]]
  [[ -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  touch "$RUN_MARKER"
}

run_stage() {
  case "$1" in
    env) stage_env ;;
    prepare) stage_prepare ;;
    run) stage_run ;;
    *) usage ;;
  esac
}

validate_layout
if [[ "$MODE" == "from" ]]; then clear_from "$START_STAGE"; fi
if [[ "$MODE" == "stage" ]]; then
  run_stage "$START_STAGE"
elif [[ "$MODE" == "from" ]]; then
  case "$START_STAGE" in
    env) stage_env; stage_prepare; stage_run ;;
    prepare) stage_prepare; stage_run ;;
    run) stage_run ;;
  esac
else
  stage_env
  stage_prepare
  stage_run
fi
