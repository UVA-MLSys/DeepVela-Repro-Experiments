#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=""
ONLY_STAGE=""
FROM_STAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET=${2:-}
      shift 2
      ;;
    --stage)
      ONLY_STAGE=${2:-}
      shift 2
      ;;
    --from)
      FROM_STAGE=${2:-}
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "Expected --target 3d-unet" >&2
  exit 2
fi
if [[ -n "$ONLY_STAGE" && -n "$FROM_STAGE" ]]; then
  echo "Use only one of --stage or --from" >&2
  exit 2
fi
for requested_stage in "$ONLY_STAGE" "$FROM_STAGE"; do
  if [[ -n "$requested_stage" && ! "$requested_stage" =~ ^(setup|preprocess|run)$ ]]; then
    echo "Unknown stage: $requested_stage" >&2
    exit 2
  fi
done

CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
UV_EXCLUDE_NEWER=2026-11-10
KITS19_RAW_DIR=${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}
MODEL_DIR=${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}
WORK_DIR=${REPRO_WORK_DIR:-$SCRIPT_DIR/3d-unet}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK_DIR/out}
SOURCE_DIR=$WORK_DIR/source
VENV_DIR=$WORK_DIR/venv
RAW_VIEW=$WORK_DIR/raw_data
PREPROCESSED_DIR=$WORK_DIR/preprocessed_data
POSTPROCESSED_DIR=$WORK_DIR/postprocessed_data
MODEL_PATH=$MODEL_DIR/3dunet_kits19_pytorch.ptc
PREFLIGHT_CASES=${PREFLIGHT_CASES:-}

export UV_CACHE_DIR UV_EXCLUDE_NEWER

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
WORK_DIR=$(cd "$WORK_DIR" && pwd)
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

case "$WORK_DIR" in
  "$SCRIPT_DIR"/*) ;;
  *) echo "REPRO_WORK_DIR must be inside $SCRIPT_DIR" >&2; exit 2 ;;
esac
case "$OUTPUT_DIR" in
  "$WORK_DIR"/*) ;;
  *) echo "OUTPUT_DIR must be inside $WORK_DIR" >&2; exit 2 ;;
esac

require_inputs() {
  [[ -d "$SCRIPT_DIR/repo" ]] || { echo "Missing repository: $SCRIPT_DIR/repo" >&2; exit 1; }
  [[ -d "$KITS19_RAW_DIR" ]] || { echo "Missing raw data: $KITS19_RAW_DIR" >&2; exit 1; }
  [[ -f "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
  [[ -d "$UV_CACHE_DIR" ]] || { echo "Missing uv cache: $UV_CACHE_DIR" >&2; exit 1; }
}

clear_from() {
  local stage=$1
  if [[ "$stage" == "setup" ]]; then
    rm -rf "$SOURCE_DIR" "$VENV_DIR" "$RAW_VIEW" "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR"
    rm -f "$WORK_DIR/.setup.done" "$WORK_DIR/.preprocess.done" "$WORK_DIR/.run.done"
  elif [[ "$stage" == "preprocess" ]]; then
    rm -rf "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR"
    rm -f "$WORK_DIR/.preprocess.done" "$WORK_DIR/.run.done"
  else
    rm -rf "$POSTPROCESSED_DIR"
    rm -f "$WORK_DIR/.run.done"
  fi
  rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
    "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" \
    "$OUTPUT_DIR/accuracy.txt"
}

stage_setup() {
  if [[ -f "$WORK_DIR/.setup.done" ]]; then
    echo "setup stage already complete"
    return
  fi
  require_inputs
  {
    date -Is
    echo "source snapshot: 3fbc329939999c13d0a7b5e67fb2092287e06047 (task-provided, Git metadata absent)"
    echo "uv cache: $UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv --version
    uv venv --python 3.10 "$VENV_DIR"
    # The cached CUDA wheels lack upload-date metadata. Installing their exact,
    # Torch-pinned versions without dependency resolution keeps the global cutoff
    # active while avoiding an indeterminate candidate search.
    uv pip install --offline --no-deps --python "$VENV_DIR/bin/python" \
      "nvidia-cuda-nvrtc-cu12==12.1.105" "nvidia-cuda-runtime-cu12==12.1.105" \
      "nvidia-cuda-cupti-cu12==12.1.105" "nvidia-cudnn-cu12==8.9.2.26" \
      "nvidia-cublas-cu12==12.1.3.1" "nvidia-cufft-cu12==11.0.2.54" \
      "nvidia-curand-cu12==10.3.2.106" "nvidia-cusolver-cu12==11.4.5.107" \
      "nvidia-cusparse-cu12==12.1.0.106" "nvidia-nccl-cu12==2.19.3" \
      "nvidia-nvtx-cu12==12.1.105" "nvidia-nvjitlink-cu12==12.1.105"
    uv pip install --offline --no-deps --python "$VENV_DIR/bin/python" \
      --index-url https://download.pytorch.org/whl/cu121 \
      "torch==2.2.0+cu121" "triton==2.2.0"
    uv pip install --offline --python "$VENV_DIR/bin/python" \
      "numpy==1.23.5" "scipy==1.9.3" "nibabel==3.2.2" "pandas==1.5.3" \
      "mlcommons-loadgen==6.0.17" filelock typing-extensions sympy networkx jinja2 fsspec
    "$VENV_DIR/bin/python" -c 'import mlperf_loadgen, numpy, scipy, nibabel, pandas, torch; print("torch", torch.__version__, "cuda", torch.version.cuda); print("numpy", numpy.__version__, "scipy", scipy.__version__, "nibabel", nibabel.__version__, "pandas", pandas.__version__)'
  } 2>&1 | tee "$WORK_DIR/build.log"

  cp -a "$SCRIPT_DIR/repo" "$SOURCE_DIR"
  mkdir -p "$RAW_VIEW"
  for case_path in "$KITS19_RAW_DIR"/case_*; do
    if [[ -d "$case_path" ]]; then
      ln -s "$case_path" "$RAW_VIEW/$(basename "$case_path")"
    fi
  done
  if [[ ! -e "$RAW_VIEW/case_00400" ]]; then
    ln -s "$KITS19_RAW_DIR/case_00185" "$RAW_VIEW/case_00400"
  fi

  if [[ -n "$PREFLIGHT_CASES" ]]; then
    PREFLIGHT_CASES="$PREFLIGHT_CASES" SOURCE_DIR="$SOURCE_DIR" "$VENV_DIR/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["SOURCE_DIR"]) / "vision/medical_imaging/3d-unet-kits19/meta"
requested = [x for x in os.environ["PREFLIGHT_CASES"].split(",") if x]
with (root / "inference_cases.json").open() as handle:
    original = json.load(handle)
if not requested or any(case not in original for case in requested):
    raise SystemExit(f"Invalid PREFLIGHT_CASES: {requested}")
with (root / "inference_cases.json").open("w") as handle:
    json.dump(requested, handle, indent=4)
with (root / "checksum_inference.json").open() as handle:
    checksums = json.load(handle)
filtered = {f"{case}.pkl": checksums[f"{case}.pkl"] for case in requested}
with (root / "checksum_inference.json").open("w") as handle:
    json.dump(filtered, handle, indent=4, sort_keys=True)
print(f"bounded preflight cases: {requested}")
PY
  fi
  touch "$WORK_DIR/.setup.done"
}

stage_preprocess() {
  if [[ -f "$WORK_DIR/.preprocess.done" ]]; then
    echo "preprocess stage already complete"
    return
  fi
  [[ -f "$WORK_DIR/.setup.done" ]] || { echo "setup stage has not completed" >&2; exit 1; }
  mkdir -p "$PREPROCESSED_DIR"
  (
    cd "$SOURCE_DIR/vision/medical_imaging/3d-unet-kits19"
    time "$VENV_DIR/bin/python" preprocess.py \
      --raw_data_dir "$RAW_VIEW" \
      --results_dir "$PREPROCESSED_DIR" \
      --mode preprocess --num_proc 4
  ) 2>&1 | tee "$WORK_DIR/preprocess.log"
  touch "$WORK_DIR/.preprocess.done"
}

stage_run() {
  if [[ -f "$WORK_DIR/.run.done" ]]; then
    echo "run stage already complete"
    return
  fi
  [[ -f "$WORK_DIR/.preprocess.done" ]] || { echo "preprocess stage has not completed" >&2; exit 1; }
  mkdir -p "$POSTPROCESSED_DIR" "$OUTPUT_DIR"
  {
    date -Is
    echo "GPU memory immediately before run (MiB):"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
    echo "pytorch device: cuda:0"
  } | tee "$WORK_DIR/gpu-memory.log" "$WORK_DIR/run.log"

  (
    cd "$SOURCE_DIR/vision/medical_imaging/3d-unet-kits19"
    export PATH="$VENV_DIR/bin:$PATH"
    export LOG_PATH="$OUTPUT_DIR"
    export SKIP_VERIFY_ACCURACY=1
    time python3 run.py --backend=pytorch --scenario=Offline --accuracy \
      --model "$MODEL_PATH" \
      --preprocessed_data_dir "$PREPROCESSED_DIR" \
      --mlperf_conf "$SOURCE_DIR/loadgen/mlperf.conf" \
      --user_conf user.conf
  ) 2>&1 | tee -a "$WORK_DIR/run.log"

  (
    cd "$SOURCE_DIR/vision/medical_imaging/3d-unet-kits19"
    { time "$VENV_DIR/bin/python" accuracy_kits.py \
        --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        --preprocessed_data_dir "$PREPROCESSED_DIR" \
        --postprocessed_data_dir "$POSTPROCESSED_DIR" \
        --num_proc 4; \
    } 2> >(tee "$WORK_DIR/evaluator.stderr.log" >&2)
  ) | tee "$WORK_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"

  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" ]]
  [[ -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  grep -q '^pytorch device: ' "$WORK_DIR/run.log"
  touch "$WORK_DIR/.run.done"
}

run_stage() {
  case "$1" in
    setup) stage_setup ;;
    preprocess) stage_preprocess ;;
    run) stage_run ;;
  esac
}

if [[ -n "$FROM_STAGE" ]]; then
  clear_from "$FROM_STAGE"
fi

if [[ -n "$ONLY_STAGE" ]]; then
  run_stage "$ONLY_STAGE"
elif [[ -n "$FROM_STAGE" ]]; then
  start=0
  for stage in setup preprocess run; do
    [[ "$stage" == "$FROM_STAGE" ]] && start=1
    [[ $start -eq 1 ]] && run_stage "$stage"
  done
else
  for stage in setup preprocess run; do
    run_stage "$stage"
  done
fi
