#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad [--stage setup|preflight|run] [--from setup|preflight|run]" >&2
  exit 2
}

TARGET=""
STAGE=""
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; TARGET="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; STAGE="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; FROM="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$TARGET" == "bert-squad" ]] || usage
[[ -z "$STAGE" || -z "$FROM" ]] || usage
case "${STAGE:-${FROM:-setup}}" in setup|preflight|run) ;; *) usage ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/repo"
TARGET_ROOT="$SCRIPT_DIR/$TARGET"
WORK_DIR="$TARGET_ROOT/work"
STAGED_REPO="$WORK_DIR/repo"
BERT_DIR="$STAGED_REPO/language/bert"
VENV_DIR="$WORK_DIR/venv"
PYTHON_INSTALL_DIR="$WORK_DIR/uv-python"
MARKER_DIR="$TARGET_ROOT/.markers"

CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
SQUAD_DIR="${SQUAD_DIR:-$CACHE_ROOT/bert-squad/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/bert-squad/model}"
DEEPLEARNING_EXAMPLES_DIR="${DEEPLEARNING_EXAMPLES_DIR:-$CACHE_ROOT/bert-squad/DeepLearningExamples}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_EXCLUDE_NEWER="2026-11-10"
export UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$TARGET_ROOT/out"
fi
mkdir -p "$OUTPUT_DIR" "$TARGET_ROOT" "$MARKER_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

require_file() {
  [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 1; }
}

validate_inputs() {
  [[ -d "$REPO_ROOT/language/bert" && -d "$REPO_ROOT/loadgen" ]] || {
    echo "expected fresh source tree at $REPO_ROOT" >&2
    exit 1
  }
  require_file "$SQUAD_DIR/dev-v1.1.json"
  require_file "$MODEL_DIR/model.pytorch"
  require_file "$MODEL_DIR/vocab.txt"
  require_file "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"
}

clear_stage_and_later() {
  local first="$1"
  case "$first" in
    setup)
      # WORK_DIR is the fixed target-local path above, never user supplied.
      rm -rf -- "$WORK_DIR"
      rm -f -- "$MARKER_DIR/setup.done" "$MARKER_DIR/preflight.done" "$MARKER_DIR/run.done"
      rm -f -- "$OUTPUT_DIR/build.log" "$OUTPUT_DIR/preflight.log" \
        "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/gpu-memory.log" \
        "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" \
        "$OUTPUT_DIR/mlperf_log_trace.json"
      ;;
    preflight)
      rm -f -- "$MARKER_DIR/preflight.done" "$MARKER_DIR/run.done" \
        "$OUTPUT_DIR/preflight.log"
      ;;
    run)
      rm -f -- "$MARKER_DIR/run.done" "$OUTPUT_DIR/run.log" \
        "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/gpu-memory.log" "$OUTPUT_DIR/accuracy.txt" \
        "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json"
      ;;
  esac
}

stage_setup() {
  if [[ -f "$MARKER_DIR/setup.done" && -f "$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling/BERT/tokenization.py" ]]; then
    return 0
  fi
  if [[ -f "$MARKER_DIR/setup.done" && -L "$BERT_DIR/DeepLearningExamples/DeepLearningExamples" ]]; then
    rm -f -- "$BERT_DIR/DeepLearningExamples/DeepLearningExamples"
    rmdir "$BERT_DIR/DeepLearningExamples"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"
    require_file "$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling/BERT/tokenization.py"
    return 0
  fi
  validate_inputs
  mkdir -p "$WORK_DIR"
  {
    echo "stage: setup"
    echo "source: stated commit 3fbc329939999c13d0a7b5e67fb2092287e06047 (tree has no Git metadata)"
    echo "DeepLearningExamples: stated commit 729963dd47e7c8bd462ad10bfac7a7b0b604e6dd"
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    mkdir -p "$STAGED_REPO"
    cp -a "$REPO_ROOT/." "$STAGED_REPO/"
    rmdir "$BERT_DIR/DeepLearningExamples"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"
    require_file "$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling/BERT/tokenization.py"
    uv venv --python 3.12 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" \
      "torch==2.5.1" \
      "transformers==4.46.3" \
      "numpy==1.26.4" \
      "absl-py==2.1.0" \
      "boto3==1.40.61" \
      "six==1.16.0" \
      "setuptools==75.8.0" \
      "pybind11==2.11.1" \
      "wheel==0.45.1"
    uv pip install --python "$VENV_DIR/bin/python" --no-build-isolation "$STAGED_REPO/loadgen"
    "$VENV_DIR/bin/python" -m pip --version 2>/dev/null || true
    uv pip freeze --python "$VENV_DIR/bin/python"
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("python packages import successfully")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__, "numpy", numpy.__version__)
PY
  } 2>&1 | tee "$OUTPUT_DIR/build.log"
  touch "$MARKER_DIR/setup.done"
}

gpu_check() {
  local log_file="$1"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free \
    --format=csv,noheader | tee "$log_file"
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ ! "$free_mib" =~ ^[0-9]+$ || "$free_mib" -lt 4096 ]]; then
    echo "insufficient free GPU memory: ${free_mib:-unknown} MiB; need at least 4096 MiB" | tee -a "$log_file" >&2
    exit 1
  fi
}

run_benchmark() {
  local max_examples="$1"
  local run_log="$2"
  local evaluator_log="$3"
  local accuracy_out="$4"
  local gpu_log="$5"
  local log_dir="$6"
  local result_dir="$7"

  mkdir -p "$log_dir" "$result_dir"
  gpu_check "$gpu_log"
  (
    cd "$BERT_DIR"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export LOG_PATH="$log_dir"
    export SKIP_VERIFY_ACCURACY=1
    args=(--backend=pytorch --accuracy)
    if [[ -n "$max_examples" ]]; then
      args+=(--max_examples "$max_examples")
    fi
    "$VENV_DIR/bin/python" run.py "${args[@]}"
    "$VENV_DIR/bin/python" - <<'PY'
import torch
print("pytorch device: " + ("cuda:0" if torch.cuda.is_available() else "cpu"))
PY
  ) 2>&1 | tee "$run_log"

  (
    cd "$BERT_DIR"
    args=(
      --vocab_file "$MODEL_DIR/vocab.txt"
      --val_data "$SQUAD_DIR/dev-v1.1.json"
      --log_file "$log_dir/mlperf_log_accuracy.json"
      --out_file "$result_dir/predictions.json"
      --features_cache_file "$BERT_DIR/eval_features.pickle"
    )
    if [[ -n "$max_examples" ]]; then
      args+=(--max_examples "$max_examples")
    fi
    "$VENV_DIR/bin/python" accuracy-squad.py "${args[@]}"
  ) 2>&1 | tee "$accuracy_out" "$evaluator_log"
}

stage_preflight() {
  [[ -f "$MARKER_DIR/preflight.done" ]] && return 0
  if [[ -f "$OUTPUT_DIR/preflight.log" && -f "$OUTPUT_DIR/preflight-accuracy.txt" ]] && \
     grep -q '^pytorch device: ' "$OUTPUT_DIR/preflight.log" && \
     grep -q '"exact_match"' "$OUTPUT_DIR/preflight-accuracy.txt"; then
    touch "$MARKER_DIR/preflight.done"
    return 0
  fi
  stage_setup
  local pf_root="$WORK_DIR/preflight"
  mkdir -p "$pf_root"
  local start end
  start="$(date +%s)"
  run_benchmark "${PREFLIGHT_EXAMPLES:-8}" \
    "$OUTPUT_DIR/preflight.log" "$OUTPUT_DIR/preflight-evaluator.log" \
    "$OUTPUT_DIR/preflight-accuracy.txt" "$OUTPUT_DIR/preflight-gpu-memory.log" \
    "$pf_root/logs" "$pf_root/result"
  end="$(date +%s)"
  echo "preflight_elapsed_seconds=$((end-start))" | tee -a "$OUTPUT_DIR/preflight.log"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/preflight.log"
  grep -q '"exact_match"' "$OUTPUT_DIR/preflight-accuracy.txt"
  touch "$MARKER_DIR/preflight.done"
}

stage_run() {
  [[ -f "$MARKER_DIR/run.done" ]] && return 0
  stage_setup
  run_benchmark "" \
    "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt" \
    "$OUTPUT_DIR/gpu-memory.log" "$OUTPUT_DIR" "$WORK_DIR/result"
  require_file "$OUTPUT_DIR/mlperf_log_accuracy.json"
  require_file "$OUTPUT_DIR/mlperf_log_detail.txt"
  require_file "$OUTPUT_DIR/accuracy.txt"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  grep -q '"exact_match"' "$OUTPUT_DIR/accuracy.txt"
  touch "$MARKER_DIR/run.done"
}

if [[ -n "$FROM" ]]; then
  clear_stage_and_later "$FROM"
  case "$FROM" in
    setup) stage_setup; stage_preflight; stage_run ;;
    preflight) stage_preflight; stage_run ;;
    run) stage_run ;;
  esac
elif [[ -n "$STAGE" ]]; then
  case "$STAGE" in
    setup) stage_setup ;;
    preflight) stage_preflight ;;
    run) stage_run ;;
  esac
else
  stage_setup
  stage_run
fi
