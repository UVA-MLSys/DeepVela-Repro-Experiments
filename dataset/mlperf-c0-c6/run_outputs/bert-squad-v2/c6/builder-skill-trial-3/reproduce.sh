#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad [--stage build|preprocess|run] [--from build|preprocess|run]" >&2
  exit 2
}

target=""
stage=""
from_stage=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; target="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; stage="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; from_stage="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$target" == "bert-squad" ]] || usage
[[ -z "$stage" || -z "$from_stage" ]] || usage
case "${stage:-${from_stage:-build}}" in build|preprocess|run) ;; *) usage ;; esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$script_dir/repo"
target_dir="$script_dir/$target"
work_dir="$target_dir/work"
bert_dir="$work_dir/bert"
loadgen_dir="$work_dir/loadgen"
venv_dir="$work_dir/venv"
state_dir="$target_dir/.state"

: "${CACHE_ROOT:=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
: "${SQUAD_DIR:=$CACHE_ROOT/bert-squad/data}"
: "${MODEL_DIR:=$CACHE_ROOT/bert-squad/model}"
: "${DEEPLEARNING_EXAMPLES_DIR:=$CACHE_ROOT/bert-squad/DeepLearningExamples}"
: "${UV_CACHE_DIR:=$CACHE_ROOT/uv}"
: "${OUTPUT_DIR:=$target_dir/out}"
export UV_CACHE_DIR
export UV_EXCLUDE_NEWER=2026-11-10
export HF_HOME="$target_dir/huggingface"

mkdir -p "$OUTPUT_DIR"
output_dir="$(cd "$OUTPUT_DIR" && pwd)"
mkdir -p "$state_dir"

require_file() {
  [[ -f "$1" ]] || { echo "required file missing: $1" >&2; exit 1; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "required directory missing: $1" >&2; exit 1; }
}

require_dir "$repo_dir/language/bert"
require_dir "$repo_dir/loadgen"
require_file "$SQUAD_DIR/dev-v1.1.json"
require_file "$MODEL_DIR/model.pytorch"
require_file "$MODEL_DIR/vocab.txt"
require_dir "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT"

run_logged() {
  local log_file="$1"
  shift
  set +e
  { /usr/bin/time -v "$@"; } 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

clean_from() {
  case "$1" in
    build)
      if [[ -d "$work_dir" ]]; then rm -rf -- "$work_dir"; fi
      rm -f -- "$state_dir/build.success" "$state_dir/preprocess.success" "$state_dir/run.success"
      ;;
    preprocess)
      rm -f -- "$bert_dir/eval_features.pickle"
      rm -f -- "$state_dir/preprocess.success" "$state_dir/run.success"
      ;;
    run)
      rm -f -- "$state_dir/run.success"
      ;;
  esac
}

build_stage() {
  if [[ -f "$state_dir/build.success" ]]; then
    echo "build stage already complete"
    return
  fi
  mkdir -p "$work_dir"
  cp -a "$repo_dir/language/bert" "$bert_dir"
  cp -a "$repo_dir/loadgen" "$loadgen_dir"
  rmdir "$bert_dir/DeepLearningExamples"
  ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$bert_dir/DeepLearningExamples"

  # This pinned source archive omits trace_generator.h, and no translation unit uses it.
  # Remove the stale setuptools dependency entry only in the private build copy.
  sed -i '/"trace_generator.h",/d' "$loadgen_dir/setup.py"

  run_logged "$output_dir/build.log" bash -c '
    set -euo pipefail
    python_base="$(uv python find 3.11)"
    test -f "$("$python_base" -c "import sysconfig; print(sysconfig.get_path(\"include\"))")/Python.h"
    uv venv --python "$python_base" "$1"
    uv pip install --offline --python "$1/bin/python" \
      "setuptools==75.8.0" "wheel==0.45.1" "pybind11==2.11.1" \
      "numpy==1.26.4" "six==1.17.0" "absl-py==2.3.1" \
      "boto3==1.43.121" "torch==2.5.1" "transformers==4.45.2"
    uv pip install --offline --no-build-isolation --python "$1/bin/python" "$2"
    "$1/bin/python" -c "import mlperf_loadgen, numpy, torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"transformers\", transformers.__version__, \"numpy\", numpy.__version__)"
  ' _ "$venv_dir" "$loadgen_dir"
  touch "$state_dir/build.success"
}

preprocess_stage() {
  if [[ -f "$state_dir/preprocess.success" ]]; then
    echo "preprocess stage already complete"
    return
  fi
  [[ -f "$state_dir/build.success" ]] || { echo "build stage has not completed" >&2; exit 1; }
  mkdir -p "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2" "$bert_dir/build/result"
  ln -sf "$SQUAD_DIR/dev-v1.1.json" "$bert_dir/build/data/dev-v1.1.json"
  ln -sf "$MODEL_DIR/model.pytorch" "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
  ln -sf "$MODEL_DIR/vocab.txt" "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
  cp "$repo_dir/loadgen/mlperf.conf" "$bert_dir/build/mlperf.conf"

  run_logged "$output_dir/preprocess.log" bash -c '
    set -euo pipefail
    cd "$1"
    export DATASET_FILE="$2/dev-v1.1.json"
    export VOCAB_FILE="$3/vocab.txt"
    export PYTHONPATH="$1/DeepLearningExamples/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}"
    "$4/bin/python" - <<"PY"
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL(cache_path="eval_features.pickle")
print("preprocessed_features:", len(qsl.eval_features))
PY
  ' _ "$bert_dir" "$SQUAD_DIR" "$MODEL_DIR" "$venv_dir"
  touch "$state_dir/preprocess.success"
}

run_stage() {
  [[ -f "$state_dir/preprocess.success" ]] || { echo "preprocess stage has not completed" >&2; exit 1; }
  local max_examples="${PREFLIGHT_MAX_EXAMPLES:-}"
  local max_args=()
  if [[ -n "$max_examples" ]]; then
    [[ "$max_examples" =~ ^[1-9][0-9]*$ ]] || { echo "PREFLIGHT_MAX_EXAMPLES must be a positive integer" >&2; exit 1; }
    max_args=(--max_examples "$max_examples")
  fi

  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | tee "$output_dir/gpu-before-run.log"
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "could not determine free GPU memory" >&2; exit 1; }
  if (( free_mib < 10000 )); then
    echo "insufficient free GPU memory: ${free_mib} MiB (need at least 10000 MiB)" >&2
    exit 1
  fi

  rm -f -- "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" \
    "$output_dir/mlperf_log_summary.txt" "$output_dir/mlperf_log_trace.json" \
    "$output_dir/accuracy.txt" "$output_dir/run.log" "$output_dir/evaluator.log"

  (
    cd "$bert_dir"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export LOG_PATH="$output_dir"
    export SKIP_VERIFY_ACCURACY=1
    echo "pytorch device: $("$venv_dir/bin/python" -c 'import torch; print(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))')"
    "$venv_dir/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
  ) 2>&1 | tee "$output_dir/run.log"
  local run_status=${PIPESTATUS[0]}
  (( run_status == 0 )) || return "$run_status"

  local evaluator_args=(
    --vocab_file "$MODEL_DIR/vocab.txt"
    --val_data "$SQUAD_DIR/dev-v1.1.json"
    --log_file "$output_dir/mlperf_log_accuracy.json"
    --out_file "$bert_dir/build/result/predictions.json"
    --features_cache_file "$bert_dir/eval_features.pickle"
  )
  if [[ -n "$max_examples" ]]; then evaluator_args+=(--max_examples "$max_examples"); fi
  (
    cd "$bert_dir"
    "$venv_dir/bin/python" accuracy-squad.py "${evaluator_args[@]}"
  ) 2>&1 | tee "$output_dir/accuracy.txt" "$output_dir/evaluator.log"
  local eval_status=${PIPESTATUS[0]}
  (( eval_status == 0 )) || return "$eval_status"

  require_file "$output_dir/mlperf_log_accuracy.json"
  require_file "$output_dir/mlperf_log_detail.txt"
  require_file "$output_dir/accuracy.txt"
  grep -q '^pytorch device: ' "$output_dir/run.log"
  touch "$state_dir/run.success"
}

if [[ -n "$from_stage" ]]; then clean_from "$from_stage"; fi

if [[ -n "$stage" ]]; then
  "${stage}_stage"
  exit 0
fi

start="${from_stage:-build}"
case "$start" in
  build) build_stage; preprocess_stage; run_stage ;;
  preprocess) preprocess_stage; run_stage ;;
  run) run_stage ;;
esac
