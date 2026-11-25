#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:?missing target id}"; shift 2 ;;
    --stage) STAGE="${2:?missing stage name}"; shift 2 ;;
    --from) FROM="${2:?missing stage name}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "supported target: bert-squad" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$ROOT_DIR/repo"
CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER="2026-11-10"

TARGET_DIR="$ROOT_DIR/$TARGET"
OUTPUT_DIR="${OUTPUT_DIR:-$TARGET_DIR/out}"
WORK_DIR="$TARGET_DIR/work"
VENV_DIR="$TARGET_DIR/venv"
BERT_DIR="$WORK_DIR/bert"
MARKER_DIR="$TARGET_DIR/markers"
mkdir -p "$OUTPUT_DIR" "$WORK_DIR" "$MARKER_DIR"

stages=(setup preprocess run)
valid_stage() {
  local requested="$1" candidate
  for candidate in "${stages[@]}"; do
    [[ "$candidate" == "$requested" ]] && return 0
  done
  return 1
}

if [[ -n "$STAGE" && -n "$FROM" ]]; then
  echo "use only one of --stage and --from" >&2
  exit 2
fi
if [[ -n "$STAGE" ]] && ! valid_stage "$STAGE"; then
  echo "unknown stage: $STAGE" >&2
  exit 2
fi
if [[ -n "$FROM" ]] && ! valid_stage "$FROM"; then
  echo "unknown stage: $FROM" >&2
  exit 2
fi

if [[ -n "$FROM" ]]; then
  clearing=0
  for candidate in "${stages[@]}"; do
    [[ "$candidate" == "$FROM" ]] && clearing=1
    if [[ $clearing -eq 1 ]]; then
      rm -f "$MARKER_DIR/$candidate.ok"
    fi
  done
  if [[ "$FROM" == "setup" ]]; then
    rm -rf "$VENV_DIR" "$WORK_DIR"
    mkdir -p "$WORK_DIR"
    rm -f "$OUTPUT_DIR"/*.log "$OUTPUT_DIR"/mlperf_log_* "$OUTPUT_DIR"/accuracy.txt
  elif [[ "$FROM" == "preprocess" ]]; then
    rm -f "$BERT_DIR/eval_features.pickle"
    rm -f "$OUTPUT_DIR/preprocess.log" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log"
    rm -f "$OUTPUT_DIR"/mlperf_log_* "$OUTPUT_DIR/accuracy.txt"
  else
    rm -f "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log"
    rm -f "$OUTPUT_DIR"/mlperf_log_* "$OUTPUT_DIR/accuracy.txt"
    rm -rf "$BERT_DIR/build/result"
  fi
fi

run_setup() {
  local log="$OUTPUT_DIR/build.log"
  {
    date -Is
    echo "Creating Python environment and building repository LoadGen"
    test -f "$REPO_DIR/language/bert/run.py"
    test -f "$SQUAD_DIR/dev-v1.1.json"
    test -f "$MODEL_DIR/model.pytorch"
    test -f "$MODEL_DIR/vocab.txt"
    test -f "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"
    uv venv --python 3.11 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" --torch-backend=cu124 \
      'torch==2.5.1' 'transformers==4.44.2' 'numpy==1.26.4' \
      'absl-py==2.1.0' 'six==1.16.0' 'boto3==1.35.54' \
      'setuptools==70.3.0' 'packaging==24.1'
    rm -rf "$BERT_DIR" "$WORK_DIR/loadgen"
    cp -a "$REPO_DIR/language/bert" "$BERT_DIR"
    cp -a "$REPO_DIR/loadgen" "$WORK_DIR/loadgen"
    mkdir -p "$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling"
    cp -a "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT" \
      "$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling/BERT"
    uv pip install --python "$VENV_DIR/bin/python" "$WORK_DIR/loadgen"
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda available", torch.cuda.is_available())
PY
    date -Is
  } 2>&1 | tee "$log"
  touch "$MARKER_DIR/setup.ok"
}

run_preprocess() {
  local log="$OUTPUT_DIR/preprocess.log"
  {
    date -Is
    cd "$BERT_DIR"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export PYTHONPATH="$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}"
    "$VENV_DIR/bin/python" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL()
print("preprocessed feature count:", len(qsl.eval_features))
del qsl
PY
    test -s eval_features.pickle
    date -Is
  } 2>&1 | tee "$log"
  touch "$MARKER_DIR/preprocess.ok"
}

run_benchmark() {
  local run_log="$OUTPUT_DIR/run.log"
  local eval_log="$OUTPUT_DIR/evaluator.log"
  local max_examples="${RUN_MAX_EXAMPLES:-}"
  local max_args=()
  if [[ -n "$max_examples" ]]; then
    [[ "$max_examples" =~ ^[1-9][0-9]*$ ]] || { echo "RUN_MAX_EXAMPLES must be positive" >&2; exit 2; }
    max_args=(--max_examples "$max_examples")
  fi
  {
    date -Is
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
      --format=csv,noheader,nounits
    cd "$BERT_DIR"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export LOG_PATH="$OUTPUT_DIR"
    export SKIP_VERIFY_ACCURACY=1
    "$VENV_DIR/bin/python" - <<'PY'
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("pytorch device:", device)
if device.type != "cuda":
    raise SystemExit("CUDA is required for this target")
PY
    "$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
    date -Is
  } 2>&1 | tee "$run_log"

  {
    date -Is
    cd "$BERT_DIR"
    mkdir -p build/result
    "$VENV_DIR/bin/python" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --out_file build/result/predictions.json \
      --features_cache_file eval_features.pickle \
      "${max_args[@]}"
    date -Is
  } 2>&1 | tee "$eval_log" "$OUTPUT_DIR/accuracy.txt"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  test -s "$OUTPUT_DIR/accuracy.txt"
  touch "$MARKER_DIR/run.ok"
}

run_one() {
  local name="$1"
  [[ -f "$MARKER_DIR/$name.ok" ]] && return 0
  case "$name" in
    setup) run_setup ;;
    preprocess) run_preprocess ;;
    run) run_benchmark ;;
  esac
}

if [[ -n "$STAGE" ]]; then
  run_one "$STAGE"
else
  start=1
  [[ -n "$FROM" ]] && start=0
  for candidate in "${stages[@]}"; do
    [[ "$candidate" == "$FROM" ]] && start=1
    [[ $start -eq 1 ]] && run_one "$candidate"
  done
fi
