#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:?missing target id}"; shift 2 ;;
    --stage) MODE="stage"; START_STAGE="${2:?missing stage}"; shift 2 ;;
    --from) MODE="from"; START_STAGE="${2:?missing stage}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "usage: $0 --target bert-squad [--stage environment|prepare|run|evaluate] [--from STAGE]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="$ROOT/repo"
INPUT_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache/bert-squad
UV_CACHE=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache/uv
OUT=${OUTPUT_DIR:-$ROOT/$TARGET/out}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
WORK="$ROOT/$TARGET/work"
VENV="$WORK/venv"
SOURCE="$WORK/source"
BERT="$SOURCE/language/bert"
STAGES=(environment prepare run evaluate)

stage_index() {
  case "$1" in
    environment) echo 0 ;; prepare) echo 1 ;; run) echo 2 ;; evaluate) echo 3 ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}

if [[ "$MODE" == "from" ]]; then
  first=$(stage_index "$START_STAGE")
  for ((i=first; i<${#STAGES[@]}; i++)); do
    rm -f "$WORK/.${STAGES[$i]}.success"
  done
  if (( first <= 2 )); then
    rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
      "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json" "$OUT/run.log"
  fi
  if (( first <= 3 )); then
    rm -f "$OUT/accuracy.txt" "$OUT/evaluator.log" "$WORK/predictions.json"
  fi
fi

run_environment() {
  mkdir -p "$WORK"
  {
    echo "Creating isolated Python environment"
    rm -rf "$SOURCE"
    cp -a "$REPO" "$SOURCE"
    rm -rf "$BERT/DeepLearningExamples"
    ln -s "$INPUT_ROOT/DeepLearningExamples" "$BERT/DeepLearningExamples"
    UV_CACHE_DIR="$UV_CACHE" UV_EXCLUDE_NEWER=2026-11-10 uv venv --clear --python 3.12 --seed "$VENV"
    UV_CACHE_DIR="$UV_CACHE" UV_EXCLUDE_NEWER=2026-11-10 uv pip install --python "$VENV/bin/python" \
      'torch==2.4.1' 'transformers==4.46.3' 'numpy==1.26.4' \
      'absl-py==2.1.0' 'pybind11==2.13.6' 'setuptools==75.6.0' 'six==1.16.0' boto3
    UV_CACHE_DIR="$UV_CACHE" UV_EXCLUDE_NEWER=2026-11-10 uv pip install --python "$VENV/bin/python" "$SOURCE/loadgen"
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("mlperf_loadgen", mlperf_loadgen.__file__)
PY
  } 2>&1 | tee "$OUT/build.log"
  touch "$WORK/.environment.success"
}

run_prepare() {
  mkdir -p "$WORK"
  (
    cd "$BERT"
    export DATASET_FILE="$INPUT_ROOT/data/dev-v1.1.json"
    export VOCAB_FILE="$INPUT_ROOT/model/vocab.txt"
    export PYTHONPATH="$BERT/DeepLearningExamples/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}"
    "$VENV/bin/python" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL()
print("preprocessed feature count:", qsl.count)
PY
  ) 2>&1 | tee "$OUT/preprocess.log"
  touch "$WORK/.prepare.success"
}

run_inference() {
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
  if (( free_mib < 8192 )); then
    echo "GPU 0 has only ${free_mib} MiB free; at least 8192 MiB is required" | tee "$OUT/run.log" >&2
    exit 1
  fi
  max_args=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  (
    cd "$BERT"
    export DATASET_FILE="$INPUT_ROOT/data/dev-v1.1.json"
    export VOCAB_FILE="$INPUT_ROOT/model/vocab.txt"
    export ML_MODEL_FILE_WITH_PATH="$INPUT_ROOT/model/model.pytorch"
    export LOG_PATH="$OUT"
    export SKIP_VERIFY_ACCURACY=1
    echo "GPU free memory before run: ${free_mib} MiB"
    "$VENV/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("pytorch device: cuda:0 (" + torch.cuda.get_device_name(0) + ")")
PY
    "$VENV/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
  ) 2>&1 | tee "$OUT/run.log"
  test -s "$OUT/mlperf_log_accuracy.json"
  test -s "$OUT/mlperf_log_detail.txt"
  touch "$WORK/.run.success"
}

run_evaluate() {
  max_args=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  (
    cd "$BERT"
    "$VENV/bin/python" accuracy-squad.py \
      --vocab_file "$INPUT_ROOT/model/vocab.txt" \
      --val_data "$INPUT_ROOT/data/dev-v1.1.json" \
      --log_file "$OUT/mlperf_log_accuracy.json" \
      --out_file "$WORK/predictions.json" \
      --features_cache_file "$BERT/eval_features.pickle" \
      "${max_args[@]}"
  ) 2>&1 | tee "$OUT/accuracy.txt" "$OUT/evaluator.log"
  touch "$WORK/.evaluate.success"
}

run_one() {
  local stage=$1 marker="$WORK/.$1.success"
  if [[ -f "$marker" ]]; then
    echo "Skipping completed stage: $stage"
    return
  fi
  case "$stage" in
    environment) run_environment ;; prepare) run_prepare ;;
    run) run_inference ;; evaluate) run_evaluate ;;
  esac
}

if [[ "$MODE" == "stage" ]]; then
  run_one "$START_STAGE"
else
  first=0
  [[ "$MODE" == "from" ]] && first=$(stage_index "$START_STAGE")
  for ((i=first; i<${#STAGES[@]}; i++)); do
    run_one "${STAGES[$i]}"
  done
fi
