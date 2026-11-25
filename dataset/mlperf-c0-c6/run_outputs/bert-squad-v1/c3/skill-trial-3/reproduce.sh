#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
MAX_EXAMPLES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --max-examples) MAX_EXAMPLES="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "target must be bert-squad" >&2
  exit 2
fi
if [[ -n "$STAGE" && -n "$FROM" ]]; then
  echo "use only one of --stage and --from" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/repo"
TARGET_DIR="$ROOT/$TARGET"
OUT="${OUTPUT_DIR:-$TARGET_DIR/out}"
WORK="$TARGET_DIR/work"
VENV="$WORK/venv"
BERT="$WORK/bert"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DLE_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$OUT" "$WORK"

clear_from() {
  case "$1" in
    build)
      rm -f "$WORK/build.done" "$WORK/preprocess.done" "$WORK/run.done" "$WORK/run.preflight.done"
      rm -f "$BERT/eval_features.pickle" "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/accuracy.txt"
      ;;
    preprocess)
      rm -f "$WORK/preprocess.done" "$WORK/run.done" "$WORK/run.preflight.done" "$BERT/eval_features.pickle"
      rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/accuracy.txt"
      ;;
    run)
      rm -f "$WORK/run.done" "$WORK/run.preflight.done"
      rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/accuracy.txt"
      ;;
    *) echo "stage must be build, preprocess, or run" >&2; exit 2 ;;
  esac
}

if [[ -n "$FROM" ]]; then
  clear_from "$FROM"
fi

run_build() {
  [[ ! -e "$WORK/build.done" ]] || return 0
  {
    echo "stage: build"
    echo "python: /home/mi3se/.local/bin/python3.11"
    echo "UV_CACHE_DIR: $UV_CACHE_DIR"
    uv venv --clear --python /home/mi3se/.local/bin/python3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'torch==2.5.1' 'transformers==4.46.3' 'numpy<2' \
      'absl-py' 'six' 'boto3' 'pybind11==2.11.1' 'setuptools==80.9.0' 'wheel'
    mkdir -p "$BERT" "$WORK/loadgen" "$WORK/lon"
    cp -a "$REPO/language/bert/." "$BERT/"
    cp -a "$REPO/loadgen/." "$WORK/loadgen/"
    cp -a "$REPO/lon/." "$WORK/lon/"
    ln -sfn "$DLE_DIR" "$BERT/DeepLearningExamples"
    sed -i '/self.version = transformers.__version__/a\        print("pytorch device: {}".format(self.dev))' "$BERT/pytorch_SUT.py"
    uv pip install --python "$VENV/bin/python" "$WORK/loadgen"
    "$VENV/bin/python" -c 'import torch, transformers, mlperf_loadgen; print("torch:", torch.__version__); print("transformers:", transformers.__version__); print("cuda available:", torch.cuda.is_available())'
  } 2>&1 | tee "$OUT/build.log"
  touch "$WORK/build.done"
}

run_preprocess() {
  [[ ! -e "$WORK/preprocess.done" ]] || return 0
  [[ -e "$WORK/build.done" ]] || { echo "build stage has not succeeded" >&2; exit 1; }
  {
    echo "stage: preprocess"
    cd "$BERT"
    PYTHONPATH="$DLE_DIR/PyTorch/LanguageModeling/BERT" \
    VOCAB_FILE="$MODEL_DIR/vocab.txt" DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
      "$VENV/bin/python" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL(cache_path="eval_features.pickle")
print("feature_count:", qsl.count)
PY
  } 2>&1 | tee "$OUT/preprocess.log"
  touch "$WORK/preprocess.done"
}

run_inference() {
  local marker="$WORK/run.done"
  [[ -z "$MAX_EXAMPLES" ]] || marker="$WORK/run.preflight.done"
  [[ ! -e "$marker" ]] || return 0
  [[ -e "$WORK/preprocess.done" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  mkdir -p "$WORK/results"
  nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.free,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tee "$OUT/gpu-before-run.log"
  local max_args=()
  if [[ -n "$MAX_EXAMPLES" ]]; then
    max_args=(--max_examples "$MAX_EXAMPLES")
  fi
  {
    echo "stage: run"
    cd "$BERT"
    PYTHONPATH="$DLE_DIR/PyTorch/LanguageModeling/BERT" \
    LOG_PATH="$OUT" \
    VOCAB_FILE="$MODEL_DIR/vocab.txt" \
    DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
    ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch" \
    SKIP_VERIFY_ACCURACY=1 \
      "$VENV/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
  } 2>&1 | tee "$OUT/run.log"
  {
    cd "$BERT"
    PYTHONPATH="$DLE_DIR/PyTorch/LanguageModeling/BERT" "$VENV/bin/python" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUT/mlperf_log_accuracy.json" \
      --out_file "$WORK/results/predictions.json" \
      --features_cache_file "$BERT/eval_features.pickle" \
      "${max_args[@]}"
  } 2> >(tee "$OUT/evaluator-stderr.log" >&2) | tee "$OUT/accuracy.txt" "$OUT/evaluator.log"
  touch "$marker"
}

run_one() {
  case "$1" in
    build) run_build ;;
    preprocess) run_preprocess ;;
    run) run_inference ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}

if [[ -n "$STAGE" ]]; then
  run_one "$STAGE"
elif [[ -n "$FROM" ]]; then
  case "$FROM" in
    build) run_build; run_preprocess; run_inference ;;
    preprocess) run_preprocess; run_inference ;;
    run) run_inference ;;
  esac
else
  run_build
  run_preprocess
  run_inference
fi
