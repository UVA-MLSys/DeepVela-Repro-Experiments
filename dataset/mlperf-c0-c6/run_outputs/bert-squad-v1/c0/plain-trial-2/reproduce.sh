#!/usr/bin/env bash
set -euo pipefail

TARGET=""
if [[ "${1:-}" == "--target" && -n "${2:-}" ]]; then
  TARGET="$2"
fi
if [[ "$TARGET" != "bert-squad" || $# -ne 2 ]]; then
  echo "usage: $0 --target bert-squad" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/repo"
TARGET_DIR="$ROOT/$TARGET"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUT="$TARGET_DIR/out"
elif [[ "$OUTPUT_DIR" == /* ]]; then
  OUT="$OUTPUT_DIR"
else
  OUT="$PWD/$OUTPUT_DIR"
fi
WORK="$TARGET_DIR/work"
BUILD="$TARGET_DIR/build"
VENV="$ROOT/.venv-bert-squad"

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$TARGET_DIR" "$OUT" "$BUILD"

{
  echo "Building reproducible Python environment"
  echo "UV_CACHE_DIR=$UV_CACHE_DIR"
  echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
  if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python /usr/bin/python3.11 "$VENV"
  fi
  uv pip install --python "$VENV/bin/python" --offline \
    'numpy<2' 'torch==2.1.2' 'transformers==4.31.0' \
    absl-py six boto3 'setuptools<81' wheel 'pybind11==2.11.1'

  uv pip install --python "$VENV/bin/python" --offline \
    'mlcommons-loadgen==6.0.17'
  "$VENV/bin/python" -c \
    'import torch, transformers, mlperf_loadgen; print("torch", torch.__version__, "cuda", torch.version.cuda); print("transformers", transformers.__version__)'
} >"$TARGET_DIR/build.log" 2>&1

rm -rf "$WORK"
mkdir -p "$WORK"
cp -a "$REPO/language/bert/." "$WORK/"
rm -rf "$WORK/DeepLearningExamples"
ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$WORK/DeepLearningExamples"

export PATH="$VENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORK/DeepLearningExamples/PyTorch/LanguageModeling/BERT"
export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export LOG_PATH="$OUT"

(
  cd "$WORK"
  python3 - <<'PY'
from squad_QSL import get_squad_QSL
qsl = get_squad_QSL()
print("preprocessed features:", qsl.count)
PY
) >"$TARGET_DIR/preprocess.log" 2>&1

{
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits
} >"$TARGET_DIR/gpu.log" 2>&1

FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if [[ ! "$FREE_MIB" =~ ^[0-9]+$ || "$FREE_MIB" -lt 10000 ]]; then
  echo "Insufficient free GPU memory immediately before run: ${FREE_MIB:-unknown} MiB" \
    | tee -a "$TARGET_DIR/gpu.log" >&2
  exit 1
fi

(
  cd "$WORK"
  python3 -c 'import torch; assert torch.cuda.is_available(); print("pytorch device: cuda:0", flush=True)'
  SKIP_VERIFY_ACCURACY=1 python3 run.py --backend=pytorch --accuracy
) >"$TARGET_DIR/run.log" 2>&1

(
  cd "$WORK"
  python3 accuracy-squad.py \
    --vocab_file "$VOCAB_FILE" \
    --val_data "$DATASET_FILE" \
    --log_file "$OUT/mlperf_log_accuracy.json" \
    --out_file "$WORK/predictions.json" \
    --features_cache_file "$WORK/eval_features.pickle"
) > >(tee "$TARGET_DIR/evaluator.log" "$OUT/accuracy.txt") \
  2>"$TARGET_DIR/evaluator-stderr.log"

test -s "$OUT/mlperf_log_accuracy.json"
test -s "$OUT/mlperf_log_detail.txt"
test -s "$OUT/accuracy.txt"
