#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

usage() {
  echo "usage: $0 --target bert-squad" >&2
  exit 2
}

[[ $# -eq 2 && "$1" == "--target" ]] || usage
TARGET_ID=$2
[[ "$TARGET_ID" == "bert-squad" ]] || usage

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET_DIR="$ROOT_DIR/$TARGET_ID"
OUTPUT_DIR=${OUTPUT_DIR:-"$TARGET_DIR/out"}
WORK_DIR="$TARGET_DIR/work"
VENV_DIR="$TARGET_DIR/.venv"

mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" \
      "$OUTPUT_DIR/mlperf_log_trace.json" \
      "$OUTPUT_DIR/accuracy.txt" \
      "$OUTPUT_DIR/bootstrap.log" \
      "$OUTPUT_DIR/build.log" \
      "$OUTPUT_DIR/run.log" \
      "$OUTPUT_DIR/evaluator.log" \
      "$OUTPUT_DIR/gpu-memory.log"

{
  test -r "$SQUAD_DIR/dev-v1.1.json"
  test -r "$MODEL_DIR/model.pytorch"
  test -r "$MODEL_DIR/vocab.txt"
  test -d "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT"
  echo "Creating Python 3.11 environment at $VENV_DIR"
  uv venv --clear --python /usr/bin/python3.11 "$VENV_DIR"
  uv pip install --python "$VENV_DIR/bin/python" \
    'torch==2.4.1' \
    'transformers==4.31.0' \
    'numpy==1.26.4' \
    'absl-py==2.3.1' \
    'boto3==1.35.99' \
    'six==1.17.0' \
    'setuptools==80.9.0' \
    'wheel==0.45.1' \
    'pybind11==2.11.1'
  "$VENV_DIR/bin/python" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
PY
} 2>&1 | tee "$OUTPUT_DIR/bootstrap.log"

cp -a "$ROOT_DIR/repo" "$WORK_DIR/repo"
BERT_DIR="$WORK_DIR/repo/language/bert"
rmdir "$BERT_DIR/DeepLearningExamples"
ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"

# Emit the required diagnostic from the model that was actually loaded. This
# modifies only the disposable work copy, never the supplied repository.
sed -i '/self.model.load_state_dict(torch.load(model_file), strict=False)/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
  "$BERT_DIR/pytorch_SUT.py"

{
  uv pip install --python "$VENV_DIR/bin/python" 'mlcommons-loadgen==6.0.17'
  "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen
print("mlperf_loadgen", mlperf_loadgen.__file__)
PY
} 2>&1 | tee "$OUTPUT_DIR/build.log"

export PATH="$VENV_DIR/bin:$PATH"
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export LOG_PATH="$OUTPUT_DIR"
export SKIP_VERIFY_ACCURACY=1
export PYTHONUNBUFFERED=1

{
  echo "GPU memory immediately before run:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 8192 )); then
    echo "Insufficient free GPU memory: ${free_mib:-unknown} MiB; need at least 8192 MiB" >&2
    exit 1
  fi
} 2>&1 | tee "$OUTPUT_DIR/gpu-memory.log"

{
  cd "$BERT_DIR"
  python3 run.py --backend=pytorch --accuracy
} 2>&1 | tee "$OUTPUT_DIR/run.log"

{
  cd "$BERT_DIR"
  mkdir -p build/result
  python3 accuracy-squad.py \
    --vocab_file "$MODEL_DIR/vocab.txt" \
    --val_data "$SQUAD_DIR/dev-v1.1.json" \
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --out_file build/result/predictions.json \
    --features_cache_file eval_features.pickle
} 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
grep -F 'pytorch device: ' "$OUTPUT_DIR/run.log"
