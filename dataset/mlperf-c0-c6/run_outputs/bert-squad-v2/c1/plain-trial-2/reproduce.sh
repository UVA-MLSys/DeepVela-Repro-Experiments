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
TARGET=$2
[[ "$TARGET" == "bert-squad" ]] || usage

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CALL_DIR=$PWD
case ${OUTPUT_DIR:-} in
    "") OUTPUT_DIR="$SCRIPT_DIR/$TARGET/out" ;;
    /*) ;;
    *) OUTPUT_DIR="$CALL_DIR/$OUTPUT_DIR" ;;
esac

TARGET_DIR="$SCRIPT_DIR/$TARGET"
WORK_DIR="$TARGET_DIR/work"
VENV_DIR="$TARGET_DIR/.venv"
BERT_DIR="$WORK_DIR/language/bert"
mkdir -p "$TARGET_DIR" "$OUTPUT_DIR" "$WORK_DIR/language"

BUILD_LOG="$TARGET_DIR/build.log"
RUN_LOG="$TARGET_DIR/run.log"
EVAL_LOG="$TARGET_DIR/evaluator.log"
GPU_LOG="$TARGET_DIR/gpu-before-run.log"
: > "$BUILD_LOG"
: > "$RUN_LOG"
: > "$EVAL_LOG"
: > "$GPU_LOG"

{
    echo "Preparing target-local source tree"
    cp -a "$SCRIPT_DIR/repo/language/bert" "$WORK_DIR/language/"
    mkdir -p "$BERT_DIR/DeepLearningExamples"
    if [[ ! -e "$BERT_DIR/DeepLearningExamples/PyTorch" ]]; then
        ln -s "$DEEPLEARNING_EXAMPLES_DIR/PyTorch" \
            "$BERT_DIR/DeepLearningExamples/PyTorch"
    fi

    # Log the actual parameter device after moving the model to its device.
    if ! grep -q 'pytorch device: ' "$BERT_DIR/pytorch_SUT.py"; then
        sed -i '/self\.model\.to(self\.dev)/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
            "$BERT_DIR/pytorch_SUT.py"
    fi

    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        uv venv --python 3.10 "$VENV_DIR"
    fi
    uv pip install --python "$VENV_DIR/bin/python" \
        'numpy==1.26.4' \
        'torch==2.6.0' \
        'transformers==4.46.3' \
        'absl-py==2.3.1' \
        'boto3==1.40.70' \
        'six==1.17.0' \
        'setuptools==75.6.0' \
        'pybind11==2.13.6'
    if ! "$VENV_DIR/bin/python" -c 'import mlperf_loadgen' 2>/dev/null; then
        uv pip install --python "$VENV_DIR/bin/python" "$SCRIPT_DIR/repo/loadgen"
    fi
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen
import numpy
import torch
import transformers
print("python dependencies ready")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
PY
} > >(tee -a "$BUILD_LOG") 2>&1

{
    echo "GPU state immediately before the target run"
    nvidia-smi \
        --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
} | tee -a "$GPU_LOG"

export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export LOG_PATH="$OUTPUT_DIR"
export SKIP_VERIFY_ACCURACY=1
export PYTHONUNBUFFERED=1

cd "$BERT_DIR"
"$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy \
    2>&1 | tee -a "$RUN_LOG"

mkdir -p "$WORK_DIR/result"
"$VENV_DIR/bin/python" accuracy-squad.py \
    --vocab_file "$VOCAB_FILE" \
    --val_data "$DATASET_FILE" \
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --out_file "$WORK_DIR/result/predictions.json" \
    --features_cache_file "$BERT_DIR/eval_features.pickle" \
    2>&1 | tee "$OUTPUT_DIR/accuracy.txt" | tee -a "$EVAL_LOG"

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
grep -q '^pytorch device: ' "$RUN_LOG"
