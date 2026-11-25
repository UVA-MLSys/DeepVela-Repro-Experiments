#!/usr/bin/env bash
set -euo pipefail

TARGET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ $# -ge 2 ]] || { echo "--target requires a value" >&2; exit 2; }
            TARGET="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
    echo "unsupported target: ${TARGET:-<missing>}" >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$ROOT_DIR/repo"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

[[ -d "$REPO_DIR/language/bert" ]] || { echo "missing repository at $REPO_DIR" >&2; exit 1; }
[[ -r "$SQUAD_DIR/dev-v1.1.json" ]] || { echo "missing SQuAD input" >&2; exit 1; }
[[ -r "$MODEL_DIR/model.pytorch" && -r "$MODEL_DIR/vocab.txt" ]] || { echo "missing model inputs" >&2; exit 1; }
[[ -r "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py" ]] || { echo "missing DeepLearningExamples input" >&2; exit 1; }

TARGET_DIR="$ROOT_DIR/$TARGET"
mkdir -p "$TARGET_DIR"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR="$TARGET_DIR/out"
elif [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

BUILD_LOG="$TARGET_DIR/build.log"
RUN_LOG="$TARGET_DIR/run.log"
EVAL_LOG="$TARGET_DIR/evaluator.log"
: >"$BUILD_LOG"
: >"$RUN_LOG"
: >"$EVAL_LOG"

WORK_DIR="$TARGET_DIR/work"
VENV_DIR="$TARGET_DIR/venv"
LOADGEN_SRC="$TARGET_DIR/loadgen-src"
rm -rf "$WORK_DIR" "$LOADGEN_SRC"
mkdir -p "$WORK_DIR" "$LOADGEN_SRC"
cp -a "$REPO_DIR/language/bert/." "$WORK_DIR/"
rm -rf "$WORK_DIR/DeepLearningExamples"
ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$WORK_DIR/DeepLearningExamples"
cp -a "$REPO_DIR/loadgen/." "$LOADGEN_SRC/"

patch -d "$WORK_DIR" -p0 >>"$BUILD_LOG" 2>&1 <<'PATCH'
--- pytorch_SUT.py
+++ pytorch_SUT.py
@@ -54,6 +54,7 @@
             torch.device("cuda:0") if torch.cuda.is_available(
             ) else torch.device("cpu")
         )
+        print("pytorch device: {}".format(self.dev))
         self.version = transformers.__version__
 
         print("Loading PyTorch model...")
PATCH

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    uv venv --python python3.11 "$VENV_DIR" >>"$BUILD_LOG" 2>&1
fi

uv pip install --python "$VENV_DIR/bin/python" \
    'torch==2.5.1' 'transformers==4.44.2' 'numpy==1.26.4' \
    'absl-py==2.1.0' 'six==1.17.0' 'boto3==1.35.99' 'setuptools<81' \
    >>"$BUILD_LOG" 2>&1
uv pip install --python "$VENV_DIR/bin/python" "$LOADGEN_SRC" \
    >>"$BUILD_LOG" 2>&1

cd "$WORK_DIR"
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export LOG_PATH="$WORK_DIR/loadgen_logs"
export SKIP_VERIFY_ACCURACY=1
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_PATH" "$WORK_DIR/result"

{
    echo "GPU memory immediately before accuracy run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader
} | tee -a "$RUN_LOG"

FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
if [[ ! "$FREE_MIB" =~ ^[0-9]+$ ]] || (( FREE_MIB < 12000 )); then
    echo "insufficient free GPU memory: ${FREE_MIB:-unknown} MiB (need 12000 MiB)" | tee -a "$RUN_LOG" >&2
    exit 1
fi

"$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy 2>&1 | tee -a "$RUN_LOG"

cp "$LOG_PATH/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
cp "$LOG_PATH/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"

"$VENV_DIR/bin/python" accuracy-squad.py \
    --vocab_file "$MODEL_DIR/vocab.txt" \
    --val_data "$SQUAD_DIR/dev-v1.1.json" \
    --log_file "$LOG_PATH/mlperf_log_accuracy.json" \
    --out_file "$WORK_DIR/result/predictions.json" \
    --features_cache_file "$WORK_DIR/eval_features.pickle" \
    > >(tee "$OUTPUT_DIR/accuracy.txt" | tee -a "$EVAL_LOG") \
    2> >(tee -a "$EVAL_LOG" >&2)

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
