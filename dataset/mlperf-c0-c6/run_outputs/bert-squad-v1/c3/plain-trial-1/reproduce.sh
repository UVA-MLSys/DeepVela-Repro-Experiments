#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad" >&2
  exit 2
}

if [[ ${1:-} != "--target" || ${2:-} != "bert-squad" || $# -ne 2 ]]; then
  usage
fi

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
TARGET_ID=$2
TARGET_DIR="$ROOT_DIR/$TARGET_ID"
WORK_DIR="$TARGET_DIR/work"
VENV_DIR="$TARGET_DIR/.venv"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

if [[ ! -d "$ROOT_DIR/repo/language/bert" || ! -d "$ROOT_DIR/repo/loadgen" ]]; then
  echo "fresh repository is missing beside reproduce.sh" >&2
  exit 1
fi
for input in "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt" \
             "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"; do
  if [[ ! -r "$input" ]]; then
    echo "required staged input is missing: $input" >&2
    exit 1
  fi
done

mkdir -p "$TARGET_DIR" "$WORK_DIR"
if [[ -n ${OUTPUT_DIR:-} ]]; then
  mkdir -p "$OUTPUT_DIR"
  OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
else
  OUTPUT_DIR="$TARGET_DIR/out"
  mkdir -p "$OUTPUT_DIR"
fi
export OUTPUT_DIR

# Work only on target-private copies. This also keeps LoadGen's generated build
# files and BERT's feature cache out of the supplied source tree.
if [[ -L "$WORK_DIR/DeepLearningExamples" ]]; then
  rm "$WORK_DIR/DeepLearningExamples"
fi
cp -a "$ROOT_DIR/repo/language/bert/." "$WORK_DIR/"
mkdir -p "$TARGET_DIR/loadgen-src"
cp -a "$ROOT_DIR/repo/loadgen/." "$TARGET_DIR/loadgen-src/"
if [[ -d "$WORK_DIR/DeepLearningExamples" && ! -L "$WORK_DIR/DeepLearningExamples" ]]; then
  rmdir "$WORK_DIR/DeepLearningExamples"
fi
if [[ ! -e "$WORK_DIR/DeepLearningExamples" ]]; then
  ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$WORK_DIR/DeepLearningExamples"
fi

# The target requires this prefix, but the pinned implementation did not print
# its chosen torch device. Add the diagnostic to the private source copy.
if ! grep -q 'pytorch device: ' "$WORK_DIR/pytorch_SUT.py"; then
  sed -i '/self.version = transformers.__version__/i\        print("pytorch device: {}".format(self.dev))' "$WORK_DIR/pytorch_SUT.py"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv --python 3.11 "$VENV_DIR" >"$TARGET_DIR/venv-build.log" 2>&1
fi

uv pip install --python "$VENV_DIR/bin/python" \
  'setuptools<81' wheel pybind11 absl-py six boto3 'numpy==1.26.4' \
  'torch==2.5.1' 'transformers==4.31.0' \
  >"$TARGET_DIR/packages-build.log" 2>&1
uv pip install --python "$VENV_DIR/bin/python" --no-build-isolation \
  "$TARGET_DIR/loadgen-src" >"$TARGET_DIR/loadgen-build.log" 2>&1

export PATH="$VENV_DIR/bin:$PATH"
export PYTHONUNBUFFERED=1
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export LOG_PATH="$OUTPUT_DIR"
export SKIP_VERIFY_ACCURACY=1

# Record and enforce a modest memory floor immediately before the sole GPU run.
# BERT-Large fp32 needs far less than this on the A100, leaving headroom for
# concurrent users without assuming exclusive ownership of the device.
{
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')
  echo "selected GPU 0 free memory: ${free_mib} MiB"
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 8192 )); then
    echo "GPU 0 has less than 8192 MiB free; refusing to start" >&2
    exit 1
  fi
} | tee "$TARGET_DIR/gpu-check.log"

cd "$WORK_DIR"
python3 run.py --backend=pytorch --accuracy 2>&1 | tee "$TARGET_DIR/run.log"

set +e
python3 accuracy-squad.py \
  --vocab_file "$MODEL_DIR/vocab.txt" \
  --val_data "$SQUAD_DIR/dev-v1.1.json" \
  --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
  --out_file "$WORK_DIR/predictions.json" \
  --features_cache_file "$WORK_DIR/eval_features.pickle" \
  >"$OUTPUT_DIR/accuracy.txt" 2>"$TARGET_DIR/evaluator-stderr.log"
evaluator_status=$?
set -e
tee "$TARGET_DIR/evaluator.log" <"$OUTPUT_DIR/accuracy.txt"
cat "$TARGET_DIR/evaluator-stderr.log" >&2
if (( evaluator_status != 0 )); then
  exit "$evaluator_status"
fi

for output in mlperf_log_accuracy.json mlperf_log_detail.txt accuracy.txt; do
  if [[ ! -s "$OUTPUT_DIR/$output" ]]; then
    echo "required output was not produced: $OUTPUT_DIR/$output" >&2
    exit 1
  fi
done
