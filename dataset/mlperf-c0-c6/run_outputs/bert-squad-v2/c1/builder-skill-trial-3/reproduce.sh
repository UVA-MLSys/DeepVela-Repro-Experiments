#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
MAX_EXAMPLES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:?missing target}"; shift 2 ;;
    --stage) STAGE="${2:?missing stage}"; shift 2 ;;
    --from) FROM="${2:?missing stage}"; shift 2 ;;
    --max-examples) MAX_EXAMPLES="${2:?missing count}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "supported target: bert-squad" >&2
  exit 2
fi
if [[ -n "$STAGE" && -n "$FROM" ]]; then
  echo "--stage and --from are mutually exclusive" >&2
  exit 2
fi

: "${CACHE_ROOT:=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
: "${SQUAD_DIR:=$CACHE_ROOT/bert-squad/data}"
: "${MODEL_DIR:=$CACHE_ROOT/bert-squad/model}"
: "${DEEPLEARNING_EXAMPLES_DIR:=$CACHE_ROOT/bert-squad/DeepLearningExamples}"
: "${UV_CACHE_DIR:=$CACHE_ROOT/uv}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

ROOT=$(pwd -P)
mkdir -p "$OUTPUT_DIR"
OUT=$(cd "$OUTPUT_DIR" && pwd -P)
TARGET_DIR="$ROOT/$TARGET"
WORK="$TARGET_DIR/work"
SRC="$WORK/repo"
VENV="$TARGET_DIR/venv"
MARKERS="$TARGET_DIR/markers"
mkdir -p "$TARGET_DIR" "$MARKERS"

run_setup() {
  : > "$OUT/setup.log"
  {
    echo "stage: setup"
    echo "source: $ROOT/repo"
    echo "python: $(command -v python3.11)"
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    [[ $(sha256sum "$SQUAD_DIR/dev-v1.1.json" | awk '{print $1}') == 95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9 ]]
    [[ $(sha256sum "$MODEL_DIR/model.pytorch" | awk '{print $1}') == 71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f ]]
    [[ $(sha256sum "$MODEL_DIR/vocab.txt" | awk '{print $1}') == 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3 ]]
    test -f "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"
    mkdir -p "$WORK"
    if [[ ! -f "$SRC/language/bert/run.py" ]]; then
      mkdir -p "$SRC"
      cp -a "$ROOT/repo/." "$SRC/"
    fi
    BERT="$SRC/language/bert"
    mkdir -p "$BERT/build/data/bert_tf_v1_1_large_fp32_384_v2" "$BERT/build/logs" "$BERT/build/result"
    ln -sfn "$SQUAD_DIR/dev-v1.1.json" "$BERT/build/data/dev-v1.1.json"
    ln -sfn "$MODEL_DIR/model.pytorch" "$BERT/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
    ln -sfn "$MODEL_DIR/vocab.txt" "$BERT/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
    rm -f "$BERT/DeepLearningExamples/DeepLearningExamples"
    if [[ -d "$BERT/DeepLearningExamples" && ! -L "$BERT/DeepLearningExamples" ]]; then
      rmdir "$BERT/DeepLearningExamples"
    fi
    ln -sfn "$DEEPLEARNING_EXAMPLES_DIR" "$BERT/DeepLearningExamples"
    if ! grep -q 'pytorch device: ' "$BERT/pytorch_SUT.py"; then
      sed -i '/self.version = transformers.__version__/i\        print("pytorch device: {}".format(self.dev))' "$BERT/pytorch_SUT.py"
    fi
    if [[ ! -x "$VENV/bin/python" ]]; then
      uv venv --python python3.11 "$VENV"
    fi
    uv pip install --python "$VENV/bin/python" \
      torch==2.5.1 transformers==4.46.3 numpy==1.26.4 \
      absl-py==2.3.1 six==1.17.0 setuptools==75.8.0 \
      pybind11==2.11.1 boto3==1.40.76
    if ! "$VENV/bin/python" -c 'import mlperf_loadgen' 2>/dev/null; then
      uv pip install --python "$VENV/bin/python" --no-build-isolation "$SRC/loadgen"
    fi
    uv pip freeze --python "$VENV/bin/python" | sort > "$TARGET_DIR/requirements.lock"
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "numpy", numpy.__version__)
print("cuda available", torch.cuda.is_available())
assert torch.cuda.is_available()
PY
  } 2>&1 | tee -a "$OUT/setup.log"
  touch "$MARKERS/setup.ok"
}

gpu_preflight() {
  : > "$OUT/gpu_preflight.log"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader | tee -a "$OUT/gpu_preflight.log"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if (( free_mib < 8192 )); then
    echo "insufficient free GPU memory: ${free_mib} MiB; require 8192 MiB" | tee -a "$OUT/gpu_preflight.log" >&2
    return 1
  fi
}

run_accuracy() {
  local count="$1"
  local label="$2"
  local logdir="$3"
  local eval_out="$4"
  local run_log="$OUT/${label}_run.log"
  local eval_log="$OUT/${label}_evaluator.log"
  mkdir -p "$logdir"
  gpu_preflight
  cd "$SRC/language/bert"
  local extra=()
  if [[ -n "$count" ]]; then extra=(--max_examples "$count"); fi
  export LOG_PATH="$logdir"
  export SKIP_VERIFY_ACCURACY=1
  export MLC_MAX_NUM_THREADS=1
  "$VENV/bin/python" run.py --backend=pytorch --accuracy "${extra[@]}" > "$run_log" 2>&1
  grep -q 'pytorch device: cuda:0' "$run_log"
  "$VENV/bin/python" accuracy-squad.py \
    --val_data "$SQUAD_DIR/dev-v1.1.json" \
    --vocab_file "$MODEL_DIR/vocab.txt" \
    --log_file "$logdir/mlperf_log_accuracy.json" \
    --out_file "$logdir/predictions.json" \
    --features_cache_file "$SRC/language/bert/eval_features.pickle" \
    "${extra[@]}" > "$eval_out" 2> "$eval_log"
  cat "$eval_out" >> "$eval_log"
}

run_preflight() {
  local count="${MAX_EXAMPLES:-8}"
  local pre="$OUT/preflight"
  mkdir -p "$pre"
  run_accuracy "$count" preflight "$pre" "$pre/accuracy.txt"
  test -s "$pre/mlperf_log_accuracy.json"
  test -s "$pre/mlperf_log_detail.txt"
  grep -q '"exact_match"' "$pre/accuracy.txt"
  touch "$MARKERS/preflight.ok"
}

run_full() {
  rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
    "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json" \
    "$OUT/accuracy.txt" "$OUT/run.log" "$OUT/evaluator.log" \
    "$MARKERS/run.ok"
  run_accuracy "" full "$OUT" "$OUT/accuracy.txt"
  mv "$OUT/full_run.log" "$OUT/run.log"
  mv "$OUT/full_evaluator.log" "$OUT/evaluator.log"
  test -s "$OUT/mlperf_log_accuracy.json"
  test -s "$OUT/mlperf_log_detail.txt"
  grep -q 'pytorch device: cuda:0' "$OUT/run.log"
  grep -q '"exact_match"' "$OUT/accuracy.txt"
  touch "$MARKERS/run.ok"
}

case "${STAGE:-}" in
  setup) run_setup ;;
  preflight) [[ -f "$MARKERS/setup.ok" ]] || run_setup; run_preflight ;;
  run) [[ -f "$MARKERS/setup.ok" ]] || run_setup; run_full ;;
  "")
    case "${FROM:-setup}" in
      setup) run_setup; run_full ;;
      preflight) [[ -f "$MARKERS/setup.ok" ]] || run_setup; run_preflight; run_full ;;
      run) [[ -f "$MARKERS/setup.ok" ]] || run_setup; run_full ;;
      *) echo "unknown --from stage: $FROM" >&2; exit 2 ;;
    esac
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
