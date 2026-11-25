#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TARGET=""
MODE=all
START_STAGE=""
ONLY_STAGE=""

usage() {
  echo "usage: $0 --target bert-squad [--stage env|preprocess|run | --from env|preprocess|run]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || { usage; exit 2; }; TARGET=$2; shift 2 ;;
    --stage) [[ $# -ge 2 && -z "$START_STAGE" ]] || { usage; exit 2; }; MODE=stage; ONLY_STAGE=$2; shift 2 ;;
    --from) [[ $# -ge 2 && -z "$ONLY_STAGE" ]] || { usage; exit 2; }; MODE=from; START_STAGE=$2; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

[[ "$TARGET" == bert-squad ]] || { usage; exit 2; }
for requested in "$ONLY_STAGE" "$START_STAGE"; do
  [[ -z "$requested" || "$requested" == env || "$requested" == preprocess || "$requested" == run ]] || { usage; exit 2; }
done

: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
: "${CACHE_ROOT:=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
: "${SQUAD_DIR:=$CACHE_ROOT/bert-squad/data}"
: "${MODEL_DIR:=$CACHE_ROOT/bert-squad/model}"
: "${DEEPLEARNING_EXAMPLES_DIR:=$CACHE_ROOT/bert-squad/DeepLearningExamples}"
: "${UV_CACHE_DIR:=$CACHE_ROOT/uv}"
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

TARGET_DIR="$SCRIPT_DIR/$TARGET"
mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
WORK_DIR="$TARGET_DIR/work"
VENV_DIR="$WORK_DIR/venv"
SRC_DIR="$WORK_DIR/source"
BERT_DIR="$SRC_DIR/language/bert"
DLE_BERT_DIR="$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling/BERT"
PYTHON="$VENV_DIR/bin/python"
ENV_MARKER="$TARGET_DIR/.env.success"
PREPROCESS_MARKER="$TARGET_DIR/.preprocess.success"
RUN_MARKER="$TARGET_DIR/.run.success"

check_inputs() {
  [[ -d "$SCRIPT_DIR/repo/language/bert" && -d "$SCRIPT_DIR/repo/loadgen" ]] || { echo "missing source at $SCRIPT_DIR/repo" >&2; exit 1; }
  [[ -f "$SQUAD_DIR/dev-v1.1.json" ]] || { echo "missing SQuAD input" >&2; exit 1; }
  [[ -f "$MODEL_DIR/model.pytorch" && -f "$MODEL_DIR/vocab.txt" ]] || { echo "missing model inputs" >&2; exit 1; }
  [[ -f "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py" ]] || { echo "missing DeepLearningExamples input" >&2; exit 1; }
  echo "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9  $SQUAD_DIR/dev-v1.1.json" | sha256sum -c -
  echo "71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f  $MODEL_DIR/model.pytorch" | sha256sum -c -
  echo "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3  $MODEL_DIR/vocab.txt" | sha256sum -c -
}

clear_from() {
  case "$1" in
    env)
      rm -rf -- "$VENV_DIR" "$SRC_DIR"
      rm -f -- "$ENV_MARKER" "$PREPROCESS_MARKER" "$RUN_MARKER" "$TARGET_DIR"/*.log
      rm -f -- "$OUTPUT_DIR"/mlperf_log_* "$OUTPUT_DIR/accuracy.txt"
      ;;
    preprocess)
      rm -f -- "$PREPROCESS_MARKER" "$RUN_MARKER" "$BERT_DIR/eval_features.pickle" "$BERT_DIR/build/result/predictions.json"
      rm -f -- "$TARGET_DIR/preprocess.log" "$TARGET_DIR/run.log" "$TARGET_DIR/evaluator.log" "$TARGET_DIR/gpu_memory.log"
      rm -f -- "$OUTPUT_DIR"/mlperf_log_* "$OUTPUT_DIR/accuracy.txt"
      ;;
    run)
      rm -f -- "$RUN_MARKER" "$BERT_DIR/build/result/predictions.json"
      rm -f -- "$TARGET_DIR/run.log" "$TARGET_DIR/evaluator.log" "$TARGET_DIR/gpu_memory.log"
      rm -f -- "$OUTPUT_DIR"/mlperf_log_* "$OUTPUT_DIR/accuracy.txt"
      ;;
  esac
}

stage_env() {
  [[ -f "$ENV_MARKER" ]] && return
  check_inputs
  mkdir -p "$WORK_DIR"
  local python_bin=${PYTHON_BIN:-}
  [[ -n "$python_bin" ]] || python_bin=$(command -v python3.11 || true)
  [[ -n "$python_bin" ]] || { echo "Python 3.11 is required" >&2; exit 1; }
  {
    echo "environment start: $(date --iso-8601=seconds)"
    echo "python bootstrap: $python_bin"
    echo "uv cache: $UV_CACHE_DIR"
    uv venv --python "$python_bin" "$VENV_DIR"
    uv pip install --python "$PYTHON" --offline \
      torch==2.4.1 transformers==4.44.2 numpy==1.26.4 absl-py==2.1.0 \
      boto3==1.35.99 six==1.17.0 pybind11==2.11.1 setuptools==75.8.0 wheel==0.45.1
    mkdir -p "$SRC_DIR/language"
    cp -a "$SCRIPT_DIR/repo/language/bert" "$BERT_DIR"
    cp -a "$SCRIPT_DIR/repo/loadgen" "$SRC_DIR/loadgen"
    cp -a "$SCRIPT_DIR/repo/lon" "$SRC_DIR/lon"
    rmdir "$BERT_DIR/DeepLearningExamples"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"
    (
      cd "$BERT_DIR"
      patch -p0 <<'PATCH'
*** pytorch_SUT.py.orig
--- pytorch_SUT.py
***************
*** 74,79 ****
--- 74,80 ----
          )
          self.model.load_state_dict(torch.load(model_file), strict=False)
+         print("pytorch device: {}".format(next(self.model.parameters()).device))
  
          print("Constructing SUT...")
          self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
PATCH
    )
    uv pip install --python "$PYTHON" --offline --no-build-isolation "$SRC_DIR/loadgen"
    "$PYTHON" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("mlperf_loadgen", getattr(mlperf_loadgen, "__version__", "imported"))
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
PY
    echo "environment end: $(date --iso-8601=seconds)"
  } 2>&1 | tee "$TARGET_DIR/build.log"
  touch "$ENV_MARKER"
}

stage_preprocess() {
  [[ -f "$PREPROCESS_MARKER" ]] && return
  [[ -f "$ENV_MARKER" ]] || { echo "env stage has not succeeded" >&2; exit 1; }
  mkdir -p "$BERT_DIR/build/result"
  {
    echo "preprocess start: $(date --iso-8601=seconds)"
    cd "$BERT_DIR"
    DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" VOCAB_FILE="$MODEL_DIR/vocab.txt" \
      PYTHONPATH="$DLE_BERT_DIR:$BERT_DIR:$SRC_DIR/loadgen" "$PYTHON" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL()
print("preprocessed feature count:", len(qsl.eval_features))
PY
    echo "preprocess end: $(date --iso-8601=seconds)"
  } 2>&1 | tee "$TARGET_DIR/preprocess.log"
  touch "$PREPROCESS_MARKER"
}

stage_run() {
  [[ -f "$RUN_MARKER" ]] && return
  [[ -f "$PREPROCESS_MARKER" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  {
    echo "GPU check immediately before run: $(date --iso-8601=seconds)"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader
  } 2>&1 | tee "$TARGET_DIR/gpu_memory.log"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ && "$free_mib" -ge 6144 ]] || { echo "insufficient GPU memory: ${free_mib:-unknown} MiB" >&2; exit 1; }
  local max_args=()
  if [[ -n ${REPRO_MAX_EXAMPLES:-} ]]; then
    [[ "$REPRO_MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]] || { echo "REPRO_MAX_EXAMPLES must be positive" >&2; exit 1; }
    max_args=(--max_examples "$REPRO_MAX_EXAMPLES")
  fi
  mkdir -p "$BERT_DIR/build/result" "$OUTPUT_DIR"
  (
    cd "$BERT_DIR"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch" LOG_PATH="$OUTPUT_DIR"
    export PYTHONPATH="$DLE_BERT_DIR:$BERT_DIR:$SRC_DIR/loadgen" SKIP_VERIFY_ACCURACY=1
    "$PYTHON" run.py --backend=pytorch --accuracy "${max_args[@]}"
  ) 2>&1 | tee "$TARGET_DIR/run.log"
  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" && -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]] || { echo "LoadGen outputs missing" >&2; exit 1; }
  grep -q '^pytorch device: ' "$TARGET_DIR/run.log" || { echo "required device line missing" >&2; exit 1; }
  (
    cd "$BERT_DIR"
    PYTHONPATH="$DLE_BERT_DIR:$BERT_DIR:$SRC_DIR/loadgen" "$PYTHON" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --out_file "$BERT_DIR/build/result/predictions.json" \
      --features_cache_file "$BERT_DIR/eval_features.pickle" "${max_args[@]}"
  ) 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$TARGET_DIR/evaluator.log"
  "$PYTHON" - "$OUTPUT_DIR/accuracy.txt" <<'PY'
import json, sys
found=[]
for line in open(sys.argv[1], encoding="utf-8"):
    try: value=json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(value, dict) and {"exact_match","f1"} <= value.keys(): found.append(value)
if len(found) != 1: raise SystemExit(f"expected one evaluator result, found {len(found)}")
print("validated evaluator result:", json.dumps(found[0]))
PY
  touch "$RUN_MARKER"
}

run_stage() { case "$1" in env) stage_env;; preprocess) stage_preprocess;; run) stage_run;; esac; }
[[ "$MODE" == from ]] && clear_from "$START_STAGE"
if [[ "$MODE" == stage ]]; then
  run_stage "$ONLY_STAGE"
else
  start=0
  [[ "$MODE" == from && "$START_STAGE" == preprocess ]] && start=1
  [[ "$MODE" == from && "$START_STAGE" == run ]] && start=2
  stages=(env preprocess run)
  for ((i=start; i<3; i++)); do run_stage "${stages[$i]}"; done
fi
