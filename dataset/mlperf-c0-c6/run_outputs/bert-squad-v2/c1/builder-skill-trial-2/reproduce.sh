#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad [--stage env|preprocess|gpu|evaluate] [--from env|preprocess|gpu|evaluate] [--max-examples N]" >&2
  exit 2
}

TARGET=""
ONLY_STAGE=""
FROM_STAGE=""
MAX_EXAMPLES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; TARGET="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; ONLY_STAGE="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; FROM_STAGE="$2"; shift 2 ;;
    --max-examples) [[ $# -ge 2 ]] || usage; MAX_EXAMPLES="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$TARGET" == "bert-squad" ]] || usage
[[ -z "$ONLY_STAGE" || -z "$FROM_STAGE" ]] || usage
[[ -z "$MAX_EXAMPLES" || "$MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]] || usage

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="$SCRIPT_DIR/repo"
TARGET_DIR="$SCRIPT_DIR/$TARGET"
WORK_DIR="$TARGET_DIR/work"
SOURCE_DIR="$WORK_DIR/source"
VENV_DIR="$WORK_DIR/venv"
MARKER_DIR="$TARGET_DIR/.stages"

CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
SQUAD_DIR=${SQUAD_DIR:-$CACHE_ROOT/bert-squad/data}
MODEL_DIR=${MODEL_DIR:-$CACHE_ROOT/bert-squad/model}
DEEPLEARNING_EXAMPLES_DIR=${DEEPLEARNING_EXAMPLES_DIR:-$CACHE_ROOT/bert-squad/DeepLearningExamples}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
export UV_EXCLUDE_NEWER=2026-11-10
OUTPUT_DIR=${OUTPUT_DIR:-$TARGET_DIR/out}
mkdir -p "$TARGET_DIR" "$WORK_DIR" "$MARKER_DIR" "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

stages=(env preprocess gpu evaluate)
stage_index() {
  case "$1" in
    env) echo 0 ;; preprocess) echo 1 ;; gpu) echo 2 ;; evaluate) echo 3 ;; *) usage ;;
  esac
}

if [[ -n "$FROM_STAGE" ]]; then
  start=$(stage_index "$FROM_STAGE")
  for ((i=start; i<${#stages[@]}; i++)); do
    rm -f "$MARKER_DIR/${stages[$i]}.ok"
  done
  if (( start <= 0 )); then
    [[ "$SOURCE_DIR" == "$WORK_DIR/source" && "$VENV_DIR" == "$WORK_DIR/venv" ]] || exit 2
    rm -rf "$SOURCE_DIR" "$VENV_DIR"
    rm -f "$OUTPUT_DIR/build.log"
  fi
  if (( start <= 1 )); then
    rm -f "$WORK_DIR/eval_features.pickle" "$OUTPUT_DIR/preprocess.log"
  fi
  if (( start <= 2 )); then
    rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" "$OUTPUT_DIR/gpu.log"
  fi
  if (( start <= 3 )); then
    rm -f "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log" "$WORK_DIR/predictions.json"
  fi
fi

run_env() {
  [[ -d "$REPO_DIR/language/bert" && -d "$REPO_DIR/loadgen" ]] || { echo "missing repository at $REPO_DIR" >&2; return 1; }
  [[ -r "$SQUAD_DIR/dev-v1.1.json" && -r "$MODEL_DIR/model.pytorch" && -r "$MODEL_DIR/vocab.txt" ]] || { echo "missing BERT inputs" >&2; return 1; }
  [[ -r "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py" ]] || { echo "missing pinned DeepLearningExamples input" >&2; return 1; }
  {
    echo "stage: env"
    date --iso-8601=seconds
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    cp -a "$REPO_DIR" "$SOURCE_DIR"
    rmdir "$SOURCE_DIR/language/bert/DeepLearningExamples"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$SOURCE_DIR/language/bert/DeepLearningExamples"
    # Required task evidence: report the device holding parameters after weights load.
    sed -i '/self\.model\.load_state_dict(torch\.load(model_file), strict=False)/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
      "$SOURCE_DIR/language/bert/pytorch_SUT.py"
    uv venv --python 3.11 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" \
      'setuptools==75.8.2' 'wheel==0.45.1' 'pybind11==2.13.6' \
      'numpy==1.26.4' 'absl-py==2.1.0' 'six==1.16.0' \
      'boto3==1.43.111' 'torch==2.4.1' 'transformers==4.45.2'
    uv pip install --python "$VENV_DIR/bin/python" "$SOURCE_DIR/loadgen"
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("python environment imports: ok")
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("transformers:", transformers.__version__, "numpy:", numpy.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch is required")
PY
    uv pip freeze --python "$VENV_DIR/bin/python"
    date --iso-8601=seconds
  } >"$OUTPUT_DIR/build.log" 2>&1
  touch "$MARKER_DIR/env.ok"
}

run_preprocess() {
  [[ -f "$MARKER_DIR/env.ok" ]] || { echo "env stage has not succeeded" >&2; return 1; }
  {
    echo "stage: preprocess"
    date --iso-8601=seconds
    cd "$SOURCE_DIR/language/bert"
    PYTHONPATH="$SOURCE_DIR/language/bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT" \
    DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" VOCAB_FILE="$MODEL_DIR/vocab.txt" \
      "$VENV_DIR/bin/python" - <<PY
from squad_QSL import SQuAD_v1_QSL
q = SQuAD_v1_QSL(total_count_override=${MAX_EXAMPLES:-None}, cache_path=r"$WORK_DIR/eval_features.pickle")
print("preprocessed_features:", len(q.eval_features))
print("query_count:", q.count)
PY
    ln -sf "$WORK_DIR/eval_features.pickle" "$SOURCE_DIR/language/bert/eval_features.pickle"
    date --iso-8601=seconds
  } >"$OUTPUT_DIR/preprocess.log" 2>&1
  touch "$MARKER_DIR/preprocess.ok"
}

run_gpu() {
  [[ -f "$MARKER_DIR/preprocess.ok" ]] || { echo "preprocess stage has not succeeded" >&2; return 1; }
  {
    echo "stage: gpu"
    date --iso-8601=seconds
    echo "GPU state immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu --format=csv,noheader
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [[ "$free_mib" =~ ^[0-9]+$ && "$free_mib" -ge 8192 ]] || { echo "insufficient free GPU memory: ${free_mib:-unknown} MiB" >&2; return 1; }
    cd "$SOURCE_DIR/language/bert"
    args=(--backend=pytorch --accuracy --scenario=Offline --user_conf=user.conf)
    if [[ -n "$MAX_EXAMPLES" ]]; then args+=(--max_examples "$MAX_EXAMPLES"); fi
    DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
    VOCAB_FILE="$MODEL_DIR/vocab.txt" \
    ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch" \
    LOG_PATH="$OUTPUT_DIR" \
    SKIP_VERIFY_ACCURACY=1 \
    PYTHONUNBUFFERED=1 \
      "$VENV_DIR/bin/python" run.py "${args[@]}"
    date --iso-8601=seconds
  } >"$OUTPUT_DIR/gpu.log" 2>&1
  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" && -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]] || { echo "LoadGen outputs missing" >&2; return 1; }
  grep -q '^pytorch device: ' "$OUTPUT_DIR/gpu.log" || { echo "device evidence missing" >&2; return 1; }
  touch "$MARKER_DIR/gpu.ok"
}

run_evaluate() {
  [[ -f "$MARKER_DIR/gpu.ok" ]] || { echo "gpu stage has not succeeded" >&2; return 1; }
  cd "$SOURCE_DIR/language/bert"
  args=(--vocab_file "$MODEL_DIR/vocab.txt" --val_data "$SQUAD_DIR/dev-v1.1.json" \
        --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" --out_file "$WORK_DIR/predictions.json" \
        --features_cache_file "$WORK_DIR/eval_features.pickle")
  if [[ -n "$MAX_EXAMPLES" ]]; then args+=(--max_examples "$MAX_EXAMPLES"); fi
  {
    echo "stage: evaluate"
    date --iso-8601=seconds
  } >"$OUTPUT_DIR/evaluator.log"
  "$VENV_DIR/bin/python" accuracy-squad.py "${args[@]}" \
    >"$OUTPUT_DIR/accuracy.txt" 2>>"$OUTPUT_DIR/evaluator.log"
  cat "$OUTPUT_DIR/accuracy.txt" >>"$OUTPUT_DIR/evaluator.log"
  date --iso-8601=seconds >>"$OUTPUT_DIR/evaluator.log"
  cat "$OUTPUT_DIR/accuracy.txt"
  grep -q '"exact_match"' "$OUTPUT_DIR/accuracy.txt" || { echo "exact_match missing" >&2; return 1; }
  grep -q '"f1"' "$OUTPUT_DIR/accuracy.txt" || { echo "f1 missing" >&2; return 1; }
  touch "$MARKER_DIR/evaluate.ok"
}

run_one() {
  local stage=$1
  [[ -f "$MARKER_DIR/$stage.ok" ]] && return 0
  case "$stage" in
    env) run_env ;; preprocess) run_preprocess ;; gpu) run_gpu ;; evaluate) run_evaluate ;;
  esac
}

if [[ -n "$ONLY_STAGE" ]]; then
  stage_index "$ONLY_STAGE" >/dev/null
  run_one "$ONLY_STAGE"
else
  start=0
  [[ -n "$FROM_STAGE" ]] && start=$(stage_index "$FROM_STAGE")
  for ((i=start; i<${#stages[@]}; i++)); do run_one "${stages[$i]}"; done
fi
