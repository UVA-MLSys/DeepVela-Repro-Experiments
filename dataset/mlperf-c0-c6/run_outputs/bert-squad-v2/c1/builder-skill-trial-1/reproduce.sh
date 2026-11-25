#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START_STAGE="environment"
MAX_EXAMPLES=""

usage() {
  echo "usage: $0 --target bert-squad [--stage environment|preflight|run] [--from environment|preflight|run] [--max-examples N]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE="stage"; START_STAGE="${2:-}"; shift 2 ;;
    --from) MODE="from"; START_STAGE="${2:-}"; shift 2 ;;
    --max-examples) MAX_EXAMPLES="${2:-}"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then usage; exit 2; fi
case "$START_STAGE" in environment|preflight|run) ;; *) usage; exit 2 ;; esac
if [[ -n "$MAX_EXAMPLES" ]] && ! [[ "$MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-examples must be a positive integer" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
TARGET_DIR="$SCRIPT_DIR/$TARGET"
WORK_DIR="$TARGET_DIR/work"
BERT_DIR="$WORK_DIR/language/bert"
LOADGEN_DIR="$WORK_DIR/loadgen"
VENV_DIR="$TARGET_DIR/venv"
OUTPUT_DIR="${OUTPUT_DIR:-$TARGET_DIR/out}"

CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
SQUAD_DIR="${SQUAD_DIR:-$CACHE_ROOT/bert-squad/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/bert-squad/model}"
DEEPLEARNING_EXAMPLES_DIR="${DEEPLEARNING_EXAMPLES_DIR:-$CACHE_ROOT/bert-squad/DeepLearningExamples}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_EXCLUDE_NEWER="2026-11-10"

ENV_MARKER="$TARGET_DIR/.environment.success"
PREFLIGHT_MARKER="$TARGET_DIR/.preflight.success"
RUN_MARKER="$OUTPUT_DIR/.run.success"

require_file() { [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "missing required directory: $1" >&2; exit 1; }; }

validate_layout() {
  require_file "$REPO_DIR/language/bert/run.py"
  require_file "$REPO_DIR/language/bert/accuracy-squad.py"
  require_file "$REPO_DIR/loadgen/setup.py"
  require_file "$SQUAD_DIR/dev-v1.1.json"
  require_file "$MODEL_DIR/model.pytorch"
  require_file "$MODEL_DIR/vocab.txt"
  require_file "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"
  require_dir "$UV_CACHE_DIR"
}

reset_from() {
  case "$1" in
    environment)
      [[ "$TARGET_DIR" == "$SCRIPT_DIR/$TARGET" ]] || exit 1
      rm -rf "$WORK_DIR" "$VENV_DIR"
      rm -f "$ENV_MARKER" "$PREFLIGHT_MARKER"
      ;&
    preflight)
      rm -rf "$OUTPUT_DIR/preflight" "$TARGET_DIR/preflight"
      rm -f "$PREFLIGHT_MARKER"
      ;&
    run)
      rm -rf "$TARGET_DIR/full"
      rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" \
        "$OUTPUT_DIR/mlperf_log_trace.json" \
        "$OUTPUT_DIR/accuracy.txt" \
        "$OUTPUT_DIR/run.log" \
        "$OUTPUT_DIR/evaluator.log" \
        "$OUTPUT_DIR/timing.log" \
        "$RUN_MARKER"
      ;;
  esac
}

stage_environment_impl() {
  if [[ -f "$ENV_MARKER" ]]; then echo "environment stage already complete"; return; fi
  mkdir -p "$TARGET_DIR" "$OUTPUT_DIR" "$WORK_DIR/language"
  echo "environment stage start: $(date --iso-8601=seconds)"
  validate_layout
  cp -a "$REPO_DIR/language/bert" "$BERT_DIR"
  cp -a "$REPO_DIR/loadgen" "$LOADGEN_DIR"
  rmdir "$BERT_DIR/DeepLearningExamples"
  ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"
  mkdir -p "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2" "$BERT_DIR/build/result"
  ln -s "$SQUAD_DIR/dev-v1.1.json" "$BERT_DIR/build/data/dev-v1.1.json"
  ln -s "$MODEL_DIR/model.pytorch" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
  ln -s "$MODEL_DIR/vocab.txt" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"

  patch "$BERT_DIR/pytorch_SUT.py" <<'PATCH_DEVICE'
--- pytorch_SUT.py.orig
+++ pytorch_SUT.py
@@ -70,6 +70,7 @@
         )
         self.model.load_state_dict(torch.load(model_file), strict=False)
+        print("pytorch device: {}".format(next(self.model.parameters()).device))
 
         print("Constructing SUT...")
PATCH_DEVICE

  uv venv --python 3.10 "$VENV_DIR"
  uv pip install --offline --python "$VENV_DIR/bin/python" \
    'setuptools==75.1.0' 'wheel==0.44.0' 'pybind11==2.11.1'
  uv pip install --offline --python "$VENV_DIR/bin/python" \
    'numpy==1.26.4' 'torch==2.4.1' 'transformers==4.44.2' \
    'absl-py==2.1.0' 'six==1.16.0' 'boto3==1.43.35'
  uv pip install --offline --no-build-isolation --python "$VENV_DIR/bin/python" "$LOADGEN_DIR"
  uv pip freeze --python "$VENV_DIR/bin/python"
  "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("imports ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "numpy", numpy.__version__)
PY
  sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
  touch "$ENV_MARKER"
  echo "environment stage end: $(date --iso-8601=seconds)"
}

stage_environment() { stage_environment_impl 2>&1 | tee "$OUTPUT_DIR/build.log"; }

run_accuracy() {
  local count="$1" log_dir="$2" run_log="$3" eval_log="$4" accuracy_file="$5" feature_cache="$6" predictions_file="$7"
  local count_args=()
  if [[ -n "$count" ]]; then count_args=(--max_examples "$count"); fi
  mkdir -p "$log_dir" "$(dirname "$feature_cache")" "$(dirname "$predictions_file")"
  echo "gpu state immediately before run: $(date --iso-8601=seconds)" | tee "$run_log"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used \
    --format=csv,noheader | tee -a "$run_log"
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 8192 )); then
    echo "insufficient free GPU memory: ${free_mib:-unknown} MiB; require at least 8192 MiB" | tee -a "$run_log" >&2
    return 1
  fi
  (
    cd "$BERT_DIR"
    LOG_PATH="$log_dir" SKIP_VERIFY_ACCURACY=1 \
    ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch" \
    DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" VOCAB_FILE="$MODEL_DIR/vocab.txt" \
    "$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy "${count_args[@]}"
  ) 2>&1 | tee -a "$run_log"
  (
    cd "$BERT_DIR"
    "$VENV_DIR/bin/python" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$log_dir/mlperf_log_accuracy.json" --out_file "$predictions_file" \
      --features_cache_file "$feature_cache" "${count_args[@]}"
  ) 2>&1 | tee "$eval_log" "$accuracy_file"
}

stage_preflight() {
  if [[ -f "$PREFLIGHT_MARKER" ]]; then echo "preflight stage already complete"; return; fi
  [[ -f "$ENV_MARKER" ]] || { echo "environment stage has not completed" >&2; exit 1; }
  local preflight_dir="$OUTPUT_DIR/preflight" started="$SECONDS"
  run_accuracy "${MAX_EXAMPLES:-8}" "$preflight_dir" \
    "$preflight_dir/run.log" "$preflight_dir/evaluator.log" "$preflight_dir/accuracy.txt" \
    "$TARGET_DIR/preflight/eval_features.pickle" "$TARGET_DIR/preflight/predictions.json"
  grep -q '^pytorch device: ' "$preflight_dir/run.log"
  grep -q '"exact_match"' "$preflight_dir/accuracy.txt"
  echo "preflight elapsed_seconds=$((SECONDS-started))" | tee "$preflight_dir/timing.log"
  touch "$PREFLIGHT_MARKER"
}

stage_run() {
  if [[ -f "$RUN_MARKER" ]]; then echo "run stage already complete"; return; fi
  [[ -f "$ENV_MARKER" ]] || { echo "environment stage has not completed" >&2; exit 1; }
  local started="$SECONDS"
  run_accuracy "$MAX_EXAMPLES" "$OUTPUT_DIR" \
    "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt" \
    "$TARGET_DIR/full/eval_features.pickle" "$TARGET_DIR/full/predictions.json"
  require_file "$OUTPUT_DIR/mlperf_log_accuracy.json"
  require_file "$OUTPUT_DIR/mlperf_log_detail.txt"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  grep -q '"exact_match"' "$OUTPUT_DIR/accuracy.txt"
  echo "run elapsed_seconds=$((SECONDS-started))" | tee "$OUTPUT_DIR/timing.log"
  touch "$RUN_MARKER"
}

validate_layout
mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"
if [[ "$MODE" == "from" ]]; then reset_from "$START_STAGE"; fi
if [[ "$MODE" == "stage" ]]; then "stage_$START_STAGE"; exit 0; fi
stages=(environment preflight run)
run_now=false
for stage in "${stages[@]}"; do
  if [[ "$stage" == "$START_STAGE" ]]; then run_now=true; fi
  if [[ "$run_now" == true ]]; then "stage_$stage"; fi
done
