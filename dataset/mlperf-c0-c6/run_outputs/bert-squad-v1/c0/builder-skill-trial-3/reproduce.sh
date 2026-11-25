#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
FROM_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE="${2:-}"; shift 2 ;;
    --from) FROM_STAGE="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "supported target: bert-squad" >&2
  exit 2
fi
case "$MODE" in env|preflight|run|all) ;; *) echo "invalid stage: $MODE" >&2; exit 2 ;; esac
case "$FROM_STAGE" in ""|env|preflight|run) ;; *) echo "invalid --from stage: $FROM_STAGE" >&2; exit 2 ;; esac

: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
: "${CACHE_ROOT:=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
: "${SQUAD_DIR:=$CACHE_ROOT/bert-squad/data}"
: "${MODEL_DIR:=$CACHE_ROOT/bert-squad/model}"
: "${DEEPLEARNING_EXAMPLES_DIR:=$CACHE_ROOT/bert-squad/DeepLearningExamples}"
: "${UV_CACHE_DIR:=$CACHE_ROOT/uv}"
export UV_CACHE_DIR
export UV_EXCLUDE_NEWER=2026-11-10
export UV_LINK_MODE=copy

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="$SCRIPT_DIR/repo"
TARGET_DIR="$SCRIPT_DIR/$TARGET"
mkdir -p "$OUTPUT_DIR" "$TARGET_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
MARKERS="$TARGET_DIR/.markers"
SRC="$TARGET_DIR/src"
VENV="$TARGET_DIR/venv"
PY="$VENV/bin/python"
BERT="$SRC/language/bert"
PREFLIGHT="$TARGET_DIR/preflight"
FULL="$TARGET_DIR/full"
mkdir -p "$MARKERS"

require_file() {
  [[ -f "$1" ]] || { echo "required file missing: $1" >&2; exit 1; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "required directory missing: $1" >&2; exit 1; }
}

require_dir "$REPO/language/bert"
require_dir "$REPO/loadgen"
require_file "$SQUAD_DIR/dev-v1.1.json"
require_file "$MODEL_DIR/model.pytorch"
require_file "$MODEL_DIR/vocab.txt"
require_dir "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT"
command -v uv >/dev/null
command -v nvidia-smi >/dev/null

clear_from() {
  local stage=$1
  case "$stage" in
    env)
      rm -rf "$SRC" "$VENV" "$PREFLIGHT" "$FULL"
      rm -f "$MARKERS/env.ok" "$MARKERS/preflight.ok" "$MARKERS/run.ok"
      rm -f "$OUTPUT_DIR/env.log" "$OUTPUT_DIR/source_setup.log" \
        "$OUTPUT_DIR/preflight_gpu.log" "$OUTPUT_DIR/preflight_run.log" \
        "$OUTPUT_DIR/preflight_evaluator.log" "$OUTPUT_DIR/run_gpu.log" \
        "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" \
        "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        "$OUTPUT_DIR/mlperf_log_detail.txt"
      ;;
    preflight)
      rm -rf "$PREFLIGHT" "$FULL"
      rm -f "$MARKERS/preflight.ok" "$MARKERS/run.ok"
      rm -f "$OUTPUT_DIR/preflight_gpu.log" "$OUTPUT_DIR/preflight_run.log" \
        "$OUTPUT_DIR/preflight_evaluator.log" "$OUTPUT_DIR/run_gpu.log" \
        "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" \
        "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        "$OUTPUT_DIR/mlperf_log_detail.txt"
      ;;
    run)
      rm -rf "$FULL"
      rm -f "$MARKERS/run.ok"
      rm -f "$OUTPUT_DIR/run_gpu.log" "$OUTPUT_DIR/run.log" \
        "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt" \
        "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt"
      ;;
  esac
}

if [[ -n "$FROM_STAGE" ]]; then
  clear_from "$FROM_STAGE"
fi

stage_env() {
  if [[ -f "$MARKERS/env.ok" ]] && grep -qx 'env-v2' "$MARKERS/env.ok"; then
    echo "env stage already complete"
    return
  fi
  if [[ -f "$MARKERS/env.ok" ]]; then
    rm -rf "$SRC" "$VENV"
    rm -f "$MARKERS/env.ok" "$MARKERS/preflight.ok" "$MARKERS/run.ok"
  fi
  {
    echo "started_utc=$(date -u +%FT%TZ)"
    echo "python_pin=3.11"
    echo "uv=$(uv --version)"
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$PY" \
      'numpy==1.26.4' 'torch==2.4.1' 'transformers==4.45.2' \
      'absl-py==2.1.0' 'six==1.16.0' 'boto3==1.40.76' 'setuptools==75.3.0' \
      'pybind11==2.11.1' 'wheel==0.44.0'
    "$PY" -c 'import numpy, torch, transformers; print("numpy", numpy.__version__); print("torch", torch.__version__); print("transformers", transformers.__version__)'
  } 2>&1 | tee "$OUTPUT_DIR/env.log"

  {
    mkdir -p "$SRC"
    cp -a "$REPO/language" "$SRC/language"
    cp -a "$REPO/loadgen" "$SRC/loadgen"
    mkdir -p "$BERT/DeepLearningExamples/PyTorch/LanguageModeling"
    cp -a "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT" \
      "$BERT/DeepLearningExamples/PyTorch/LanguageModeling/BERT"

    # The snapshot's setup.py lists a removed header only as a rebuild dependency.
    sed -i '/"trace_generator.h",/d' "$SRC/loadgen/setup.py"
    # Required target observability; this does not alter inference values.
    sed -i '/self.model.load_state_dict(torch.load(model_file), strict=False)/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$BERT/pytorch_SUT.py"

    mkdir -p "$BERT/build/data/bert_tf_v1_1_large_fp32_384_v2" "$BERT/build/result"
    ln -sfn "$SQUAD_DIR/dev-v1.1.json" "$BERT/build/data/dev-v1.1.json"
    ln -sfn "$MODEL_DIR/model.pytorch" "$BERT/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
    ln -sfn "$MODEL_DIR/vocab.txt" "$BERT/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"

    uv pip install --python "$PY" "$SRC/loadgen"
    "$PY" -c 'import mlperf_loadgen as lg; print("loadgen import ok", lg.__file__)'
    echo "finished_utc=$(date -u +%FT%TZ)"
  } 2>&1 | tee "$OUTPUT_DIR/source_setup.log"
  printf '%s\n' 'env-v2' > "$MARKERS/env.ok"
}

check_gpu() {
  local logfile=$1
  local free_mib
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader | tee "$logfile"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 8192 )); then
    echo "GPU 0 has ${free_mib:-unknown} MiB free; at least 8192 MiB required" | tee -a "$logfile" >&2
    return 1
  fi
  echo "GPU 0 free-memory check passed: $free_mib MiB" | tee -a "$logfile"
}

run_harness() {
  local log_path=$1
  shift
  (
    cd "$BERT"
    export CUDA_VISIBLE_DEVICES=0
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export LOG_PATH="$log_path"
    export SKIP_VERIFY_ACCURACY=1
    "$PY" run.py --backend=pytorch --accuracy "$@"
  )
}

run_evaluator() {
  local loadgen_log=$1
  local predictions=$2
  shift 2
  (
    cd "$BERT"
    "$PY" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$loadgen_log" \
      --out_file "$predictions" \
      --features_cache_file "$BERT/eval_features.pickle" "$@"
  )
}

stage_preflight() {
  stage_env
  if [[ -f "$MARKERS/preflight.ok" ]]; then
    echo "preflight stage already complete"
    return
  fi
  mkdir -p "$PREFLIGHT/logs" "$PREFLIGHT/result"
  check_gpu "$OUTPUT_DIR/preflight_gpu.log"
  local started=$SECONDS
  run_harness "$PREFLIGHT/logs" --max_examples 8 2>&1 | tee "$OUTPUT_DIR/preflight_run.log"
  run_evaluator "$PREFLIGHT/logs/mlperf_log_accuracy.json" \
    "$PREFLIGHT/result/predictions.json" --max_examples 8 \
    2>&1 | tee "$OUTPUT_DIR/preflight_evaluator.log"
  grep -q '^pytorch device: cuda:0' "$OUTPUT_DIR/preflight_run.log"
  grep -q '"exact_match"' "$OUTPUT_DIR/preflight_evaluator.log"
  echo "preflight_elapsed_seconds=$((SECONDS-started))" | tee -a "$OUTPUT_DIR/preflight_run.log"
  touch "$MARKERS/preflight.ok"
}

write_report() {
  "$PY" - "$OUTPUT_DIR/accuracy.txt" "$SCRIPT_DIR/report.json" <<'PY'
import json, pathlib, sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
metrics = None
for line in reversed(lines):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(obj, dict) and "exact_match" in obj and "f1" in obj:
        metrics = obj
        break
if metrics is None:
    raise SystemExit("evaluator metrics JSON not found")
out = {"bert-squad": {"exact_match": metrics["exact_match"], "f1": metrics["f1"]}}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, separators=(",", ":")) + "\n")
PY
}

stage_run() {
  stage_preflight
  if [[ -f "$MARKERS/run.ok" ]]; then
    echo "run stage already complete"
    return
  fi
  mkdir -p "$FULL/logs" "$FULL/result"
  check_gpu "$OUTPUT_DIR/run_gpu.log"
  local started=$SECONDS
  run_harness "$FULL/logs" 2>&1 | tee "$OUTPUT_DIR/run.log"
  run_evaluator "$FULL/logs/mlperf_log_accuracy.json" \
    "$FULL/result/predictions.json" 2>&1 \
    | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  cp "$FULL/logs/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
  cp "$FULL/logs/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"
  grep -q '^pytorch device: cuda:0' "$OUTPUT_DIR/run.log"
  grep -q '"exact_match"' "$OUTPUT_DIR/accuracy.txt"
  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" ]]
  [[ -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
  echo "full_run_elapsed_seconds=$((SECONDS-started))" | tee -a "$OUTPUT_DIR/run.log"
  write_report
  touch "$MARKERS/run.ok"
}

if [[ -n "$FROM_STAGE" ]]; then
  MODE=all
fi
case "$MODE" in
  env) stage_env ;;
  preflight) stage_preflight ;;
  run|all) stage_run ;;
esac
