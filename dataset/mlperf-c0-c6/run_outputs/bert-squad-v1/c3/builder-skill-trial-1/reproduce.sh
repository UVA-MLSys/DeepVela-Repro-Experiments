#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --stage)
      STAGE="${2:-}"
      shift 2
      ;;
    --from)
      FROM_STAGE="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "usage: $0 --target bert-squad [--stage source|env|preprocess|run] [--from source|env|preprocess|run]" >&2
  exit 2
fi
if [[ -n "$STAGE" && -n "$FROM_STAGE" ]]; then
  echo "--stage and --from are mutually exclusive" >&2
  exit 2
fi

START_DIR="$(pwd -P)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_SOURCE="$SCRIPT_DIR/repo"
WORK_DIR="$START_DIR/$TARGET"
SRC_DIR="$WORK_DIR/src"
VENV_DIR="$WORK_DIR/venv"
DEFAULT_OUTPUT="$WORK_DIR/out"
OUTPUT_VALUE="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
mkdir -p "$WORK_DIR" "$OUTPUT_VALUE"
OUTPUT_ABS="$(cd "$OUTPUT_VALUE" && pwd -P)"

CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER="2026-11-10"
export UV_LINK_MODE=copy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for required in "$REPO_SOURCE/language/bert/run.py" "$SQUAD_DIR/dev-v1.1.json" \
  "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt" \
  "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 1
  fi
done

STAGES=(source env preprocess run)
valid_stage() {
  local candidate="$1"
  local item
  for item in "${STAGES[@]}"; do
    [[ "$candidate" == "$item" ]] && return 0
  done
  return 1
}
if [[ -n "$STAGE" ]] && ! valid_stage "$STAGE"; then
  echo "invalid stage: $STAGE" >&2
  exit 2
fi
if [[ -n "$FROM_STAGE" ]] && ! valid_stage "$FROM_STAGE"; then
  echo "invalid stage: $FROM_STAGE" >&2
  exit 2
fi

stage_index() {
  case "$1" in
    source) echo 0 ;;
    env) echo 1 ;;
    preprocess) echo 2 ;;
    run) echo 3 ;;
  esac
}

if [[ -n "$FROM_STAGE" ]]; then
  from_index="$(stage_index "$FROM_STAGE")"
  for stage_name in "${STAGES[@]}"; do
    if (( $(stage_index "$stage_name") >= from_index )); then
      rm -f "$WORK_DIR/.$stage_name.success"
    fi
  done
fi

run_source() {
  : > "$WORK_DIR/source.log"
  if [[ -e "$SRC_DIR" ]]; then
    if [[ "$SRC_DIR" != "$WORK_DIR/src" || "$WORK_DIR" != "$START_DIR/bert-squad" ]]; then
      echo "refusing to replace unexpected source path: $SRC_DIR" >&2
      return 1
    fi
    rm -rf "$SRC_DIR"
  fi
  mkdir -p "$SRC_DIR"
  cp -a "$REPO_SOURCE/." "$SRC_DIR/"
  rmdir "$SRC_DIR/language/bert/DeepLearningExamples"
  ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$SRC_DIR/language/bert/DeepLearningExamples"
  mkdir -p "$SRC_DIR/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2" \
    "$SRC_DIR/language/bert/build/result"
  ln -s "$SQUAD_DIR/dev-v1.1.json" "$SRC_DIR/language/bert/build/data/dev-v1.1.json"
  ln -s "$MODEL_DIR/model.pytorch" \
    "$SRC_DIR/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
  ln -s "$MODEL_DIR/vocab.txt" \
    "$SRC_DIR/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
  sed -i '/self\.model\.load_state_dict(torch\.load(model_file), strict=False)/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
    "$SRC_DIR/language/bert/pytorch_SUT.py"
  {
    echo "source_snapshot=3fbc329939999c13d0a7b5e67fb2092287e06047"
    echo "deeplearningexamples_snapshot=729963dd47e7c8bd462ad10bfac7a7b0b604e6dd"
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
  } >> "$WORK_DIR/source.log"
  touch "$WORK_DIR/.source.success"
}

run_env() {
  : > "$WORK_DIR/build.log"
  python_cmd="$(uv python find 3.12)"
  if [[ ! -f "$(dirname "$(dirname "$python_cmd")")/include/python3.12/Python.h" ]]; then
    echo "Python 3.12 development headers not found for $python_cmd" >> "$WORK_DIR/build.log"
    return 1
  fi
  uv venv --python "$python_cmd" --clear "$VENV_DIR" >> "$WORK_DIR/build.log" 2>&1
  uv pip install --offline --python "$VENV_DIR/bin/python" \
    'torch==2.6.0' 'transformers==4.46.3' 'numpy==1.26.4' \
    'absl-py==2.5.1' 'six==1.17.0' 'setuptools==80.10.2' \
    'wheel==0.48.0' 'pybind11==2.11.1' 'boto3==1.44.0' \
    >> "$WORK_DIR/build.log" 2>&1
  uv pip install --offline --python "$VENV_DIR/bin/python" --no-build-isolation \
    "$SRC_DIR/loadgen" >> "$WORK_DIR/build.log" 2>&1
  "$VENV_DIR/bin/python" - <<'PY' >> "$WORK_DIR/build.log" 2>&1
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("mlperf_loadgen", mlperf_loadgen.__file__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
PY
  touch "$WORK_DIR/.env.success"
}

run_preprocess() {
  : > "$WORK_DIR/preprocess.log"
  (
    cd "$SRC_DIR/language/bert"
    PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
      VOCAB_FILE="$MODEL_DIR/vocab.txt" DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
      "$VENV_DIR/bin/python" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL()
print("preprocessed_features", len(qsl.eval_features))
PY
  ) >> "$WORK_DIR/preprocess.log" 2>&1
  test -s "$SRC_DIR/language/bert/eval_features.pickle"
  touch "$WORK_DIR/.preprocess.success"
}

run_gpu_and_evaluate() {
  : > "$WORK_DIR/gpu-check.log"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | tee -a "$WORK_DIR/gpu-check.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 8192 )); then
    echo "insufficient free GPU memory: ${free_mib:-unknown} MiB; require 8192 MiB" | tee -a "$WORK_DIR/gpu-check.log" >&2
    return 1
  fi

  rm -f "$OUTPUT_ABS/mlperf_log_accuracy.json" "$OUTPUT_ABS/mlperf_log_detail.txt" \
    "$OUTPUT_ABS/mlperf_log_summary.txt" "$OUTPUT_ABS/mlperf_log_trace.json" \
    "$OUTPUT_ABS/accuracy.txt" "$WORK_DIR/predictions.json"
  : > "$WORK_DIR/run.log"
  : > "$WORK_DIR/evaluator.log"
  : > "$WORK_DIR/evaluator-stderr.log"

  max_args=()
  if [[ -n "${REPRO_MAX_EXAMPLES:-}" ]]; then
    if [[ ! "$REPRO_MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then
      echo "REPRO_MAX_EXAMPLES must be a positive integer" >&2
      return 2
    fi
    max_args=(--max_examples "$REPRO_MAX_EXAMPLES")
  fi

  (
    cd "$SRC_DIR/language/bert"
    export LOG_PATH="$OUTPUT_ABS"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export SKIP_VERIFY_ACCURACY=1
    "$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
  ) 2>&1 | tee "$WORK_DIR/run.log"

  (
    cd "$SRC_DIR/language/bert"
    "$VENV_DIR/bin/python" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUTPUT_ABS/mlperf_log_accuracy.json" \
      --out_file "$WORK_DIR/predictions.json" \
      --features_cache_file "$SRC_DIR/language/bert/eval_features.pickle" \
      "${max_args[@]}"
  ) 2> >(tee "$WORK_DIR/evaluator-stderr.log" >&2) \
    | tee "$WORK_DIR/evaluator.log" "$OUTPUT_ABS/accuracy.txt"

  test -s "$OUTPUT_ABS/mlperf_log_accuracy.json"
  test -s "$OUTPUT_ABS/mlperf_log_detail.txt"
  test -s "$OUTPUT_ABS/accuracy.txt"
  grep -q '^pytorch device: ' "$WORK_DIR/run.log"
  grep -q '"effective_test_mode", "value": "AccuracyOnly"' "$OUTPUT_ABS/mlperf_log_detail.txt"
  "$VENV_DIR/bin/python" - "$OUTPUT_ABS/accuracy.txt" <<'PY'
import json, math, sys
lines = [line.strip() for line in open(sys.argv[1]) if line.strip()]
matches = []
for line in lines:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict) and set(value) == {"exact_match", "f1"}:
        matches.append(value)
assert len(matches) == 1
values = matches[0]
assert all(isinstance(values[k], (int, float)) and math.isfinite(values[k]) for k in values)
print("validated", json.dumps(values, sort_keys=True))
PY
  touch "$WORK_DIR/.run.success"
}

run_one_stage() {
  case "$1" in
    source) run_source ;;
    env) run_env ;;
    preprocess) run_preprocess ;;
    run) run_gpu_and_evaluate ;;
  esac
}

if [[ -n "$STAGE" ]]; then
  run_one_stage "$STAGE"
  exit 0
fi

start_index=0
if [[ -n "$FROM_STAGE" ]]; then
  start_index="$(stage_index "$FROM_STAGE")"
fi
for stage_name in "${STAGES[@]}"; do
  current_index="$(stage_index "$stage_name")"
  if (( current_index < start_index )); then
    continue
  fi
  if [[ -f "$WORK_DIR/.$stage_name.success" ]]; then
    echo "stage $stage_name already complete"
    continue
  fi
  echo "starting stage $stage_name"
  run_one_stage "$stage_name"
done
