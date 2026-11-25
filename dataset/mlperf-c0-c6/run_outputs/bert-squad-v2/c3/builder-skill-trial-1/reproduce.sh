#!/usr/bin/env bash
set -euo pipefail

TARGET=""; STAGE=""; FROM_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --from) FROM_STAGE="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$TARGET" != bert-squad ]]; then
  echo "usage: $0 --target bert-squad [--stage source|env|preprocess|run] [--from source|env|preprocess|run]" >&2
  exit 2
fi
if [[ -n "$STAGE" && -n "$FROM_STAGE" ]]; then
  echo "--stage and --from are mutually exclusive" >&2; exit 2
fi

START_DIR="$(pwd -P)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_SOURCE="$SCRIPT_DIR/repo"
WORK_DIR="$START_DIR/$TARGET"
SRC_DIR="$WORK_DIR/src"
VENV_DIR="$WORK_DIR/venv"
OUTPUT_VALUE="${OUTPUT_DIR:-$WORK_DIR/out}"
mkdir -p "$WORK_DIR" "$OUTPUT_VALUE"
OUTPUT_ABS="$(cd "$OUTPUT_VALUE" && pwd -P)"

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
export UV_LINK_MODE=copy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for f in "$REPO_SOURCE/language/bert/run.py" "$SQUAD_DIR/dev-v1.1.json" \
  "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt" \
  "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"; do
  [[ -e "$f" ]] || { echo "missing required input: $f" >&2; exit 1; }
done

STAGES=(source env preprocess run)
stage_index() { case "$1" in source) echo 0;; env) echo 1;; preprocess) echo 2;; run) echo 3;; *) return 1;; esac; }
if [[ -n "$STAGE" ]]; then stage_index "$STAGE" >/dev/null || { echo "invalid stage: $STAGE" >&2; exit 2; }; fi
if [[ -n "$FROM_STAGE" ]]; then stage_index "$FROM_STAGE" >/dev/null || { echo "invalid stage: $FROM_STAGE" >&2; exit 2; }; fi
if [[ -n "$FROM_STAGE" ]]; then
  from_i="$(stage_index "$FROM_STAGE")"
  for s in "${STAGES[@]}"; do
    (( $(stage_index "$s") < from_i )) || rm -f "$WORK_DIR/.$s.success"
  done
fi

run_source() {
  : > "$WORK_DIR/source.log"
  if [[ -e "$SRC_DIR" ]]; then
    [[ "$SRC_DIR" == "$WORK_DIR/src" && "$WORK_DIR" == "$START_DIR/bert-squad" ]] || return 1
    rm -rf "$SRC_DIR"
  fi
  mkdir -p "$SRC_DIR"
  cp -a "$REPO_SOURCE/." "$SRC_DIR/"
  if [[ -d "$SRC_DIR/language/bert/DeepLearningExamples" && ! -L "$SRC_DIR/language/bert/DeepLearningExamples" ]]; then
    rmdir "$SRC_DIR/language/bert/DeepLearningExamples"
  fi
  ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$SRC_DIR/language/bert/DeepLearningExamples"
  local bert_data="$SRC_DIR/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2"
  mkdir -p "$bert_data" "$SRC_DIR/language/bert/build/result"
  ln -s "$SQUAD_DIR/dev-v1.1.json" "$SRC_DIR/language/bert/build/data/dev-v1.1.json"
  ln -s "$MODEL_DIR/model.pytorch" "$bert_data/model.pytorch"
  ln -s "$MODEL_DIR/vocab.txt" "$bert_data/vocab.txt"
  sed -i '/self\.model\.load_state_dict(torch\.load(model_file), strict=False)/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$SRC_DIR/language/bert/pytorch_SUT.py"
  {
    echo source_snapshot=3fbc329939999c13d0a7b5e67fb2092287e06047
    echo deeplearningexamples_snapshot=729963dd47e7c8bd462ad10bfac7a7b0b604e6dd
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
  } >> "$WORK_DIR/source.log"
  touch "$WORK_DIR/.source.success"
}

run_env() {
  : > "$WORK_DIR/build.log"
  local py
  py="$(uv python find 3.12)"
  uv venv --python "$py" --clear "$VENV_DIR" >> "$WORK_DIR/build.log" 2>&1
  uv pip install --offline --python "$VENV_DIR/bin/python" \
    torch==2.6.0 transformers==4.46.3 numpy==1.26.4 absl-py==2.5.1 \
    six==1.17.0 setuptools==80.10.2 wheel==0.48.0 pybind11==2.11.1 \
    boto3==1.44.0 >> "$WORK_DIR/build.log" 2>&1
  uv pip install --offline --python "$VENV_DIR/bin/python" --no-build-isolation \
    "$SRC_DIR/loadgen" >> "$WORK_DIR/build.log" 2>&1
  "$VENV_DIR/bin/python" - <<'PY' >> "$WORK_DIR/build.log" 2>&1
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("mlperf_loadgen", mlperf_loadgen.__file__)
assert torch.cuda.is_available()
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
  [[ -s "$SRC_DIR/language/bert/eval_features.pickle" ]]
  touch "$WORK_DIR/.preprocess.success"
}

run_gpu_and_evaluate() {
  : > "$WORK_DIR/gpu-check.log"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | tee -a "$WORK_DIR/gpu-check.log"
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 8192 )) || {
    echo "insufficient free GPU memory: ${free_mib:-unknown} MiB; require 8192 MiB" | tee -a "$WORK_DIR/gpu-check.log" >&2
    return 1
  }
  rm -f "$OUTPUT_ABS"/mlperf_log_{accuracy.json,detail.txt,summary.txt,trace.json} \
    "$OUTPUT_ABS/accuracy.txt" "$WORK_DIR/predictions.json"
  : > "$WORK_DIR/run.log"; : > "$WORK_DIR/evaluator.log"; : > "$WORK_DIR/evaluator-stderr.log"
  local max_args=()
  if [[ -n "${REPRO_MAX_EXAMPLES:-}" ]]; then
    [[ "$REPRO_MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]] || return 2
    max_args=(--max_examples "$REPRO_MAX_EXAMPLES")
  fi
  (
    cd "$SRC_DIR/language/bert"
    export LOG_PATH="$OUTPUT_ABS" ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt" DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export SKIP_VERIFY_ACCURACY=1
    "$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
  ) 2>&1 | tee "$WORK_DIR/run.log"
  (
    cd "$SRC_DIR/language/bert"
    "$VENV_DIR/bin/python" accuracy-squad.py --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" --log_file "$OUTPUT_ABS/mlperf_log_accuracy.json" \
      --out_file "$WORK_DIR/predictions.json" \
      --features_cache_file "$SRC_DIR/language/bert/eval_features.pickle" "${max_args[@]}"
  ) 2> >(tee "$WORK_DIR/evaluator-stderr.log" >&2) | tee "$WORK_DIR/evaluator.log" "$OUTPUT_ABS/accuracy.txt"
  [[ -s "$OUTPUT_ABS/mlperf_log_accuracy.json" && -s "$OUTPUT_ABS/mlperf_log_detail.txt" && -s "$OUTPUT_ABS/accuracy.txt" ]]
  grep -q '^pytorch device: cuda:' "$WORK_DIR/run.log"
  grep -q '"effective_test_mode", "value": "AccuracyOnly"' "$OUTPUT_ABS/mlperf_log_detail.txt"
  "$VENV_DIR/bin/python" - "$OUTPUT_ABS/accuracy.txt" <<'PY'
import json, math, sys
found=[]
for line in open(sys.argv[1]):
    try: value=json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(value, dict) and set(value)=={"exact_match","f1"}: found.append(value)
assert len(found)==1 and all(isinstance(found[0][k],(int,float)) and math.isfinite(found[0][k]) for k in found[0])
print("validated", json.dumps(found[0], sort_keys=True))
PY
  touch "$WORK_DIR/.run.success"
}

run_one() { case "$1" in source) run_source;; env) run_env;; preprocess) run_preprocess;; run) run_gpu_and_evaluate;; esac; }
if [[ -n "$STAGE" ]]; then run_one "$STAGE"; exit 0; fi
start_i=0; [[ -z "$FROM_STAGE" ]] || start_i="$(stage_index "$FROM_STAGE")"
for s in "${STAGES[@]}"; do
  (( $(stage_index "$s") >= start_i )) || continue
  if [[ -f "$WORK_DIR/.$s.success" ]]; then echo "stage $s already complete"; continue; fi
  echo "starting stage $s"; run_one "$s"
done
