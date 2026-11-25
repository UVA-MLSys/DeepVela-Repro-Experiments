#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad [--stage environment|preprocess|run] [--from environment|preprocess|run]" >&2
  exit 2
}

target=""
stage=""
from=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; target="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; stage="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; from="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$target" == "bert-squad" ]] || usage
[[ -z "$stage" || -z "$from" ]] || usage

root_dir=$(cd "$(dirname "$0")" && pwd)
target_dir="$root_dir/$target"
work_dir="$target_dir/work"
source_dir="$work_dir/repo"
bert_dir="$source_dir/language/bert"
venv_dir="$target_dir/venv"
output_dir=${OUTPUT_DIR:-$target_dir/out}

cache_root=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
squad_dir=${SQUAD_DIR:-$cache_root/bert-squad/data}
model_dir=${MODEL_DIR:-$cache_root/bert-squad/model}
dle_dir=${DEEPLEARNING_EXAMPLES_DIR:-$cache_root/bert-squad/DeepLearningExamples}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$cache_root/uv}
export UV_EXCLUDE_NEWER=2026-11-10

python_bin="$venv_dir/bin/python"
environment_marker="$target_dir/.environment.success"
preprocess_marker="$target_dir/.preprocess.success"
run_marker="$target_dir/.run.success"

mkdir -p "$target_dir" "$output_dir"

validate_inputs() {
  [[ -f "$root_dir/repo/language/bert/run.py" ]] || { echo "missing fresh repository at $root_dir/repo" >&2; exit 1; }
  [[ -f "$squad_dir/dev-v1.1.json" ]] || { echo "missing SQuAD input" >&2; exit 1; }
  [[ -f "$model_dir/model.pytorch" && -f "$model_dir/vocab.txt" ]] || { echo "missing model input" >&2; exit 1; }
  [[ -f "$dle_dir/PyTorch/LanguageModeling/BERT/tokenization.py" ]] || { echo "missing DeepLearningExamples input" >&2; exit 1; }
  echo "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9  $squad_dir/dev-v1.1.json" | sha256sum -c -
  echo "71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f  $model_dir/model.pytorch" | sha256sum -c -
  echo "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3  $model_dir/vocab.txt" | sha256sum -c -
}

clear_from() {
  case "$1" in
    environment)
      rm -rf "$work_dir" "$venv_dir"
      rm -f "$environment_marker" "$preprocess_marker" "$run_marker"
      rm -f "$output_dir"/*.log "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" "$output_dir/accuracy.txt"
      ;;
    preprocess)
      rm -f "$preprocess_marker" "$run_marker" "$bert_dir/eval_features.pickle"
      rm -f "$output_dir/preprocess.log" "$output_dir/gpu.log" "$output_dir/evaluator.log" "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" "$output_dir/accuracy.txt"
      ;;
    run)
      rm -f "$run_marker"
      rm -rf "$bert_dir/build/logs" "$bert_dir/build/result"
      rm -f "$output_dir/gpu.log" "$output_dir/evaluator.log" "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" "$output_dir/accuracy.txt"
      ;;
    *) usage ;;
  esac
}

environment_stage() {
  [[ -f "$environment_marker" ]] && return
  validate_inputs
  rm -rf "$work_dir" "$venv_dir"
  mkdir -p "$work_dir"
  cp -a "$root_dir/repo" "$source_dir"
  if [[ -d "$bert_dir/DeepLearningExamples" && ! -L "$bert_dir/DeepLearningExamples" ]]; then
    rmdir "$bert_dir/DeepLearningExamples"
  fi
  ln -s "$dle_dir" "$bert_dir/DeepLearningExamples"
  mkdir -p "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2" "$bert_dir/build/logs" "$bert_dir/build/result"
  ln -s "$squad_dir/dev-v1.1.json" "$bert_dir/build/data/dev-v1.1.json"
  ln -s "$model_dir/model.pytorch" "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
  ln -s "$model_dir/vocab.txt" "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"

  {
    date -Is
    echo "Creating Python 3.11 environment"
    uv venv --python 3.11 --managed-python "$venv_dir"
    uv pip install --python "$python_bin" \
      "torch==2.5.1" "transformers==4.44.2" "numpy==1.26.4" \
      "absl-py==2.3.1" "six==1.17.0" "boto3==1.40.76" \
      "pybind11==2.13.6" "setuptools<81" wheel
    uv pip install --python "$python_bin" "$source_dir/loadgen"
    "$python_bin" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("mlperf_loadgen", mlperf_loadgen.__file__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
PY
    date -Is
  } 2>&1 | tee "$output_dir/build.log"
  touch "$environment_marker"
}

preprocess_stage() {
  [[ -f "$preprocess_marker" ]] && return
  [[ -f "$environment_marker" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
  local max_examples=${PREFLIGHT_MAX_EXAMPLES:-}
  {
    date -Is
    cd "$bert_dir"
    PYTHONPATH="$bert_dir/DeepLearningExamples/PyTorch/LanguageModeling/BERT" \
    DATASET_FILE="$squad_dir/dev-v1.1.json" VOCAB_FILE="$model_dir/vocab.txt" \
      "$python_bin" - "$max_examples" <<'PY'
import os, pickle, sys
from create_squad_data import read_squad_examples, convert_examples_to_features
from transformers import BertTokenizer
limit = int(sys.argv[1]) if sys.argv[1] else None
examples = read_squad_examples(os.environ["DATASET_FILE"], False, False)
if limit is not None:
    examples = examples[:limit]
features = []
tokenizer = BertTokenizer(os.environ["VOCAB_FILE"])
convert_examples_to_features(examples, tokenizer, 384, 128, 64, False, features.append, False)
with open("eval_features.pickle", "wb") as handle:
    pickle.dump(features, handle)
print("preprocessed examples", len(examples), "features", len(features))
PY
    date -Is
  } 2>&1 | tee "$output_dir/preprocess.log"
  touch "$preprocess_marker"
}

run_stage() {
  [[ -f "$run_marker" ]] && return
  [[ -f "$preprocess_marker" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  local max_examples=${PREFLIGHT_MAX_EXAMPLES:-}
  local max_args=()
  [[ -n "$max_examples" ]] && max_args=(--max_examples "$max_examples")

  mkdir -p "$bert_dir/build/logs" "$bert_dir/build/result"
  {
    date -Is
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    "$python_bin" - <<'PY'
import torch
print("pytorch device: cuda:0")
print("cuda device name:", torch.cuda.get_device_name(0))
PY
    cd "$bert_dir"
    LOG_PATH="$bert_dir/build/logs" \
    DATASET_FILE="$squad_dir/dev-v1.1.json" \
    VOCAB_FILE="$model_dir/vocab.txt" \
    ML_MODEL_FILE_WITH_PATH="$model_dir/model.pytorch" \
    SKIP_VERIFY_ACCURACY=1 \
      "$python_bin" run.py --backend=pytorch --accuracy "${max_args[@]}"
    date -Is
  } 2>&1 | tee "$output_dir/gpu.log"

  {
    date -Is
    cd "$bert_dir"
    "$python_bin" accuracy-squad.py \
      --vocab_file "$model_dir/vocab.txt" \
      --val_data "$squad_dir/dev-v1.1.json" \
      --log_file "$bert_dir/build/logs/mlperf_log_accuracy.json" \
      --out_file "$bert_dir/build/result/predictions.json" \
      --features_cache_file "$bert_dir/eval_features.pickle" \
      "${max_args[@]}"
    date -Is
  } 2>&1 | tee "$output_dir/evaluator.log" | tee "$output_dir/accuracy.txt"

  cp "$bert_dir/build/logs/mlperf_log_accuracy.json" "$output_dir/mlperf_log_accuracy.json"
  cp "$bert_dir/build/logs/mlperf_log_detail.txt" "$output_dir/mlperf_log_detail.txt"
  grep -q '^pytorch device: ' "$output_dir/gpu.log"
  "$python_bin" - "$output_dir/accuracy.txt" <<'PY'
import json, sys
result = None
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        value = json.loads(line)
    except Exception:
        continue
    if isinstance(value, dict) and {"exact_match", "f1"} <= value.keys():
        result = value
if result is None:
    raise SystemExit("evaluator result JSON missing")
print(result)
PY
  touch "$run_marker"
}

if [[ -n "$from" ]]; then
  clear_from "$from"
fi

if [[ -n "$stage" ]]; then
  case "$stage" in
    environment) environment_stage ;;
    preprocess) preprocess_stage ;;
    run) run_stage ;;
    *) usage ;;
  esac
  exit 0
fi

case "${from:-environment}" in
  environment) environment_stage; preprocess_stage; run_stage ;;
  preprocess) preprocess_stage; run_stage ;;
  run) run_stage ;;
  *) usage ;;
esac
