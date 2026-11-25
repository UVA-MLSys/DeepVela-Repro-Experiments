#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad [--stage prepare|environment|preprocess|run|evaluate] [--from prepare|environment|preprocess|run|evaluate] [--max-examples N]" >&2
  exit 2
}

target=""
single_stage=""
from_stage=""
max_examples=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; target="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; single_stage="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; from_stage="$2"; shift 2 ;;
    --max-examples) [[ $# -ge 2 ]] || usage; max_examples="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$target" == "bert-squad" ]] || usage
[[ -z "$single_stage" || -z "$from_stage" ]] || usage
if [[ -n "$max_examples" && ! "$max_examples" =~ ^[1-9][0-9]*$ ]]; then
  usage
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_repo="$script_dir/repo"
target_dir="$script_dir/$target"
work_repo="$target_dir/work/repo"
bert_dir="$work_repo/language/bert"
venv_dir="$target_dir/venv"
python_bin="$venv_dir/bin/python"
state_dir="$target_dir/state"

cache_root="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
squad_dir="${SQUAD_DIR:-$cache_root/bert-squad/data}"
model_dir="${MODEL_DIR:-$cache_root/bert-squad/model}"
dle_dir="${DEEPLEARNING_EXAMPLES_DIR:-$cache_root/bert-squad/DeepLearningExamples}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache_root/uv}"
export UV_EXCLUDE_NEWER="2026-11-10"
export UV_OFFLINE=1

if [[ -n "$max_examples" ]]; then
  output_dir="${OUTPUT_DIR:-$target_dir/preflight-out}"
else
  : "${OUTPUT_DIR:?OUTPUT_DIR must be set for a full run}"
  output_dir="$OUTPUT_DIR"
fi
mkdir -p "$target_dir" "$state_dir" "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

stages=(prepare environment preprocess run evaluate)
valid_stage() {
  local candidate="$1"
  local item
  for item in "${stages[@]}"; do [[ "$candidate" == "$item" ]] && return 0; done
  return 1
}
[[ -z "$single_stage" ]] || valid_stage "$single_stage" || usage
[[ -z "$from_stage" ]] || valid_stage "$from_stage" || usage

log_versions() {
  "$python_bin" - <<'PY'
import sys
import numpy
import torch
import transformers
import mlperf_loadgen
print("python", sys.version.replace("\n", " "))
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("mlperf_loadgen", mlperf_loadgen.__file__)
PY
}

stage_prepare() {
  [[ -d "$source_repo/language/bert" && -d "$source_repo/loadgen" ]] || {
    echo "missing fresh repository beside reproduce.sh: $source_repo" >&2
    return 1
  }
  [[ -f "$squad_dir/dev-v1.1.json" ]] || { echo "missing dataset" >&2; return 1; }
  [[ -f "$model_dir/model.pytorch" && -f "$model_dir/vocab.txt" ]] || {
    echo "missing model inputs" >&2
    return 1
  }
  [[ -f "$dle_dir/PyTorch/LanguageModeling/BERT/tokenization.py" ]] || {
    echo "missing pinned DeepLearningExamples input" >&2
    return 1
  }

  case "$target_dir/work" in "$script_dir/$target/"*) ;; *) echo "unsafe work path" >&2; return 1 ;; esac
  rm -rf "$target_dir/work"
  mkdir -p "$target_dir/work"
  cp -a "$source_repo" "$work_repo"
  rmdir "$bert_dir/DeepLearningExamples"
  ln -s "$dle_dir" "$bert_dir/DeepLearningExamples"
  mkdir -p "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2" "$bert_dir/build/result"
  ln -s "$squad_dir/dev-v1.1.json" "$bert_dir/build/data/dev-v1.1.json"
  ln -s "$model_dir/model.pytorch" "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
  ln -s "$model_dir/vocab.txt" "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
  sed -i '/self.model.load_state_dict(torch.load(model_file), strict=False)/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$bert_dir/pytorch_SUT.py"
  {
    echo "source tree: $source_repo"
    sha256sum "$squad_dir/dev-v1.1.json" "$model_dir/model.pytorch" "$model_dir/vocab.txt"
  } >"$output_dir/prepare.log"
  touch "$state_dir/prepare.ok"
}

stage_environment() {
  case "$venv_dir" in "$script_dir/$target/"*) ;; *) echo "unsafe venv path" >&2; return 1 ;; esac
  rm -rf "$venv_dir"
  {
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv --python 3.11 "$venv_dir"
    uv pip install --python "$python_bin" \
      torch==2.5.1 transformers==4.46.3 numpy==1.26.4 absl-py==2.1.0 \
      six==1.16.0 boto3==1.40.76 setuptools==75.6.0 wheel==0.45.1 pybind11==2.11.1
    uv pip install --python "$python_bin" --no-build-isolation "$work_repo/loadgen"
    log_versions
  } 2>&1 | tee "$output_dir/build.log"
  touch "$state_dir/environment.ok"
}

stage_preprocess() {
  [[ -x "$python_bin" ]] || { echo "environment stage has not succeeded" >&2; return 1; }
  (
    cd "$bert_dir"
    export VOCAB_FILE="$model_dir/vocab.txt"
    export DATASET_FILE="$squad_dir/dev-v1.1.json"
    export PYTHONPATH="$dle_dir/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}"
    export TOKENIZERS_PARALLELISM=false
    "$python_bin" - <<'PY'
from squad_QSL import get_squad_QSL
qsl = get_squad_QSL()
print("preprocessed_features", qsl.count)
PY
  ) 2>&1 | tee "$output_dir/preprocess.log"
  touch "$state_dir/preprocess.ok"
}

stage_run() {
  [[ -f "$bert_dir/eval_features.pickle" ]] || { echo "preprocess stage has not succeeded" >&2; return 1; }
  rm -f "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" \
    "$output_dir/mlperf_log_summary.txt" "$output_dir/mlperf_log_trace.json" \
    "$output_dir/run.log" "$output_dir/gpu-memory.log"

  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits | tee "$output_dir/gpu-memory.log"
  gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ ! "$gpu_free_mib" =~ ^[0-9]+$ || "$gpu_free_mib" -lt 6000 ]]; then
    echo "insufficient free GPU memory: ${gpu_free_mib:-unknown} MiB; need at least 6000 MiB" | tee -a "$output_dir/gpu-memory.log" >&2
    return 1
  fi

  run_args=(run.py --backend=pytorch --accuracy)
  if [[ -n "$max_examples" ]]; then run_args+=(--max_examples "$max_examples"); fi
  (
    cd "$bert_dir"
    export VOCAB_FILE="$model_dir/vocab.txt"
    export DATASET_FILE="$squad_dir/dev-v1.1.json"
    export ML_MODEL_FILE_WITH_PATH="$model_dir/model.pytorch"
    export LOG_PATH="$output_dir"
    export SKIP_VERIFY_ACCURACY=1
    export TOKENIZERS_PARALLELISM=false
    "$python_bin" "${run_args[@]}"
  ) 2>&1 | tee "$output_dir/run.log"
  touch "$state_dir/run.ok"
}

stage_evaluate() {
  [[ -s "$output_dir/mlperf_log_accuracy.json" && -s "$output_dir/mlperf_log_detail.txt" ]] || {
    echo "run stage outputs are missing" >&2
    return 1
  }
  rm -f "$output_dir/accuracy.txt" "$output_dir/evaluator.log" "$bert_dir/build/result/predictions.json"
  eval_args=(accuracy-squad.py
    --vocab_file "$model_dir/vocab.txt"
    --val_data "$squad_dir/dev-v1.1.json"
    --log_file "$output_dir/mlperf_log_accuracy.json"
    --out_file "$bert_dir/build/result/predictions.json"
    --features_cache_file "$bert_dir/eval_features.pickle")
  if [[ -n "$max_examples" ]]; then eval_args+=(--max_examples "$max_examples"); fi
  (
    cd "$bert_dir"
    export TOKENIZERS_PARALLELISM=false
    "$python_bin" -u "${eval_args[@]}"
  ) 2>&1 | tee "$output_dir/accuracy.txt" "$output_dir/evaluator.log"
  "$python_bin" - "$output_dir/accuracy.txt" <<'PY'
import json, math, pathlib, sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
value = json.loads(lines[-1])
assert set(("exact_match", "f1")) <= value.keys()
assert all(math.isfinite(float(value[k])) for k in ("exact_match", "f1"))
PY
  touch "$state_dir/evaluate.ok"
}

run_one() {
  local stage="$1"
  "stage_$stage"
}

if [[ -n "$single_stage" ]]; then
  run_one "$single_stage"
elif [[ -n "$from_stage" ]]; then
  active=0
  for stage in "${stages[@]}"; do
    [[ "$stage" == "$from_stage" ]] && active=1
    [[ "$active" -eq 1 ]] && run_one "$stage"
  done
else
  for stage in "${stages[@]}"; do run_one "$stage"; done
fi
