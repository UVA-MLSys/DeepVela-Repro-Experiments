#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target <target-id>" >&2
  exit 2
}

target=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || usage
      target="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done
[[ "$target" == "bert-squad" ]] || usage

root=$(cd "$(dirname "$0")" && pwd)
repo="$root/repo"
target_dir="$root/$target"
work_dir="$target_dir/work"
out_dir=${OUTPUT_DIR:-$target_dir/out}
cache_root=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache

export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$target_dir" "$out_dir"
rm -rf "$work_dir"
mkdir -p "$work_dir"
rm -f \
  "$out_dir/mlperf_log_accuracy.json" \
  "$out_dir/mlperf_log_detail.txt" \
  "$out_dir/mlperf_log_summary.txt" \
  "$out_dir/mlperf_log_trace.json" \
  "$out_dir/accuracy.txt"

cp -a "$repo/language/bert" "$work_dir/bert"
rm -rf "$work_dir/bert/DeepLearningExamples"
ln -s "$cache_root/bert-squad/DeepLearningExamples" \
  "$work_dir/bert/DeepLearningExamples"

# Emit the target-required device line immediately after loading the weights.
perl -0pi -e \
  's/(self\.model\.load_state_dict\(torch\.load\(model_file\), strict=False\)\n)/$1        print("pytorch device: ", next(self.model.parameters()).device)\n/' \
  "$work_dir/bert/pytorch_SUT.py"

{
  echo "Creating Python environment"
  uv venv --python 3.11 "$work_dir/venv"
  uv pip install --offline --python "$work_dir/venv/bin/python" \
    'torch==2.7.1' 'transformers==4.40.2' 'numpy<2' six absl-py \
    'setuptools==80.9.0' boto3 pybind11
  uv pip install --offline --python "$work_dir/venv/bin/python" \
    --no-build-isolation "$repo/loadgen"
  "$work_dir/venv/bin/python" - <<'PY'
import mlperf_loadgen
import numpy
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("loadgen import ok")
PY
} >"$target_dir/build.log" 2>&1

# The A100 is shared, so inspect its current availability immediately before
# launching the only GPU workload.
free_mib=$(nvidia-smi --query-gpu=memory.free \
  --format=csv,noheader,nounits -i 0 | tr -d ' ')
{
  echo "GPU free memory before run: ${free_mib} MiB"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits -i 0
} >"$target_dir/gpu-check.log"
if [[ "$free_mib" -lt 8000 ]]; then
  echo "Insufficient free GPU memory: ${free_mib} MiB" \
    | tee -a "$target_dir/gpu-check.log" >&2
  exit 1
fi

export ML_MODEL_FILE_WITH_PATH="$cache_root/bert-squad/model/model.pytorch"
export VOCAB_FILE="$cache_root/bert-squad/model/vocab.txt"
export DATASET_FILE="$cache_root/bert-squad/data/dev-v1.1.json"
export LOG_PATH="$out_dir"
export SKIP_VERIFY_ACCURACY=1
export CUDA_VISIBLE_DEVICES=0

cd "$work_dir/bert"
"$work_dir/venv/bin/python" run.py --backend=pytorch --accuracy \
  >"$target_dir/run.log" 2>&1

mkdir -p "$work_dir/result"
"$work_dir/venv/bin/python" accuracy-squad.py \
  --vocab_file "$VOCAB_FILE" \
  --val_data "$DATASET_FILE" \
  --log_file "$out_dir/mlperf_log_accuracy.json" \
  --out_file "$work_dir/result/predictions.json" \
  --features_cache_file "$work_dir/bert/eval_features.pickle" \
  2>&1 | tee "$target_dir/evaluator.log" "$out_dir/accuracy.txt"

test -s "$out_dir/mlperf_log_accuracy.json"
test -s "$out_dir/mlperf_log_detail.txt"
test -s "$out_dir/accuracy.txt"
