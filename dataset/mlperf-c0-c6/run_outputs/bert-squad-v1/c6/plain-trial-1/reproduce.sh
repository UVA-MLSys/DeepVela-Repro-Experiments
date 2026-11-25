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

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir="$root_dir/repo"
target_dir="$root_dir/$target"
work_dir="$target_dir/work"
output_dir=${OUTPUT_DIR:-"$target_dir/out"}

cache_root=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
squad_dir="$cache_root/bert-squad/data"
model_dir="$cache_root/bert-squad/model"
dle_dir="$cache_root/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$target_dir" "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
: > "$target_dir/build.log"
: > "$target_dir/run.log"
: > "$target_dir/evaluator.log"

for required in \
    "$repo_dir/language/bert/run.py" \
    "$squad_dir/dev-v1.1.json" \
    "$model_dir/model.pytorch" \
    "$model_dir/vocab.txt" \
    "$dle_dir/PyTorch/LanguageModeling/BERT/tokenization.py"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required input: $required" | tee -a "$target_dir/build.log" >&2
        exit 1
    fi
done

# Recreate all derived state so one invocation is sufficient on a fresh tree.
if [[ -e "$work_dir" ]]; then
    stale_dir="$target_dir/work.stale.$$"
    mv "$work_dir" "$stale_dir"
    rm -rf "$stale_dir"
fi
mkdir -p "$work_dir"
cp -a "$repo_dir" "$work_dir/repo"

{
    echo "Creating Python environment"
    uv venv --python /home/mi3se/.local/bin/python3.11 "$work_dir/venv"
    uv pip install --python "$work_dir/venv/bin/python" \
        'torch==2.4.1' \
        'transformers==4.45.2' \
        'numpy==1.26.4' \
        'absl-py==2.1.0' \
        'boto3==1.35.54' \
        'six==1.16.0' \
        'pybind11==2.11.1' \
        'setuptools==75.3.0' \
        'wheel==0.44.0'
    uv pip install --python "$work_dir/venv/bin/python" \
        --no-build-isolation "$work_dir/repo/loadgen"
} 2>&1 | tee -a "$target_dir/build.log"

bert_dir="$work_dir/repo/language/bert"
mkdir -p "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2" \
    "$bert_dir/build/logs" "$bert_dir/build/result"
ln -s "$squad_dir/dev-v1.1.json" "$bert_dir/build/data/dev-v1.1.json"
ln -s "$model_dir/model.pytorch" \
    "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
ln -s "$model_dir/vocab.txt" \
    "$bert_dir/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
ln -s "$dle_dir" "$bert_dir/DeepLearningExamples"

export DATASET_FILE="$squad_dir/dev-v1.1.json"
export VOCAB_FILE="$model_dir/vocab.txt"
export ML_MODEL_FILE_WITH_PATH="$model_dir/model.pytorch"
export LOG_PATH="$bert_dir/build/logs"
export SKIP_VERIFY_ACCURACY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$dle_dir/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}"

cd "$bert_dir"

# Check current GPU capacity immediately before launching the GPU process.
{
    echo "GPU state immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
    "$work_dir/venv/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
print(f"pytorch device: {torch.device('cuda:0')}")
print(f"torch CUDA memory free: {free_bytes // 2**20} MiB / {total_bytes // 2**20} MiB")
if free_bytes < 8 * 2**30:
    raise SystemExit("less than 8 GiB GPU memory is free")
PY
    "$work_dir/venv/bin/python" run.py --backend=pytorch --accuracy
} 2>&1 | tee -a "$target_dir/run.log"

cp "$bert_dir/build/logs/mlperf_log_accuracy.json" \
    "$output_dir/mlperf_log_accuracy.json"
cp "$bert_dir/build/logs/mlperf_log_detail.txt" \
    "$output_dir/mlperf_log_detail.txt"

"$work_dir/venv/bin/python" accuracy-squad.py \
    --vocab_file "$model_dir/vocab.txt" \
    --val_data "$squad_dir/dev-v1.1.json" \
    --log_file "$output_dir/mlperf_log_accuracy.json" \
    --out_file "$bert_dir/build/result/predictions.json" \
    --features_cache_file "$bert_dir/eval_features.pickle" \
    2> "$target_dir/evaluator-stderr.log" \
    | tee "$target_dir/evaluator.log" "$output_dir/accuracy.txt"

for expected in mlperf_log_accuracy.json mlperf_log_detail.txt accuracy.txt; do
    [[ -s "$output_dir/$expected" ]] || {
        echo "missing or empty output: $output_dir/$expected" >&2
        exit 1
    }
done
