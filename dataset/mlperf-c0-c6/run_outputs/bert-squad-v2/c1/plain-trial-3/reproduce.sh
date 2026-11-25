#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target bert-squad" >&2
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_bert_dir="$script_dir/repo/language/bert"
work_dir="$script_dir/$target"
bert_dir="$work_dir/bert"
output_dir="${OUTPUT_DIR:-$work_dir/out}"
mkdir -p "$work_dir" "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

cache_root=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER=2026-11-10
export UV_LINK_MODE=copy

build_log="$work_dir/build.log"
run_log="$work_dir/run.log"
evaluator_log="$work_dir/evaluator.log"
: >"$build_log"
: >"$run_log"
: >"$evaluator_log"

venv="$work_dir/.venv"
if [[ ! -x "$venv/bin/python" ]]; then
    uv venv --python /usr/bin/python3.11 "$venv" >>"$build_log" 2>&1
fi

uv pip install --python "$venv/bin/python" \
    torch==2.4.1 >>"$build_log" 2>&1
uv pip install --python "$venv/bin/python" \
    numpy==1.26.4 transformers==4.45.2 absl-py==2.1.0 six==1.16.0 \
    setuptools==75.3.0 \
    boto3==1.35.54 mlcommons-loadgen==6.0.17 \
    >>"$build_log" 2>&1

# Run from a private copy so generated feature caches never alter the input tree.
if [[ ! -d "$bert_dir" ]]; then
    cp -a "$source_bert_dir" "$bert_dir"
fi
dle_src="$cache_root/bert-squad/DeepLearningExamples"
if [[ -d "$bert_dir/DeepLearningExamples" && ! -L "$bert_dir/DeepLearningExamples" ]]; then
    rmdir "$bert_dir/DeepLearningExamples"
fi
if [[ ! -e "$bert_dir/DeepLearningExamples" ]]; then
    ln -s "$dle_src" "$bert_dir/DeepLearningExamples"
fi

export DATASET_FILE="$cache_root/bert-squad/data/dev-v1.1.json"
export VOCAB_FILE="$cache_root/bert-squad/model/vocab.txt"
export ML_MODEL_FILE_WITH_PATH="$cache_root/bert-squad/model/model.pytorch"
export LOG_PATH="$output_dir"
export SKIP_VERIFY_ACCURACY=1
export TOKENIZERS_PARALLELISM=false

# Check immediately before the sole GPU process and retain the evidence.
gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
echo "GPU free memory before run: ${gpu_free_mib} MiB" | tee -a "$run_log"
if [[ ! "$gpu_free_mib" =~ ^[0-9]+$ ]] || (( gpu_free_mib < 8192 )); then
    echo "insufficient free GPU memory: need at least 8192 MiB" | tee -a "$run_log" >&2
    exit 1
fi

device="$($venv/bin/python -c 'import torch; print("cuda:0" if torch.cuda.is_available() else "cpu")')"
echo "pytorch device: $device" | tee -a "$run_log"
if [[ "$device" != "cuda:0" ]]; then
    echo "CUDA is unavailable to PyTorch" | tee -a "$run_log" >&2
    exit 1
fi

(
    cd "$bert_dir"
    "$venv/bin/python" run.py --backend=pytorch --accuracy
) >>"$run_log" 2>&1

prediction_file="$work_dir/predictions.json"
(
    cd "$bert_dir"
    "$venv/bin/python" accuracy-squad.py \
        --vocab_file "$VOCAB_FILE" \
        --val_data "$DATASET_FILE" \
        --log_file "$output_dir/mlperf_log_accuracy.json" \
        --out_file "$prediction_file" \
        --features_cache_file "$bert_dir/eval_features.pickle"
) 2>>"$evaluator_log" | tee "$output_dir/accuracy.txt" | tee -a "$evaluator_log"

test -s "$output_dir/mlperf_log_accuracy.json"
test -s "$output_dir/mlperf_log_detail.txt"
test -s "$output_dir/accuracy.txt"
