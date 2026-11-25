#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target <target-id>" >&2
    exit 2
}

target=""
while (($#)); do
    case "$1" in
        --target)
            (($# >= 2)) || usage
            target="$2"
            shift 2
            ;;
        *) usage ;;
    esac
done

[[ "$target" == "bert-squad" ]] || usage
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_dir="$root_dir/repo"
target_dir="$root_dir/$target"
work_dir="$target_dir/work"
bert_dir="$work_dir/bert"
venv_dir="$target_dir/venv"

cache_root=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
squad_dir="$cache_root/bert-squad/data"
model_dir="$cache_root/bert-squad/model"
dle_dir="$cache_root/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$target_dir" "$work_dir" "$OUTPUT_DIR"
: >"$target_dir/build.log"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    uv venv --python 3.12 "$venv_dir" >>"$target_dir/build.log" 2>&1
fi

# Install all runtime requirements and build the pinned LoadGen source in a
# private copy, leaving the supplied repository untouched.
uv pip install --python "$venv_dir/bin/python" \
    'setuptools<81' 'numpy<2' 'torch==2.7.1' 'transformers==4.40.2' \
    absl-py boto3 six pybind11==2.11.1 >>"$target_dir/build.log" 2>&1

if [[ ! -f "$work_dir/loadgen/.staged" ]]; then
    mkdir -p "$work_dir/loadgen"
    cp -a "$repo_dir/loadgen/." "$work_dir/loadgen/"
    touch "$work_dir/loadgen/.staged"
fi
uv pip install --python "$venv_dir/bin/python" --no-build-isolation \
    "$work_dir/loadgen" >>"$target_dir/build.log" 2>&1

if [[ ! -f "$bert_dir/.staged" ]]; then
    mkdir -p "$bert_dir"
    cp -a "$repo_dir/language/bert/." "$bert_dir/"
    # The target requires this diagnostic. Add it only to the private copy.
    sed -i '/self.model.load_state_dict/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
        "$bert_dir/pytorch_SUT.py"
    touch "$bert_dir/.staged"
fi
# The pinned tree contains an empty submodule directory. Replace that private
# empty directory with the separately staged, read-only helper checkout.
if [[ ! -L "$bert_dir/DeepLearningExamples" ]]; then
    rmdir "$bert_dir/DeepLearningExamples"
    ln -s "$dle_dir" "$bert_dir/DeepLearningExamples"
fi

export DATASET_FILE="$squad_dir/dev-v1.1.json"
export VOCAB_FILE="$model_dir/vocab.txt"
export ML_MODEL_FILE_WITH_PATH="$model_dir/model.pytorch"
export LOG_PATH="$OUTPUT_DIR"
export SKIP_VERIFY_ACCURACY=1

# This is intentionally immediately before the sole GPU run.
{
    date -Is
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits
} | tee "$target_dir/gpu_memory.log"

(
    cd "$bert_dir"
    "$venv_dir/bin/python" run.py --backend=pytorch --accuracy
) 2>&1 | tee "$target_dir/run.log"

mkdir -p "$work_dir/result"
(
    cd "$bert_dir"
    "$venv_dir/bin/python" accuracy-squad.py \
        --vocab_file "$model_dir/vocab.txt" \
        --val_data "$squad_dir/dev-v1.1.json" \
        --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        --out_file "$work_dir/result/predictions.json" \
        --features_cache_file "$bert_dir/eval_features.pickle"
) 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$target_dir/evaluator.log"

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
