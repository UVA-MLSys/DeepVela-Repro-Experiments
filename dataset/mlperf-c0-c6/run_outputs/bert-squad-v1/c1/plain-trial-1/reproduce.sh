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

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work_dir="$root_dir/$target"
output_dir="${OUTPUT_DIR:-$work_dir/out}"
cache_root="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
squad_dir="${SQUAD_DIR:-$cache_root/bert-squad/data}"
model_dir="${MODEL_DIR:-$cache_root/bert-squad/model}"
dle_dir="${DEEPLEARNING_EXAMPLES_DIR:-$cache_root/bert-squad/DeepLearningExamples}"
uv_cache_dir="${UV_CACHE_DIR:-$cache_root/uv}"
export UV_CACHE_DIR="$uv_cache_dir"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$work_dir" "$output_dir"
exec > >(tee "$work_dir/reproduce.log") 2>&1

for required in \
    "$root_dir/repo/language/bert/run.py" \
    "$root_dir/repo/loadgen/setup.py" \
    "$squad_dir/dev-v1.1.json" \
    "$model_dir/model.pytorch" \
    "$model_dir/vocab.txt" \
    "$dle_dir/PyTorch/LanguageModeling/BERT/tokenization.py"; do
    [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

if [[ ! -d "$work_dir/src" ]]; then
    cp -a "$root_dir/repo" "$work_dir/src"
fi
bert_dir="$work_dir/src/language/bert"

# The target requires this diagnostic prefix, while the pinned harness selects
# the device silently. Patch only the disposable source copy.
if ! grep -q 'pytorch device: ' "$bert_dir/pytorch_SUT.py"; then
    sed -i '/self.model.to(self.dev)/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
        "$bert_dir/pytorch_SUT.py"
fi
mkdir -p "$bert_dir/DeepLearningExamples"
ln -sfn "$dle_dir/PyTorch" "$bert_dir/DeepLearningExamples/PyTorch"

if [[ ! -x "$work_dir/.venv/bin/python" ]]; then
    managed_python="$(uv python find --managed-python 3.11)"
    uv venv --python "$managed_python" "$work_dir/.venv"
fi

uv pip install --python "$work_dir/.venv/bin/python" \
    'numpy==1.26.4' 'torch==2.4.1' 'transformers==4.44.2' \
    'absl-py==2.1.0' 'six==1.16.0' 'setuptools==75.1.0' \
    'pybind11==2.13.6' 'boto3==1.35.24' \
    2>&1 | tee "$work_dir/build-packages.log"
if ! "$work_dir/.venv/bin/python" -c 'import mlperf_loadgen' 2>/dev/null; then
    uv pip install --python "$work_dir/.venv/bin/python" "$work_dir/src/loadgen" \
        2>&1 | tee "$work_dir/build-loadgen.log"
fi

rm -f \
    "$output_dir/mlperf_log_accuracy.json" \
    "$output_dir/mlperf_log_detail.txt" \
    "$output_dir/mlperf_log_summary.txt" \
    "$output_dir/mlperf_log_trace.json" \
    "$output_dir/accuracy.txt"
mkdir -p "$bert_dir/build/result"

export PATH="$work_dir/.venv/bin:$PATH"
export ML_MODEL_FILE_WITH_PATH="$model_dir/model.pytorch"
export VOCAB_FILE="$model_dir/vocab.txt"
export DATASET_FILE="$squad_dir/dev-v1.1.json"
export LOG_PATH="$output_dir"
export SKIP_VERIFY_ACCURACY=1
export PYTHONUNBUFFERED=1

cd "$bert_dir"
echo "GPU memory immediately before run:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader | tee "$work_dir/gpu-before-run.log"

python3 run.py --backend=pytorch --accuracy \
    2>&1 | tee "$work_dir/run.log"

python3 accuracy-squad.py \
    --vocab_file "$model_dir/vocab.txt" \
    --val_data "$squad_dir/dev-v1.1.json" \
    --log_file "$output_dir/mlperf_log_accuracy.json" \
    --out_file "$bert_dir/build/result/predictions.json" \
    --features_cache_file "$bert_dir/eval_features.pickle" \
    2>&1 | tee "$output_dir/accuracy.txt" "$work_dir/evaluator.log"

test -s "$output_dir/mlperf_log_accuracy.json"
test -s "$output_dir/mlperf_log_detail.txt"
test -s "$output_dir/accuracy.txt"
