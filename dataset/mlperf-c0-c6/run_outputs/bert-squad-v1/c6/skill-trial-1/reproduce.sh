#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

target=""
stage=""
from=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) target=${2:?missing target id}; shift 2 ;;
        --stage) stage=${2:?missing stage name}; shift 2 ;;
        --from) from=${2:?missing stage name}; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$target" != "bert-squad" ]]; then
    echo "usage: $0 --target bert-squad [--stage setup|prepare|run] [--from setup|prepare|run]" >&2
    exit 2
fi
if [[ -n "$stage" && -n "$from" ]]; then
    echo "--stage and --from are mutually exclusive" >&2
    exit 2
fi

root=$(cd "$(dirname "$0")" && pwd)
repo="$root/repo"
target_dir="$root/$target"
work="$target_dir/work"
output_dir=${OUTPUT_DIR:-$target_dir/out}
venv="$work/venv"
markers="$target_dir/markers"
mkdir -p "$output_dir" "$markers"
output_dir=$(cd "$output_dir" && pwd)

for required in \
    "$repo/language/bert/run.py" \
    "$repo/loadgen/setup.py" \
    "$SQUAD_DIR/dev-v1.1.json" \
    "$MODEL_DIR/model.pytorch" \
    "$MODEL_DIR/vocab.txt" \
    "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required input: $required" >&2
        exit 1
    fi
done

clean_from() {
    case "$1" in
        setup)
            rm -f "$markers/setup.success" "$markers/prepare.success" "$markers/run.success"
            rm -rf "$work"
            ;&
        prepare)
            rm -f "$markers/prepare.success" "$markers/run.success"
            rm -rf "$work/bert"
            ;&
        run)
            rm -f "$markers/run.success"
            rm -f "$output_dir/mlperf_log_accuracy.json" \
                  "$output_dir/mlperf_log_detail.txt" \
                  "$output_dir/mlperf_log_summary.txt" \
                  "$output_dir/mlperf_log_trace.json" \
                  "$output_dir/accuracy.txt" \
                  "$output_dir/run.log" \
                  "$output_dir/evaluator.log" \
                  "$output_dir/gpu-before-run.log"
            rm -f "$work/bert/build/result/predictions.json"
            ;;
        *) echo "unknown stage: $1" >&2; exit 2 ;;
    esac
}

run_setup() {
    [[ -f "$markers/setup.success" ]] && return
    mkdir -p "$work"
    : > "$output_dir/build.log"
    {
        echo "Creating Python 3.11 environment"
        uv venv --python python3.11 "$venv"
        uv pip install --python "$venv/bin/python" \
            'torch==2.4.1' 'transformers==4.44.2' 'numpy==1.26.4' \
            'absl-py==2.1.0' 'six==1.16.0' 'setuptools==69.5.1' \
            'pybind11==2.11.1' 'wheel==0.44.0' 'boto3==1.35.54'
        cp -a "$repo/loadgen" "$work/loadgen"
        uv pip install --python "$venv/bin/python" --no-build-isolation "$work/loadgen"
        "$venv/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("mlperf_loadgen import", mlperf_loadgen.__file__)
PY
    } 2>&1 | tee -a "$output_dir/build.log"
    touch "$markers/setup.success"
}

run_prepare() {
    [[ -f "$markers/prepare.success" ]] && return
    mkdir -p "$work"
    cp -a "$repo/language/bert" "$work/bert"
    rm -rf "$work/bert/DeepLearningExamples"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$work/bert/DeepLearningExamples"
    mkdir -p "$work/bert/build/result"
    {
        echo "dataset: $SQUAD_DIR/dev-v1.1.json"
        echo "model: $MODEL_DIR/model.pytorch"
        echo "vocabulary: $MODEL_DIR/vocab.txt"
        sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
        "$venv/bin/python" - <<'PY'
import json
p = "/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache/bert-squad/data/dev-v1.1.json"
d = json.load(open(p))
print("SQuAD version:", d["version"])
print("articles:", len(d["data"]))
print("questions:", sum(len(p["qas"]) for a in d["data"] for p in a["paragraphs"]))
PY
    } 2>&1 | tee "$output_dir/prepare.log"
    touch "$markers/prepare.success"
}

run_benchmark() {
    [[ -f "$markers/run.success" ]] && return
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader | tee "$output_dir/gpu-before-run.log"
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( free_mib < 12000 )); then
        echo "insufficient GPU memory: ${free_mib} MiB free; need at least 12000 MiB" >&2
        exit 1
    fi
    max_args=()
    if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
        max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
    fi
    cd "$work/bert"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export LOG_PATH="$output_dir"
    export SKIP_VERIFY_ACCURACY=1
    {
        "$venv/bin/python" - <<'PY'
import torch
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("pytorch device: ", device, sep="")
if device.type != "cuda":
    raise SystemExit("CUDA is required for this target")
print("pytorch gpu:", torch.cuda.get_device_name(0))
PY
        "$venv/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
    } 2>&1 | tee "$output_dir/run.log"
    "$venv/bin/python" accuracy-squad.py \
        --vocab_file "$MODEL_DIR/vocab.txt" \
        --val_data "$SQUAD_DIR/dev-v1.1.json" \
        --log_file "$output_dir/mlperf_log_accuracy.json" \
        --out_file "$work/bert/build/result/predictions.json" \
        --features_cache_file "$work/bert/eval_features.pickle" \
        "${max_args[@]}" 2>&1 | tee "$output_dir/accuracy.txt" "$output_dir/evaluator.log"
    test -s "$output_dir/mlperf_log_accuracy.json"
    test -s "$output_dir/mlperf_log_detail.txt"
    test -s "$output_dir/accuracy.txt"
    touch "$markers/run.success"
}

if [[ -n "$from" ]]; then
    clean_from "$from"
fi

if [[ -n "$stage" ]]; then
    case "$stage" in
        setup) run_setup ;;
        prepare) run_prepare ;;
        run) run_benchmark ;;
    esac
else
    case "${from:-setup}" in
        setup) run_setup; run_prepare; run_benchmark ;;
        prepare) run_prepare; run_benchmark ;;
        run) run_benchmark ;;
        *) echo "unknown stage: $from" >&2; exit 2 ;;
    esac
fi
