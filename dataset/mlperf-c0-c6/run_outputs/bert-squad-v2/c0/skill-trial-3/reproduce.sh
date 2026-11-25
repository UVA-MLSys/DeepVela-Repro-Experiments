#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=""
ONLY_STAGE=""
FROM_STAGE=""
STAGES=(setup preprocess preflight run evaluate)

usage() {
    echo "usage: $0 --target bert-squad [--stage setup|preprocess|preflight|run|evaluate] [--from setup|preprocess|preflight|run|evaluate]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) [[ $# -ge 2 ]] || usage; TARGET=$2; shift 2 ;;
        --stage) [[ $# -ge 2 ]] || usage; ONLY_STAGE=$2; shift 2 ;;
        --from) [[ $# -ge 2 ]] || usage; FROM_STAGE=$2; shift 2 ;;
        *) usage ;;
    esac
done

[[ "$TARGET" == bert-squad ]] || usage
[[ -z "$ONLY_STAGE" || -z "$FROM_STAGE" ]] || usage
valid_stage() {
    local candidate=$1 stage
    [[ -z "$candidate" ]] && return 0
    for stage in "${STAGES[@]}"; do [[ "$candidate" != "$stage" ]] || return 0; done
    return 1
}
valid_stage "$ONLY_STAGE" || usage
valid_stage "$FROM_STAGE" || usage
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "OUTPUT_DIR must be set" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

WORK_DIR="$ROOT_DIR/$TARGET"
SOURCE_REPO="$ROOT_DIR/repo"
WORK_REPO="$WORK_DIR/repo"
VENV="$WORK_DIR/.venv"
BERT_DIR="$WORK_REPO/language/bert"
mkdir -p "$WORK_DIR"

check_inputs() {
    [[ -f "$SOURCE_REPO/language/bert/run.py" ]] || { echo "fresh repository missing at $SOURCE_REPO" >&2; exit 1; }
    [[ -f "$SQUAD_DIR/dev-v1.1.json" ]] || { echo "missing SQuAD input" >&2; exit 1; }
    [[ -f "$MODEL_DIR/model.pytorch" && -f "$MODEL_DIR/vocab.txt" ]] || { echo "missing model input" >&2; exit 1; }
    [[ -f "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py" ]] || { echo "missing DeepLearningExamples input" >&2; exit 1; }
}

clear_from() {
    local requested=$1 clear=0 stage
    for stage in "${STAGES[@]}"; do
        [[ "$stage" != "$requested" ]] || clear=1
        (( clear == 0 )) && continue
        rm -f "$WORK_DIR/.${stage}.success"
        case "$stage" in
            setup)
                rm -rf "$VENV" "$WORK_REPO"
                rm -f "$OUTPUT_DIR/setup.log"
                ;;
            preprocess)
                rm -f "$OUTPUT_DIR/preprocess.log"
                [[ ! -d "$BERT_DIR" ]] || rm -f "$BERT_DIR/eval_features.pickle"
                ;;
            preflight)
                rm -rf "$OUTPUT_DIR/preflight"
                rm -f "$WORK_DIR/predictions-preflight.json"
                ;;
            run)
                rm -f "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/mlperf_log_accuracy.json" \
                    "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" \
                    "$OUTPUT_DIR/mlperf_log_trace.json"
                ;;
            evaluate)
                rm -f "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt" "$WORK_DIR/predictions.json"
                ;;
        esac
    done
}

gpu_check() {
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    echo "GPU memory free before run: ${free_mib} MiB"
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 8192 )) || {
        echo "At least 8192 MiB of free GPU memory is required" >&2
        return 1
    }
}

setup_stage() {
    [[ ! -f "$WORK_DIR/.setup.success" ]] || return 0
    check_inputs
    rm -rf "$VENV" "$WORK_REPO"
    {
        echo "Staging repository and fixed inputs"
        cp -a "$SOURCE_REPO" "$WORK_REPO"
        rmdir "$BERT_DIR/DeepLearningExamples"
        ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"
        python3 - "$BERT_DIR/pytorch_SUT.py" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
needle = "        self.model.load_state_dict(torch.load(model_file), strict=False)\n"
addition = needle + '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n'
if "pytorch device: " not in text:
    if needle not in text:
        raise SystemExit("could not locate model load in pytorch_SUT.py")
    path.write_text(text.replace(needle, addition, 1))

accuracy = path.parent / "accuracy-squad.py"
acc_text = accuracy.read_text().replace("import pkg_resources\n", "")
old = 'installed = {pkg.key for pkg in pkg_resources.working_set}\nif "tensorflow" in installed:'
new = 'try:\n    import tensorflow  # noqa: F401\nexcept ImportError:\n    tensorflow = None\nif tensorflow is not None:'
if old in acc_text:
    acc_text = acc_text.replace(old, new, 1)
acc_text = acc_text.replace('elif "torch" in installed:\n    import torch', 'else:\n    import torch', 1)
accuracy.write_text(acc_text)
PY
        uv venv --python 3.11 "$VENV"
        uv pip install --python "$VENV/bin/python" \
            "numpy==1.26.4" "torch==2.5.1" "transformers==4.44.2" \
            absl-py boto3 six "pybind11==2.11.1" "setuptools<81" wheel
        uv pip install --python "$VENV/bin/python" --no-build-isolation "$WORK_REPO/loadgen"
        "$VENV/bin/python" - <<'PY'
import mlperf_loadgen
import numpy
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "numpy", numpy.__version__)
assert torch.cuda.is_available()
PY
    } 2>&1 | tee "$OUTPUT_DIR/setup.log"
    touch "$WORK_DIR/.setup.success"
}

preprocess_stage() {
    [[ -f "$WORK_DIR/.setup.success" ]] || setup_stage
    [[ ! -f "$WORK_DIR/.preprocess.success" ]] || return 0
    (
        cd "$BERT_DIR"
        export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
        export VOCAB_FILE="$MODEL_DIR/vocab.txt"
        "$VENV/bin/python" - <<'PY'
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
from squad_QSL import get_squad_QSL
qsl = get_squad_QSL()
print("preprocessed features:", qsl.count)
PY
    ) 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
    touch "$WORK_DIR/.preprocess.success"
}

benchmark_run() {
    local max_examples=$1 log_dir=$2 run_log=$3
    mkdir -p "$log_dir"
    gpu_check | tee -a "$run_log"
    (
        cd "$BERT_DIR"
        export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
        export VOCAB_FILE="$MODEL_DIR/vocab.txt"
        export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
        export LOG_PATH="$log_dir"
        export SKIP_VERIFY_ACCURACY=1
        export CUDA_VISIBLE_DEVICES=0
        args=(--backend=pytorch --accuracy)
        [[ -z "$max_examples" ]] || args+=(--max_examples "$max_examples")
        "$VENV/bin/python" run.py "${args[@]}"
    ) 2>&1 | tee -a "$run_log"
}

evaluator_run() {
    local max_examples=$1 log_dir=$2 out_file=$3 eval_log=$4 accuracy_out=$5
    (
        cd "$BERT_DIR"
        args=(--vocab_file "$MODEL_DIR/vocab.txt" --val_data "$SQUAD_DIR/dev-v1.1.json"
              --log_file "$log_dir/mlperf_log_accuracy.json" --out_file "$out_file"
              --features_cache_file "$BERT_DIR/eval_features.pickle")
        [[ -z "$max_examples" ]] || args+=(--max_examples "$max_examples")
        "$VENV/bin/python" accuracy-squad.py "${args[@]}"
    ) 2>&1 | tee "$accuracy_out" "$eval_log"
}

preflight_stage() {
    [[ -f "$WORK_DIR/.preprocess.success" ]] || preprocess_stage
    [[ ! -f "$WORK_DIR/.preflight.success" ]] || return 0
    local preflight_dir="$OUTPUT_DIR/preflight"
    benchmark_run 8 "$preflight_dir" "$preflight_dir/run.log"
    evaluator_run 8 "$preflight_dir" "$WORK_DIR/predictions-preflight.json" \
        "$preflight_dir/evaluator.log" "$preflight_dir/accuracy.txt"
    touch "$WORK_DIR/.preflight.success"
}

run_stage() {
    [[ -f "$WORK_DIR/.preprocess.success" ]] || preprocess_stage
    [[ ! -f "$WORK_DIR/.run.success" ]] || return 0
    benchmark_run "" "$OUTPUT_DIR" "$OUTPUT_DIR/run.log"
    touch "$WORK_DIR/.run.success"
}

evaluate_stage() {
    [[ -f "$WORK_DIR/.run.success" ]] || run_stage
    [[ ! -f "$WORK_DIR/.evaluate.success" ]] || return 0
    evaluator_run "" "$OUTPUT_DIR" "$WORK_DIR/predictions.json" \
        "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
    touch "$WORK_DIR/.evaluate.success"
}

[[ -z "$FROM_STAGE" ]] || clear_from "$FROM_STAGE"

run_stage_by_name() {
    case "$1" in
        setup) setup_stage ;;
        preprocess) preprocess_stage ;;
        preflight) preflight_stage ;;
        run) run_stage ;;
        evaluate) evaluate_stage ;;
    esac
}

if [[ -n "$ONLY_STAGE" ]]; then
    run_stage_by_name "$ONLY_STAGE"
elif [[ -n "$FROM_STAGE" ]]; then
    active=0
    for stage in "${STAGES[@]}"; do
        [[ "$stage" != "$FROM_STAGE" ]] || active=1
        (( active == 0 )) || run_stage_by_name "$stage"
    done
else
    for stage in "${STAGES[@]}"; do run_stage_by_name "$stage"; done
fi
