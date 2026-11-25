#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="${CACHE_ROOT}/bert-squad/data"
MODEL_DIR="${CACHE_ROOT}/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="${CACHE_ROOT}/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_EXCLUDE_NEWER=2026-11-10

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="${SCRIPT_DIR}/repo"
TARGET=""
REQUESTED_STAGE=""
FROM_STAGE=""

usage() {
    echo "Usage: $0 --target bert-squad [--stage environment|inputs|run|evaluate] [--from environment|inputs|run|evaluate]" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET=${2-}; shift 2 ;;
        --stage) REQUESTED_STAGE=${2-}; shift 2 ;;
        --from) FROM_STAGE=${2-}; shift 2 ;;
        *) usage ;;
    esac
done

[[ "${TARGET}" == "bert-squad" ]] || usage
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "OUTPUT_DIR must be set" >&2; exit 2; }
mkdir -p -- "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd -- "${OUTPUT_DIR}" && pwd)
[[ "${OUTPUT_DIR}" != "/" ]] || { echo "Refusing OUTPUT_DIR=/" >&2; exit 2; }
[[ -d "${REPO_DIR}/language/bert" && -d "${REPO_DIR}/loadgen" ]] || {
    echo "Expected repo/ beside reproduce.sh" >&2
    exit 2
}

WORK_DIR="${OUTPUT_DIR}/work"
VENV_DIR="${WORK_DIR}/venv"
BERT_WORK="${WORK_DIR}/bert"
LOADGEN_WORK="${WORK_DIR}/loadgen"
STAGES=(environment inputs run evaluate)

valid_stage() {
    local candidate=$1 stage
    for stage in "${STAGES[@]}"; do
        [[ "${candidate}" == "${stage}" ]] && return 0
    done
    return 1
}

[[ -z "${REQUESTED_STAGE}" ]] || valid_stage "${REQUESTED_STAGE}" || usage
[[ -z "${FROM_STAGE}" ]] || valid_stage "${FROM_STAGE}" || usage
[[ -z "${REQUESTED_STAGE}" || -z "${FROM_STAGE}" ]] || usage

clear_from() {
    local start=$1 active=0 stage
    for stage in "${STAGES[@]}"; do
        [[ "${stage}" == "${start}" ]] && active=1
        if [[ ${active} -eq 1 ]]; then
            rm -f -- "${OUTPUT_DIR}/.${stage}.success"
        fi
    done
    case "${start}" in
        environment)
            rm -rf -- "${WORK_DIR}"
            rm -f -- "${OUTPUT_DIR}/environment.log" "${OUTPUT_DIR}/inputs.log"
            ;;
        inputs)
            rm -rf -- "${BERT_WORK}"
            rm -f -- "${OUTPUT_DIR}/inputs.log"
            ;;
    esac
    if [[ "${start}" == "environment" || "${start}" == "inputs" || "${start}" == "run" ]]; then
        rm -f -- "${OUTPUT_DIR}/run.log" "${OUTPUT_DIR}/mlperf_log_accuracy.json" \
            "${OUTPUT_DIR}/mlperf_log_detail.txt" "${OUTPUT_DIR}/mlperf_log_summary.txt" \
            "${OUTPUT_DIR}/mlperf_log_trace.json"
    fi
    if [[ "${start}" != "evaluate" ]]; then
        rm -f -- "${OUTPUT_DIR}/accuracy.log" "${OUTPUT_DIR}/accuracy.txt" \
            "${WORK_DIR}/predictions.json"
    else
        rm -f -- "${OUTPUT_DIR}/accuracy.log" "${OUTPUT_DIR}/accuracy.txt" \
            "${WORK_DIR}/predictions.json"
    fi
}

stage_environment() {
    mkdir -p -- "${WORK_DIR}"
    {
        echo "Creating Python 3.11 environment"
        uv venv --python 3.11 "${VENV_DIR}"
        uv pip install --python "${VENV_DIR}/bin/python" \
            --index-url https://download.pytorch.org/whl/cu118 \
            --no-deps 'torch==2.1.2+cu118'
        uv pip install --python "${VENV_DIR}/bin/python" \
            'numpy==1.26.4' 'transformers==4.31.0' 'absl-py==2.1.0' \
            'six==1.16.0' 'setuptools==70.3.0' 'pybind11==2.13.6' \
            'filelock==3.13.1' 'typing-extensions==4.12.2' 'sympy==1.12' \
            'networkx==3.2.1' 'jinja2==3.1.4' 'boto3==1.35.54'
        cp -a -- "${REPO_DIR}/loadgen" "${LOADGEN_WORK}"
        uv pip install --python "${VENV_DIR}/bin/python" --no-build-isolation "${LOADGEN_WORK}"
        "${VENV_DIR}/bin/python" - <<'PY'
import mlperf_loadgen
import numpy
import torch
import transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("loadgen import: ok")
PY
    } 2>&1 | tee "${OUTPUT_DIR}/environment.log"
    touch "${OUTPUT_DIR}/.environment.success"
}

stage_inputs() {
    [[ -f "${OUTPUT_DIR}/.environment.success" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
    {
        test -f "${SQUAD_DIR}/dev-v1.1.json"
        test -f "${MODEL_DIR}/model.pytorch"
        test -f "${MODEL_DIR}/vocab.txt"
        test -f "${DEEPLEARNING_EXAMPLES_DIR}/PyTorch/LanguageModeling/BERT/tokenization.py"
        sha256sum "${SQUAD_DIR}/dev-v1.1.json" "${MODEL_DIR}/model.pytorch" "${MODEL_DIR}/vocab.txt"
        cp -a -- "${REPO_DIR}/language/bert" "${BERT_WORK}"
        rmdir -- "${BERT_WORK}/DeepLearningExamples" 2>/dev/null || true
        ln -s -- "${DEEPLEARNING_EXAMPLES_DIR}" "${BERT_WORK}/DeepLearningExamples"
        # The target contract requires this diagnostic prefix. Report the actual
        # device of the loaded model without changing any inference operation.
        sed -i '/self.model.load_state_dict/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
            "${BERT_WORK}/pytorch_SUT.py"
        echo "Prepared BERT source and fixed read-only inputs"
    } 2>&1 | tee "${OUTPUT_DIR}/inputs.log"
    touch "${OUTPUT_DIR}/.inputs.success"
}

stage_run() {
    [[ -f "${OUTPUT_DIR}/.inputs.success" ]] || { echo "inputs stage has not succeeded" >&2; exit 1; }
    local max_args=()
    if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
        max_args=(--max_examples "${PREFLIGHT_MAX_EXAMPLES}")
    fi
    {
        echo "GPU memory immediately before run:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
        cd -- "${BERT_WORK}"
        export DATASET_FILE="${SQUAD_DIR}/dev-v1.1.json"
        export VOCAB_FILE="${MODEL_DIR}/vocab.txt"
        export ML_MODEL_FILE_WITH_PATH="${MODEL_DIR}/model.pytorch"
        export LOG_PATH="${OUTPUT_DIR}"
        export SKIP_VERIFY_ACCURACY=1
        "${VENV_DIR}/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
    } 2>&1 | tee "${OUTPUT_DIR}/run.log"
    test -s "${OUTPUT_DIR}/mlperf_log_accuracy.json"
    test -s "${OUTPUT_DIR}/mlperf_log_detail.txt"
    touch "${OUTPUT_DIR}/.run.success"
}

stage_evaluate() {
    [[ -f "${OUTPUT_DIR}/.run.success" ]] || { echo "run stage has not succeeded" >&2; exit 1; }
    local max_args=()
    if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
        max_args=(--max_examples "${PREFLIGHT_MAX_EXAMPLES}")
    fi
    {
        cd -- "${BERT_WORK}"
        "${VENV_DIR}/bin/python" accuracy-squad.py \
            --vocab_file "${MODEL_DIR}/vocab.txt" \
            --val_data "${SQUAD_DIR}/dev-v1.1.json" \
            --log_file "${OUTPUT_DIR}/mlperf_log_accuracy.json" \
            --out_file "${WORK_DIR}/predictions.json" \
            --features_cache_file "${BERT_WORK}/eval_features.pickle" \
            "${max_args[@]}"
    } 2>&1 | tee "${OUTPUT_DIR}/accuracy.txt" "${OUTPUT_DIR}/accuracy.log"
    test -s "${OUTPUT_DIR}/accuracy.txt"
    touch "${OUTPUT_DIR}/.evaluate.success"
}

run_stage() {
    local stage=$1
    if [[ -f "${OUTPUT_DIR}/.${stage}.success" ]]; then
        echo "Skipping completed stage: ${stage}"
        return
    fi
    "stage_${stage}"
}

if [[ -n "${FROM_STAGE}" ]]; then
    clear_from "${FROM_STAGE}"
    active=0
    for stage in "${STAGES[@]}"; do
        [[ "${stage}" == "${FROM_STAGE}" ]] && active=1
        [[ ${active} -eq 0 ]] || run_stage "${stage}"
    done
elif [[ -n "${REQUESTED_STAGE}" ]]; then
    run_stage "${REQUESTED_STAGE}"
else
    for stage in "${STAGES[@]}"; do
        run_stage "${stage}"
    done
fi
