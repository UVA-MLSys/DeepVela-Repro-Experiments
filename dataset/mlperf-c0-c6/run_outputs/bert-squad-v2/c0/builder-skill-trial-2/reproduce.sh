#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "Usage: $0 --target bert-squad [--stage setup|preprocess|run|evaluate] [--from setup|preprocess|run|evaluate]" >&2
}

TARGET=""
STAGE=""
FROM_STAGE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            TARGET="$2"
            shift 2
            ;;
        --stage)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            STAGE="$2"
            shift 2
            ;;
        --from)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            FROM_STAGE="$2"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

[[ "$TARGET" == "bert-squad" ]] || { echo "Unsupported target: $TARGET" >&2; exit 2; }
[[ -z "$STAGE" || -z "$FROM_STAGE" ]] || { echo "Use only one of --stage and --from" >&2; exit 2; }

for requested_stage in "$STAGE" "$FROM_STAGE"; do
    if [[ -n "$requested_stage" && ! "$requested_stage" =~ ^(setup|preprocess|run|evaluate)$ ]]; then
        echo "Unknown stage: $requested_stage" >&2
        exit 2
    fi
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="$ROOT_DIR/repo"
TARGET_DIR="$ROOT_DIR/$TARGET"
WORK_DIR="$TARGET_DIR/work"
STAGED_DIR="$WORK_DIR/source"
BERT_DIR="$STAGED_DIR/language/bert"
VENV_DIR="$WORK_DIR/venv"

CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
SQUAD_DIR=${SQUAD_DIR:-$CACHE_ROOT/bert-squad/data}
MODEL_DIR=${MODEL_DIR:-$CACHE_ROOT/bert-squad/model}
DEEPLEARNING_EXAMPLES_DIR=${DEEPLEARNING_EXAMPLES_DIR:-$CACHE_ROOT/bert-squad/DeepLearningExamples}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
export UV_EXCLUDE_NEWER=2026-11-10
OUTPUT_DIR=${OUTPUT_DIR:-$TARGET_DIR/out}
export OUTPUT_DIR

mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"

SETUP_MARKER="$WORK_DIR/.setup.success"
PREPROCESS_MARKER="$WORK_DIR/.preprocess.success"
RUN_MARKER="$WORK_DIR/.run.success"
EVALUATE_MARKER="$WORK_DIR/.evaluate.success"

validate_inputs() {
    [[ -d "$REPO_DIR/language/bert" ]] || { echo "Missing source: $REPO_DIR/language/bert" >&2; exit 1; }
    [[ -d "$REPO_DIR/loadgen" ]] || { echo "Missing source: $REPO_DIR/loadgen" >&2; exit 1; }
    [[ -f "$SQUAD_DIR/dev-v1.1.json" ]] || { echo "Missing dataset: $SQUAD_DIR/dev-v1.1.json" >&2; exit 1; }
    [[ -f "$MODEL_DIR/model.pytorch" ]] || { echo "Missing model: $MODEL_DIR/model.pytorch" >&2; exit 1; }
    [[ -f "$MODEL_DIR/vocab.txt" ]] || { echo "Missing vocabulary: $MODEL_DIR/vocab.txt" >&2; exit 1; }
    [[ -f "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py" ]] || {
        echo "Missing staged DeepLearningExamples BERT source" >&2
        exit 1
    }
    [[ -d "$UV_CACHE_DIR" ]] || { echo "Missing uv cache: $UV_CACHE_DIR" >&2; exit 1; }
}

clear_from() {
    local first="$1"
    case "$first" in
        setup)
            if [[ -d "$WORK_DIR" ]]; then
                rm -rf -- "$WORK_DIR"
            fi
            rm -f -- "$OUTPUT_DIR"/*.log "$OUTPUT_DIR"/mlperf_log_*.json \
                "$OUTPUT_DIR"/mlperf_log_*.txt "$OUTPUT_DIR"/accuracy.txt
            ;;
        preprocess)
            rm -f -- "$PREPROCESS_MARKER" "$RUN_MARKER" "$EVALUATE_MARKER" \
                "$BERT_DIR/eval_features.pickle" "$OUTPUT_DIR/preprocess.log" \
                "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt" \
                "$OUTPUT_DIR"/mlperf_log_*.json "$OUTPUT_DIR"/mlperf_log_*.txt
            ;;
        run)
            rm -f -- "$RUN_MARKER" "$EVALUATE_MARKER" "$OUTPUT_DIR/run.log" \
                "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt" \
                "$OUTPUT_DIR"/mlperf_log_*.json "$OUTPUT_DIR"/mlperf_log_*.txt
            ;;
        evaluate)
            rm -f -- "$EVALUATE_MARKER" "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
            ;;
    esac
}

stage_setup() {
    [[ ! -f "$SETUP_MARKER" ]] || return 0
    if [[ -d "$WORK_DIR" ]]; then
        rm -rf -- "$WORK_DIR"
    fi
    mkdir -p "$WORK_DIR" "$STAGED_DIR/language"
    {
        echo "stage: setup"
        echo "python requested: 3.11"
        echo "UV_CACHE_DIR=$UV_CACHE_DIR"
        echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
        cp -a "$REPO_DIR/language/bert" "$STAGED_DIR/language/bert"
        cp -a "$REPO_DIR/loadgen" "$STAGED_DIR/loadgen"
        rmdir "$BERT_DIR/DeepLearningExamples"
        ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"
        mkdir -p "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2" "$BERT_DIR/build/result"
        ln -s "$SQUAD_DIR/dev-v1.1.json" "$BERT_DIR/build/data/dev-v1.1.json"
        ln -s "$MODEL_DIR/model.pytorch" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
        ln -s "$MODEL_DIR/vocab.txt" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"

        python3 - "$BERT_DIR/pytorch_SUT.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
needle = "        self.model.load_state_dict(torch.load(model_file), strict=False)\n"
replacement = needle + '        print("pytorch device: {}".format(next(self.model.parameters()).device), flush=True)\n'
if needle not in s:
    raise SystemExit("expected checkpoint-load line was not found")
p.write_text(s.replace(needle, replacement, 1))
PY

        uv venv --python 3.11 "$VENV_DIR"
        uv pip install --python "$VENV_DIR/bin/python" --offline \
            --index-url https://download.pytorch.org/whl/cu118 \
            --no-deps 'torch==2.1.2+cu118'
        uv pip install --python "$VENV_DIR/bin/python" --offline \
            'transformers==4.31.0' 'numpy==1.26.4' 'absl-py==2.1.0' \
            'six==1.16.0' 'pybind11==2.11.1' 'setuptools==69.5.1' 'wheel==0.44.0' \
            'filelock' 'typing-extensions' 'sympy' 'networkx' 'jinja2' 'fsspec' 'boto3'
        uv pip install --python "$VENV_DIR/bin/python" --offline --no-build-isolation "$STAGED_DIR/loadgen"
        "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
        sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    } 2>&1 | tee "$OUTPUT_DIR/build.log"
    touch "$SETUP_MARKER"
}

stage_preprocess() {
    [[ -f "$SETUP_MARKER" ]] || { echo "setup stage has not succeeded" >&2; exit 1; }
    [[ ! -f "$PREPROCESS_MARKER" ]] || return 0
    {
        echo "stage: preprocess"
        cd "$BERT_DIR"
        PYTHONPATH="$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
        DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
        VOCAB_FILE="$MODEL_DIR/vocab.txt" \
        "$VENV_DIR/bin/python" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL()
print("preprocessed feature count:", qsl.count)
PY
    } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
    [[ -s "$BERT_DIR/eval_features.pickle" ]] || { echo "feature cache was not created" >&2; exit 1; }
    touch "$PREPROCESS_MARKER"
}

stage_run() {
    [[ -f "$PREPROCESS_MARKER" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
    [[ ! -f "$RUN_MARKER" ]] || return 0
    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')
    {
        echo "stage: run"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
        echo "gpu free MiB immediately before run: $free_mib"
    } | tee "$OUTPUT_DIR/gpu-check.log"
    if [[ ! "$free_mib" =~ ^[0-9]+$ || "$free_mib" -lt 8192 ]]; then
        echo "Insufficient free GPU memory: ${free_mib:-unknown} MiB; require at least 8192 MiB" | tee "$OUTPUT_DIR/run.log" >&2
        exit 1
    fi
    rm -f -- "$OUTPUT_DIR"/mlperf_log_*.json "$OUTPUT_DIR"/mlperf_log_*.txt
    local -a max_args=()
    if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
        max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
    fi
    {
        cd "$BERT_DIR"
        LOG_PATH="$OUTPUT_DIR" \
        DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
        VOCAB_FILE="$MODEL_DIR/vocab.txt" \
        ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch" \
        SKIP_VERIFY_ACCURACY=1 \
        "$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
    } 2>&1 | tee "$OUTPUT_DIR/run.log"
    [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" ]] || { echo "LoadGen accuracy log missing" >&2; exit 1; }
    [[ -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]] || { echo "LoadGen detail log missing" >&2; exit 1; }
    rg -q '^pytorch device: cuda' "$OUTPUT_DIR/run.log" || { echo "CUDA device evidence missing" >&2; exit 1; }
    touch "$RUN_MARKER"
}

stage_evaluate() {
    [[ -f "$RUN_MARKER" ]] || { echo "run stage has not succeeded" >&2; exit 1; }
    [[ ! -f "$EVALUATE_MARKER" ]] || return 0
    local -a max_args=()
    if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
        max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
    fi
    {
        cd "$BERT_DIR"
        "$VENV_DIR/bin/python" accuracy-squad.py \
            --vocab_file "$MODEL_DIR/vocab.txt" \
            --val_data "$SQUAD_DIR/dev-v1.1.json" \
            --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
            --out_file "$BERT_DIR/build/result/predictions.json" \
            --features_cache_file "$BERT_DIR/eval_features.pickle" \
            "${max_args[@]}"
    } 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
    "$VENV_DIR/bin/python" - "$OUTPUT_DIR/accuracy.txt" <<'PY'
import json
import math
from pathlib import Path
import sys

for line in reversed(Path(sys.argv[1]).read_text().splitlines()):
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(data, dict) and {"exact_match", "f1"} <= data.keys():
        assert all(math.isfinite(float(data[key])) for key in ("exact_match", "f1"))
        break
else:
    raise SystemExit("no evaluator metric JSON object found")
PY
    touch "$EVALUATE_MARKER"
}

validate_inputs

if [[ -n "$FROM_STAGE" ]]; then
    clear_from "$FROM_STAGE"
fi

if [[ -n "$STAGE" ]]; then
    "stage_$STAGE"
    exit 0
fi

start_stage=${FROM_STAGE:-setup}
run_enabled=false
for current_stage in setup preprocess run evaluate; do
    if [[ "$current_stage" == "$start_stage" ]]; then
        run_enabled=true
    fi
    if $run_enabled; then
        "stage_$current_stage"
    fi
done
