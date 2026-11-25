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
from_stage=""
max_examples=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) target=${2:?}; shift 2 ;;
        --stage) stage=${2:?}; shift 2 ;;
        --from) from_stage=${2:?}; shift 2 ;;
        --max-examples) max_examples=${2:?}; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$target" != "bert-squad" ]]; then
    echo "target must be bert-squad" >&2
    exit 2
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    echo "OUTPUT_DIR must be set" >&2
    exit 2
fi
case "$stage" in ""|setup|preprocess|run) ;; *) echo "invalid stage: $stage" >&2; exit 2;; esac
case "$from_stage" in ""|setup|preprocess|run) ;; *) echo "invalid from stage: $from_stage" >&2; exit 2;; esac
if [[ -n "$stage" && -n "$from_stage" ]]; then
    echo "use only one of --stage and --from" >&2
    exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo="$root/repo"
output_dir=$(realpath -m "$OUTPUT_DIR")
target_dir=$(dirname "$output_dir")
work_dir="$target_dir/work"
bench_dir="$work_dir/language/bert"
loadgen_dir="$work_dir/loadgen"
venv="$target_dir/venv"
state_dir="$target_dir/state"
mkdir -p "$output_dir" "$state_dir"

run_setup() {
    exec > >(tee "$output_dir/setup.log") 2>&1
    echo "setup: creating isolated source and Python environment"
    test -f "$repo/language/bert/run.py"
    test -f "$SQUAD_DIR/dev-v1.1.json"
    test -f "$MODEL_DIR/model.pytorch"
    test -f "$MODEL_DIR/vocab.txt"
    test -f "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"

    mkdir -p "$work_dir/language"
    cp -a "$repo/language/bert" "$work_dir/language/"
    cp -a "$repo/loadgen" "$work_dir/"
    mkdir -p "$bench_dir/DeepLearningExamples/PyTorch/LanguageModeling"
    cp -a "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT" \
        "$bench_dir/DeepLearningExamples/PyTorch/LanguageModeling/"

    # Add the task-required evidence line only to the isolated source copy.
    perl -0pi -e 's/(self\.model\.load_state_dict\(torch\.load\(model_file\), strict=False\)\n)/$1        print("pytorch device: {}".format(next(self.model.parameters()).device))\n/' \
        "$bench_dir/pytorch_SUT.py"
    grep -q 'pytorch device: ' "$bench_dir/pytorch_SUT.py"

    uv venv --python 3.12 --clear "$venv"
    uv pip install --python "$venv/bin/python" \
        'numpy==1.26.4' 'torch==2.7.1' 'transformers==4.53.2' \
        'absl-py==2.3.1' 'six==1.17.0' 'setuptools==80.9.0' \
        'boto3==1.40.70'
    uv pip install --python "$venv/bin/python" "$loadgen_dir"
    "$venv/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "numpy", numpy.__version__)
PY
    touch "$state_dir/setup.ok"
}

run_preprocess() {
    exec > >(tee "$output_dir/preprocess.log") 2>&1
    test -f "$state_dir/setup.ok"
    echo "preprocess: generating the complete SQuAD feature cache"
    cd "$bench_dir"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export PYTHONPATH="$bench_dir/DeepLearningExamples/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}"
    "$venv/bin/python" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL(cache_path="eval_features.pickle")
print("preprocess feature count:", len(qsl.eval_features))
PY
    test -s eval_features.pickle
    touch "$state_dir/preprocess.ok"
}

run_benchmark() {
    test -f "$state_dir/preprocess.ok"
    cd "$bench_dir"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export LOG_PATH="$output_dir"
    export SKIP_VERIFY_ACCURACY=1
    export PYTHONPATH="$bench_dir/DeepLearningExamples/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}"

    {
        echo "gpu preflight: checking free memory immediately before run"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used \
            --format=csv,noheader
    } | tee "$output_dir/gpu-check.log"
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( free_mib < 8192 )); then
        echo "insufficient free GPU memory: ${free_mib} MiB" | tee -a "$output_dir/gpu-check.log" >&2
        exit 1
    fi

    run_args=(--backend=pytorch --accuracy)
    eval_args=(
        --vocab_file "$MODEL_DIR/vocab.txt"
        --val_data "$SQUAD_DIR/dev-v1.1.json"
        --log_file "$output_dir/mlperf_log_accuracy.json"
        --out_file "$output_dir/predictions.json"
        --features_cache_file "$bench_dir/eval_features.pickle"
    )
    if [[ -n "$max_examples" ]]; then
        run_args+=(--max_examples "$max_examples")
        eval_args+=(--max_examples "$max_examples")
    fi

    echo "run: python3 run.py --backend=pytorch --accuracy${max_examples:+ --max_examples $max_examples}" \
        | tee "$output_dir/run.log"
    "$venv/bin/python" run.py "${run_args[@]}" 2>&1 | tee -a "$output_dir/run.log"
    test -s "$output_dir/mlperf_log_accuracy.json"
    test -s "$output_dir/mlperf_log_detail.txt"
    "$venv/bin/python" accuracy-squad.py "${eval_args[@]}" 2>&1 \
        | tee "$output_dir/accuracy.txt" "$output_dir/evaluator.log"
    touch "$state_dir/run.ok"
}

clear_from() {
    case "$1" in
        setup)
            rm -f "$state_dir/setup.ok" "$state_dir/preprocess.ok" "$state_dir/run.ok"
            ;;
        preprocess)
            rm -f "$state_dir/preprocess.ok" "$state_dir/run.ok"
            ;;
        run)
            rm -f "$state_dir/run.ok"
            ;;
    esac
}

if [[ -n "$stage" ]]; then
    case "$stage" in
        setup) run_setup ;;
        preprocess) run_preprocess ;;
        run) run_benchmark ;;
    esac
    exit 0
fi

start=${from_stage:-setup}
clear_from "$start"
if [[ "$start" == setup ]]; then run_setup; fi
if [[ "$start" == setup || "$start" == preprocess ]]; then run_preprocess; fi
run_benchmark
