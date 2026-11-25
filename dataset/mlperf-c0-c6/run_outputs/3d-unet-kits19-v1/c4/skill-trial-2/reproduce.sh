#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target 3d-unet [--stage environment|preprocess|run] [--from environment|preprocess|run]" >&2
    exit 2
}

target=""
stage=""
from_stage=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) target="${2:-}"; shift 2 ;;
        --stage) stage="${2:-}"; shift 2 ;;
        --from) from_stage="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done

[[ "$target" == "3d-unet" ]] || usage
[[ -z "$stage" || -z "$from_stage" ]] || usage
for selected in "$stage" "$from_stage"; do
    [[ -z "$selected" || "$selected" == "environment" || "$selected" == "preprocess" || "$selected" == "run" ]] || usage
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$script_dir/repo"
cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
raw_input="${KITS19_RAW_DIR:-$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data}"
model_input_dir="${MODEL_DIR:-$cache_root/3d-unet-kits19/model}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache_root/uv}"
export UV_EXCLUDE_NEWER=2026-11-10

# PREFLIGHT_CASE is only for a bounded end-to-end validation. Normal target runs
# always use all 43 manifest entries and ./3d-unet as required by the contract.
if [[ -n "${PREFLIGHT_CASE:-}" ]]; then
    target_dir="$script_dir/$target/preflight"
else
    target_dir="$script_dir/$target"
fi
work_dir="$target_dir/work"
state_dir="$target_dir/state"
source_dir="$work_dir/repo/vision/medical_imaging/3d-unet-kits19"
venv_dir="$work_dir/venv"
preprocessed_dir="$work_dir/preprocessed_data"
postprocessed_dir="$work_dir/postprocessed_data"
raw_dir="$work_dir/raw_data"
model_file="$model_input_dir/3dunet_kits19_pytorch.ptc"
output_dir="${OUTPUT_DIR:-$target_dir/out}"
mkdir -p "$output_dir" "$state_dir"

log_run() {
    local log_file="$1"
    shift
    "$@" 2>&1 | tee "$log_file"
}

invalidate_from() {
    local first="$1"
    case "$first" in
        environment)
            rm -rf "$work_dir" "$state_dir"
            mkdir -p "$state_dir"
            rm -f "$output_dir"/*.log "$output_dir/mlperf_log_accuracy.json" \
                "$output_dir/mlperf_log_detail.txt" "$output_dir/mlperf_log_summary.txt" \
                "$output_dir/accuracy.txt" "$output_dir/mlperf_log_trace.json"
            ;;
        preprocess)
            rm -rf "$preprocessed_dir" "$postprocessed_dir"
            rm -f "$state_dir/preprocess.done" "$state_dir/run.done"
            rm -f "$output_dir/preprocess.log" "$output_dir/run.log" \
                "$output_dir/evaluator.log" "$output_dir/gpu_pre_run.log" \
                "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" \
                "$output_dir/mlperf_log_summary.txt" "$output_dir/accuracy.txt" \
                "$output_dir/mlperf_log_trace.json"
            ;;
        run)
            rm -rf "$postprocessed_dir"
            rm -f "$state_dir/run.done"
            rm -f "$output_dir/run.log" "$output_dir/evaluator.log" \
                "$output_dir/gpu_pre_run.log" "$output_dir/mlperf_log_accuracy.json" \
                "$output_dir/mlperf_log_detail.txt" "$output_dir/mlperf_log_summary.txt" \
                "$output_dir/accuracy.txt" "$output_dir/mlperf_log_trace.json"
            ;;
    esac
}

environment_stage() {
    [[ -f "$state_dir/environment.done" ]] && return
    {
        echo "stage: environment"
        echo "source repository: $repo_dir"
        echo "raw input: $raw_input"
        echo "model input: $model_file"
        echo "UV_CACHE_DIR: $UV_CACHE_DIR"
        echo "UV_EXCLUDE_NEWER: $UV_EXCLUDE_NEWER"
        test -d "$repo_dir"
        test -d "$raw_input"
        test -s "$model_file"
        command -v uv

        mkdir -p "$work_dir"
        if [[ ! -d "$work_dir/repo" ]]; then
            cp -a "$repo_dir" "$work_dir/repo"
        fi

        # Required observability missing from the pinned source: print the
        # actual loaded parameter device, without changing model execution.
        if ! grep -q 'pytorch device: ' "$source_dir/pytorch_SUT.py"; then
            sed -i '/self.model.eval()/a\        print("pytorch device: ", next(self.model.parameters()).device)' \
                "$source_dir/pytorch_SUT.py"
        fi

        # This commit's evaluator reads a row named "mean" but omits the line
        # that creates it. Restore the documented overall mean aggregation.
        if ! grep -q 'df.loc\["mean"\]' "$source_dir/accuracy_kits.py"; then
            sed -i '/    df.to_csv(sum_path)/i\    df.loc["mean"] = df.mean(numeric_only=True)' \
                "$source_dir/accuracy_kits.py"
        fi

        # pandas 1.5 keeps the evaluator's DataFrame.append API; this is an
        # environment compatibility choice, not an evaluator modification.
        if [[ ! -x "$venv_dir/bin/python" ]]; then
            uv venv --python python3.10 "$venv_dir"
        fi
        uv pip install --offline --python "$venv_dir/bin/python" \
            numpy==1.26.4 scipy==1.10.1 nibabel==3.2.1 pandas==1.5.3 \
            torch==2.4.1 setuptools wheel pybind11==2.11.1
        uv pip install --offline --python "$venv_dir/bin/python" \
            --no-build-isolation "$work_dir/repo/loadgen"

        mkdir -p "$raw_dir"
        for case_path in "$raw_input"/case_*; do
            [[ -d "$case_path" ]] || continue
            ln -sfn "$case_path" "$raw_dir/$(basename "$case_path")"
        done
        ln -sfn "$raw_input/case_00185" "$raw_dir/case_00400"
        ln -sfn "$model_file" "$work_dir/3dunet_kits19_pytorch.ptc"
        cp -L "$work_dir/repo/mlperf.conf" "$work_dir/mlperf.conf"

        if [[ -n "${PREFLIGHT_CASE:-}" ]]; then
            "$venv_dir/bin/python" - "$source_dir" "$PREFLIGHT_CASE" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
case = sys.argv[2]
case_file = source / "meta" / "inference_cases.json"
checksum_file = source / "meta" / "checksum_inference.json"
checksums = json.loads(checksum_file.read_text())
key = case + ".pkl"
if key not in checksums:
    raise SystemExit(f"preflight case absent from checksum manifest: {case}")
case_file.write_text(json.dumps([case], indent=4) + "\n")
checksum_file.write_text(json.dumps({key: checksums[key]}, indent=4, sort_keys=True) + "\n")
PY
        fi

        "$venv_dir/bin/python" - <<'PY'
import nibabel
import numpy
import pandas
import scipy
import torch
import mlperf_loadgen
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("nibabel", nibabel.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
PY
    } 2>&1 | tee "$output_dir/build.log"
    touch "$state_dir/environment.done"
}

preprocess_stage() {
    [[ -f "$state_dir/preprocess.done" ]] && return
    [[ -f "$state_dir/environment.done" ]] || environment_stage
    mkdir -p "$preprocessed_dir"
    (
        cd "$source_dir"
        log_run "$output_dir/preprocess.log" "$venv_dir/bin/python" preprocess.py \
            --raw_data_dir "$raw_dir" --results_dir "$preprocessed_dir" \
            --mode preprocess --num_proc 16
    )
    touch "$state_dir/preprocess.done"
}

check_gpu_memory() {
    local free_mib
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader | tee "$output_dir/gpu_pre_run.log"
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 16384 )); then
        echo "insufficient free GPU memory: ${free_mib:-unknown} MiB; need at least 16384 MiB" | tee -a "$output_dir/gpu_pre_run.log" >&2
        return 1
    fi
}

run_stage() {
    [[ -f "$state_dir/run.done" ]] && return
    [[ -f "$state_dir/preprocess.done" ]] || preprocess_stage
    mkdir -p "$postprocessed_dir"
    check_gpu_memory
    (
        cd "$source_dir"
        export LOG_PATH="$output_dir"
        export SKIP_VERIFY_ACCURACY=1
        log_run "$output_dir/run.log" "$venv_dir/bin/python" run.py \
            --backend=pytorch --scenario=Offline --accuracy \
            --model="$work_dir/3dunet_kits19_pytorch.ptc" \
            --preprocessed_data_dir="$preprocessed_dir" \
            --mlperf_conf="$work_dir/mlperf.conf" --user_conf=user.conf

        "$venv_dir/bin/python" accuracy_kits.py \
            --log_file="$output_dir/mlperf_log_accuracy.json" \
            --preprocessed_data_dir="$preprocessed_dir" \
            --postprocessed_data_dir="$postprocessed_dir" --num_proc=16 \
            2>&1 | tee "$output_dir/accuracy.txt" "$output_dir/evaluator.log"
    )
    test -s "$output_dir/mlperf_log_accuracy.json"
    test -s "$output_dir/mlperf_log_detail.txt"
    grep -q '^pytorch device: ' "$output_dir/run.log"
    grep -q '^Accuracy: mean = ' "$output_dir/accuracy.txt"
    touch "$state_dir/run.done"
}

if [[ -n "$from_stage" ]]; then
    invalidate_from "$from_stage"
fi

if [[ -n "$stage" ]]; then
    "${stage}_stage"
    exit 0
fi

start="${from_stage:-environment}"
case "$start" in
    environment) environment_stage; preprocess_stage; run_stage ;;
    preprocess) preprocess_stage; run_stage ;;
    run) run_stage ;;
esac
