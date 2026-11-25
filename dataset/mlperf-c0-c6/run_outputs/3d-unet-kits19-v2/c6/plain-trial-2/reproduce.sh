#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target 3d-unet" >&2
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
[[ "$target" == "3d-unet" ]] || usage

root_dir="$(pwd -P)"
repo_dir="$root_dir/repo"
work_dir="$root_dir/$target"
output_dir="${OUTPUT_DIR:-$work_dir/out}"
cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
raw_dir="$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data"
model_dir="$cache_root/3d-unet-kits19/model"
bench_rel="vision/medical_imaging/3d-unet-kits19"
bench_dir="$work_dir/benchmark"
venv_dir="$work_dir/.venv"

[[ -f "$repo_dir/$bench_rel/run.py" ]] || { echo "missing repository beside runner: $repo_dir" >&2; exit 1; }
[[ -d "$raw_dir" ]] || { echo "missing KITS19_RAW_DIR: $raw_dir" >&2; exit 1; }
[[ -f "$model_dir/3dunet_kits19_pytorch.ptc" ]] || { echo "missing fp32 model in MODEL_DIR: $model_dir" >&2; exit 1; }

mkdir -p "$work_dir" "$output_dir"
output_dir="$(cd "$output_dir" && pwd -P)"
export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER=2026-11-10

{
    echo "Creating Python 3.11 environment"
    uv venv --python python3.11 --seed --clear "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" --offline \
        'numpy==1.26.4' 'scipy==1.12.0' 'pandas==1.5.3' \
        'nibabel==5.4.2' 'torch==2.7.1' 'pybind11==2.11.1'
    uv pip install --python "$venv_dir/bin/python" --offline --no-build-isolation "$repo_dir/loadgen"
} >"$work_dir/build.log" 2>&1

rm -rf "$bench_dir"
cp -a "$repo_dir/$bench_rel" "$bench_dir"
cp "$repo_dir/loadgen/mlperf.conf" "$bench_dir/mlperf.conf"
# The target requires this diagnostic to come from the loaded model.
sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$bench_dir/pytorch_SUT.py"
# This LoadGen version rejects multiple FromConfig calls. user.conf only tunes
# performance QPS and has no bearing on an AccuracyOnly run.
sed -i '/settings.FromConfig(args.user_conf/d' "$bench_dir/run.py"

mkdir -p "$work_dir/raw_data"
"$venv_dir/bin/python" - "$bench_dir/meta/inference_cases.json" "$raw_dir" "$work_dir/raw_data" <<'PY'
import json
import os
import pathlib
import sys

case_file, raw_root, link_root = sys.argv[1:]
for case in json.load(open(case_file)):
    source_case = "case_00185" if case == "case_00400" else case
    source = pathlib.Path(raw_root, source_case)
    target = pathlib.Path(link_root, case)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.is_symlink() or target.exists():
        target.unlink()
    os.symlink(source, target, target_is_directory=True)
PY

mkdir -p "$bench_dir/build/preprocessed_data" "$bench_dir/build/postprocessed_data"
(
    cd "$bench_dir"
    "$venv_dir/bin/python" preprocess.py \
        --raw_data_dir "$work_dir/raw_data" \
        --results_dir "$bench_dir/build/preprocessed_data" \
        --mode preprocess
) >"$work_dir/preprocess.log" 2>&1

{
    date -Is
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits
} >"$work_dir/gpu-memory.log" 2>&1

rm -f "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" \
    "$output_dir/mlperf_log_summary.txt" "$output_dir/accuracy.txt"
export LOG_PATH="$output_dir"
export SKIP_VERIFY_ACCURACY=1
(
    cd "$bench_dir"
    "$venv_dir/bin/python" run.py \
        --backend=pytorch \
        --scenario=Offline \
        --accuracy \
        --model="$model_dir/3dunet_kits19_pytorch.ptc" \
        --preprocessed_data_dir="$bench_dir/build/preprocessed_data" \
        --mlperf_conf="$bench_dir/mlperf.conf" \
        --user_conf="$bench_dir/user.conf"
) >"$work_dir/run.log" 2>&1

(
    cd "$bench_dir"
    "$venv_dir/bin/python" accuracy_kits.py \
        --log_file "$output_dir/mlperf_log_accuracy.json" \
        --preprocessed_data_dir "$bench_dir/build/preprocessed_data" \
        --postprocessed_data_dir "$bench_dir/build/postprocessed_data" \
        2>"$work_dir/evaluator.stderr.log"
) | tee "$work_dir/evaluator.log" "$output_dir/accuracy.txt"

for required in mlperf_log_accuracy.json mlperf_log_detail.txt accuracy.txt; do
    [[ -s "$output_dir/$required" ]] || { echo "missing required output: $output_dir/$required" >&2; exit 1; }
done
