#!/usr/bin/env bash
set -euo pipefail

target=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ $# -ge 2 ]] || { echo "--target requires a value" >&2; exit 2; }
            target="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "$target" != "3d-unet" ]]; then
    echo "unsupported target: $target" >&2
    exit 2
fi
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_dir="$root/$target"
work_dir="$target_dir/work"
venv_dir="$target_dir/.venv"
cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
raw_dir="$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data"
model="$cache_root/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"

if [[ "$OUTPUT_DIR" = /* ]]; then
    output_dir="$OUTPUT_DIR"
else
    output_dir="$(pwd)/$OUTPUT_DIR"
fi

mkdir -p "$target_dir" "$output_dir" "$work_dir/build/raw_data"
cp -a "$root/repo/vision/medical_imaging/3d-unet-kits19/." "$work_dir/"
cp "$root/repo/loadgen/mlperf.conf" "$work_dir/build/mlperf.conf"

for case_dir in "$raw_dir"/case_*; do
    ln -sfn "$case_dir" "$work_dir/build/raw_data/$(basename "$case_dir")"
done
ln -sfn "$raw_dir/case_00185" "$work_dir/build/raw_data/case_00400"

export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER=2026-11-10
{
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" \
        'torch==2.9.0' 'numpy<2' 'scipy==1.10.1' \
        'nibabel<6' 'pandas<2' mlcommons-loadgen
    "$venv_dir/bin/python" - <<'PY'
import mlperf_loadgen
import nibabel
import numpy
import pandas
import scipy
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__,
      "nibabel", nibabel.__version__, "pandas", pandas.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
} > "$target_dir/build.log" 2>&1

export PATH="$venv_dir/bin:$PATH"
cd "$work_dir"

python preprocess.py \
    --raw_data_dir build/raw_data \
    --results_dir build/preprocessed_data \
    --mode preprocess --num_proc 16 \
    > "$target_dir/preprocess.log" 2>&1

export LOG_PATH="$output_dir"
export SKIP_VERIFY_ACCURACY=1
{
    echo "GPU memory immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader
    python - <<'PY'
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"pytorch device: {device}", flush=True)
if device == "cpu":
    raise SystemExit("CUDA is unavailable")
PY
    python run.py --backend=pytorch --scenario=Offline --accuracy \
        --model="$model" \
        --preprocessed_data_dir=build/preprocessed_data \
        --mlperf_conf=build/mlperf.conf
} > "$target_dir/run.log" 2>&1

python accuracy_kits.py \
    --log_file="$output_dir/mlperf_log_accuracy.json" \
    --preprocessed_data_dir=build/preprocessed_data \
    --postprocessed_data_dir=build/postprocessed_data \
    --num_proc=4 \
    > "$output_dir/accuracy.txt" 2> "$target_dir/evaluator.log"

test -s "$output_dir/mlperf_log_accuracy.json"
test -s "$output_dir/mlperf_log_detail.txt"
test -s "$output_dir/accuracy.txt"
