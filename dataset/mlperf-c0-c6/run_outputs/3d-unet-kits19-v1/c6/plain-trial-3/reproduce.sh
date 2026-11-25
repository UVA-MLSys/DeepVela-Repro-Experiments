#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
KITS19_RAW_DIR="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR="$CACHE_ROOT/3d-unet-kits19/model"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

usage() {
    echo "usage: $0 --target 3d-unet" >&2
    exit 2
}

[[ $# -eq 2 && "$1" == "--target" ]] || usage
target=$2
[[ "$target" == "3d-unet" ]] || usage

root=$(cd "$(dirname "$0")" && pwd)
target_dir="$root/$target"
output_dir=${OUTPUT_DIR:-"$target_dir/out"}
mkdir -p "$target_dir" "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
work_dir="$target_dir/work"
venv="$target_dir/.venv"
benchmark_src="$root/repo/vision/medical_imaging/3d-unet-kits19"

[[ -f "$benchmark_src/run.py" ]] || {
    echo "3D U-Net source not found below $root/repo" >&2
    exit 1
}
[[ -f "$MODEL_DIR/3dunet_kits19_pytorch.ptc" ]] || {
    echo "supplied fp32 model not found in MODEL_DIR" >&2
    exit 1
}

{
    uv venv --clear --python /usr/bin/python3.11 "$venv"
    uv pip install --python "$venv/bin/python" --offline \
        'numpy==1.26.4' 'scipy==1.11.4' 'pandas==1.5.3' \
        'nibabel==5.2.1' 'torch==2.4.1'

    # The shared cache contains the CPython 3.11 wheel built from LoadGen 6.0.17,
    # the same version as this source tree. Link its extension into this venv.
    loadgen_wheel="$UV_CACHE_DIR/wheels-v6/pypi/mlcommons-loadgen/6.0.17-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64"
    loadgen_so="$loadgen_wheel/mlperf_loadgen.cpython-311-x86_64-linux-gnu.so"
    [[ -f "$loadgen_so" ]] || {
        echo "cached LoadGen 6.0.17 CPython 3.11 wheel is unavailable" >&2
        exit 1
    }
    site_dir=$("$venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')
    ln -sf "$loadgen_so" "$site_dir/"

    "$venv/bin/python" - <<'PY'
import nibabel, numpy, pandas, scipy, torch, mlperf_loadgen
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("pandas", pandas.__version__)
print("nibabel", nibabel.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("loadgen import ok")
PY
} >"$target_dir/build.log" 2>&1

if [[ ! -f "$work_dir/run.py" ]]; then
    mkdir -p "$work_dir"
    cp -a "$benchmark_src/." "$work_dir/"
fi

# SciPy moved this function into signal.windows. This is API-only compatibility.
sed -i 's/signal\.gaussian(n, std)/signal.windows.gaussian(n, std)/' \
    "$work_dir/inference_utils.py"

# Required evidence: print the actual device holding the loaded model parameters.
if ! grep -q 'pytorch device:' "$work_dir/pytorch_SUT.py"; then
    sed -i '/self\.model\.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
        "$work_dir/pytorch_SUT.py"
fi

mkdir -p "$work_dir/build/raw_data" "$work_dir/build/preprocessed_data" \
    "$work_dir/build/postprocessed_data" "$work_dir/build/model"
ln -sf "$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
    "$work_dir/build/model/3dunet_kits19_pytorch.ptc"
ln -sf "$root/repo/loadgen/mlperf.conf" "$work_dir/build/mlperf.conf"

for case_dir in "$KITS19_RAW_DIR"/case_*; do
    ln -sf "$case_dir" "$work_dir/build/raw_data/"
done
# case_00400 is the benchmark's documented duplicate of case_00185.
ln -sfn "$KITS19_RAW_DIR/case_00185" "$work_dir/build/raw_data/case_00400"

if [[ ! -f "$work_dir/build/preprocessed_data/preprocessed_files.pkl" ]]; then
    (
        cd "$work_dir"
        "$venv/bin/python" preprocess.py \
            --raw_data_dir build/raw_data \
            --results_dir build/preprocessed_data \
            --mode preprocess --num_proc 8
    ) >"$target_dir/preprocess.log" 2>&1
else
    echo "Using checksum-verified preprocessed data already in $work_dir" \
        >"$target_dir/preprocess.log"
fi

gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader -i 0 | tee "$target_dir/gpu-before-run.log"
if (( gpu_free_mib < 12000 )); then
    echo "A100 has only ${gpu_free_mib} MiB free; at least 12000 MiB is required" >&2
    exit 1
fi

(
    cd "$work_dir"
    CUDA_VISIBLE_DEVICES=0 LOG_PATH="$output_dir" SKIP_VERIFY_ACCURACY=1 \
        "$venv/bin/python" run.py \
        --backend=pytorch --scenario=Offline --accuracy \
        --model=build/model/3dunet_kits19_pytorch.ptc \
        --preprocessed_data_dir=build/preprocessed_data \
        --mlperf_conf=build/mlperf.conf --user_conf=user.conf
) >"$target_dir/run.log" 2>&1

(
    cd "$work_dir"
    "$venv/bin/python" accuracy_kits.py \
        --log_file="$output_dir/mlperf_log_accuracy.json" \
        --preprocessed_data_dir=build/preprocessed_data \
        --postprocessed_data_dir=build/postprocessed_data \
        --num_proc=8
) 2>&1 | tee "$target_dir/evaluator.log" "$output_dir/accuracy.txt"

test -s "$output_dir/mlperf_log_accuracy.json"
test -s "$output_dir/mlperf_log_detail.txt"
test -s "$output_dir/accuracy.txt"
grep -q '^pytorch device: ' "$target_dir/run.log"
grep -q '^Accuracy: mean = ' "$output_dir/accuracy.txt"
