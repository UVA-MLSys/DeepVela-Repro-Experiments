#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target <target id>" >&2
    exit 2
}

if [[ $# -ne 2 || "$1" != "--target" ]]; then
    usage
fi
target="$2"
if [[ "$target" != "3d-unet" ]]; then
    echo "unknown target: $target" >&2
    exit 2
fi

root_dir="$(cd "$(dirname "$0")" && pwd)"
target_dir="$root_dir/$target"
output_dir="${OUTPUT_DIR:-$target_dir/out}"
cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
raw_input="$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data"
model_input="$cache_root/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
uv_cache="$cache_root/uv"
work_dir="$target_dir/work"
src_dir="$work_dir/repo"
bench_dir="$src_dir/vision/medical_imaging/3d-unet-kits19"
venv_dir="$target_dir/venv"
raw_view="$work_dir/raw_data"

mkdir -p "$target_dir" "$output_dir" "$work_dir"
export UV_CACHE_DIR="$uv_cache"
export UV_EXCLUDE_NEWER=2026-11-10

if [[ ! -d "$src_dir" ]]; then
    cp -a "$root_dir/repo" "$src_dir"
fi

{
    echo "Creating Python environment"
    uv venv --python 3.10 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" \
        'numpy==1.26.4' 'scipy==1.11.4' 'nibabel==5.2.1' \
        'pandas==1.5.3' 'torch==2.1.2'
    uv pip install --python "$venv_dir/bin/python" "$src_dir/loadgen"
    "$venv_dir/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("nibabel", nibabel.__version__)
print("pandas", pandas.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
} 2>&1 | tee "$target_dir/build.log"

# This commit predates the requested device marker. Add it to the private copy.
if ! grep -q 'pytorch device:' "$bench_dir/pytorch_SUT.py"; then
    sed -i '/"cuda:0" if torch.cuda.is_available() else "cpu")/a\        print("pytorch device: {}".format(self.device))' \
        "$bench_dir/pytorch_SUT.py"
fi

mkdir -p "$raw_view"
while IFS= read -r case_path; do
    case_name="$(basename "$case_path")"
    ln -sfn "$case_path" "$raw_view/$case_name"
done < <(find "$raw_input" -mindepth 1 -maxdepth 1 -type d | sort)
ln -sfn "$raw_input/case_00185" "$raw_view/case_00400"

mkdir -p "$bench_dir/build/preprocessed_data" "$bench_dir/build/postprocessed_data" "$bench_dir/build/logs"
cp "$src_dir/loadgen/mlperf.conf" "$bench_dir/build/mlperf.conf"

cd "$bench_dir"
"$venv_dir/bin/python" preprocess.py \
    --raw_data_dir "$raw_view" \
    --results_dir "$bench_dir/build/preprocessed_data" \
    --mode preprocess --num_proc 4 \
    2>&1 | tee "$target_dir/preprocess.log"

{
    date -Is
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
} | tee "$target_dir/gpu.log"

# Keep evaluation as a separate logged step while preserving run.py's requested command.
export SKIP_VERIFY_ACCURACY=1
export LOG_PATH="$bench_dir/build/logs"
"$venv_dir/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
    --model="$model_input" \
    --preprocessed_data_dir="$bench_dir/build/preprocessed_data" \
    2>&1 | tee "$target_dir/run.log"

"$venv_dir/bin/python" accuracy_kits.py \
    --log_file="$bench_dir/build/logs/mlperf_log_accuracy.json" \
    --preprocessed_data_dir="$bench_dir/build/preprocessed_data" \
    --postprocessed_data_dir="$bench_dir/build/postprocessed_data" \
    --num_proc 4 \
    2> >(tee "$target_dir/evaluator_stderr.log" >&2) \
    | tee "$target_dir/evaluator.log" "$output_dir/accuracy.txt"

cp "$bench_dir/build/logs/mlperf_log_accuracy.json" "$output_dir/mlperf_log_accuracy.json"
cp "$bench_dir/build/logs/mlperf_log_detail.txt" "$output_dir/mlperf_log_detail.txt"

for required in mlperf_log_accuracy.json mlperf_log_detail.txt accuracy.txt; do
    test -s "$output_dir/$required"
done
