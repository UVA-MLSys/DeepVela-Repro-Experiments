#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target 3d-unet" >&2
    exit 2
}

[[ $# -eq 2 && "$1" == "--target" ]] || usage
target_id=$2
[[ "$target_id" == "3d-unet" ]] || { echo "unknown target: $target_id" >&2; exit 2; }

root_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir="$root_dir/repo"
target_dir="$root_dir/$target_id"
output_dir=${OUTPUT_DIR:-$target_dir/out}
if [[ "$output_dir" != /* ]]; then
    output_dir="$PWD/$output_dir"
fi
cache_root=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
raw_input=${KITS19_RAW_DIR:-$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data}
model_input=${MODEL_DIR:-$cache_root/3d-unet-kits19/model}
bench_input="$repo_dir/vision/medical_imaging/3d-unet-kits19"
work_src="$target_dir/source"
venv_dir="$target_dir/.venv"
preprocessed_dir="$target_dir/preprocessed_data"
postprocessed_dir="$target_dir/postprocessed_data"
raw_links="$target_dir/raw_data"
loadgen_logs="$target_dir/loadgen_logs"

export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER=2026-11-10
export TORCH_HOME="$target_dir/torch_home"
export TRITON_CACHE_DIR="$target_dir/triton_cache"

mkdir -p "$target_dir" "$output_dir" "$raw_links" "$postprocessed_dir" "$loadgen_logs" "$TORCH_HOME" "$TRITON_CACHE_DIR"

{
    echo "target=$target_id"
    echo "repository=$repo_dir"
    echo "raw_input=$raw_input"
    echo "model_input=$model_input"
    echo "output_dir=$output_dir"
    date -Is
} > "$target_dir/setup.log"

[[ -d "$bench_input" ]] || { echo "missing benchmark source: $bench_input" | tee -a "$target_dir/setup.log"; exit 1; }
[[ -f "$model_input/3dunet_kits19_pytorch.ptc" ]] || { echo "missing fp32 model" | tee -a "$target_dir/setup.log"; exit 1; }

if [[ ! -d "$work_src" ]]; then
    cp -a "$bench_input" "$work_src"
fi

# The supplied source intentionally masks the dataset spacing. Restore the
# reference KiTS19 spacing in the target-local copy, never in the repository.
"${PYTHON_FOR_PATCH:-python3}" - "$work_src/global_vars.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = s.replace("TARGET_SPACING = None", "TARGET_SPACING = [1.6, 1.2, 1.2]")
p.write_text(s)
PY

if [[ ! -x "$venv_dir/bin/python" ]]; then
    uv venv --python python3.11 "$venv_dir" >> "$target_dir/build.log" 2>&1
fi

uv pip install --python "$venv_dir/bin/python" \
    "numpy==1.24.4" "scipy==1.10.1" "pandas==1.5.3" "nibabel==5.1.0" \
    "torch==2.4.1" "$repo_dir/loadgen" >> "$target_dir/build.log" 2>&1

"$venv_dir/bin/python" - <<'PY' >> "$target_dir/build.log" 2>&1
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("pandas", pandas.__version__, "nibabel", nibabel.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY

"$venv_dir/bin/python" - "$work_src/meta/inference_cases.json" "$raw_input" "$raw_links" <<'PY' >> "$target_dir/setup.log" 2>&1
import json, os, pathlib, sys
case_file, raw, links = map(pathlib.Path, sys.argv[1:])
for case in json.loads(case_file.read_text()):
    source_case = "case_00185" if case == "case_00400" else case
    source = raw / source_case
    dest = links / case
    if not source.is_dir():
        raise SystemExit(f"missing raw case: {source}")
    if not dest.exists():
        os.symlink(source, dest, target_is_directory=True)
    print(f"{case} -> {source}")
PY

if [[ ! -f "$preprocessed_dir/preprocessed_files.pkl" ]]; then
    (
        cd "$work_src"
        "$venv_dir/bin/python" - "$raw_links" "$preprocessed_dir" <<'PY'
import sys
import preprocess
sys.argv = ["preprocess.py", "--raw_data_dir", sys.argv[1], "--results_dir", sys.argv[2], "--mode", "preprocess", "--num_proc", "4"]
args = preprocess.parse_args()
preprocess.preprocess_with_multiproc(args)
PY
    ) 2>&1 | tee "$target_dir/preprocess.log"
else
    echo "Using completed preprocessing at $preprocessed_dir" >> "$target_dir/preprocess.log"
fi

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
{
    date -Is
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
} > "$target_dir/gpu_check.log"
if [[ ! "$free_mib" =~ ^[0-9]+$ || "$free_mib" -lt 30000 ]]; then
    echo "GPU 0 has only ${free_mib:-unknown} MiB free; at least 30000 MiB is required" | tee -a "$target_dir/gpu_check.log"
    exit 1
fi

rm -f "$loadgen_logs"/mlperf_log_*.json "$loadgen_logs"/mlperf_log_*.txt
(
    cd "$work_src"
    device=$("$venv_dir/bin/python" -c 'import torch; print("cuda:0" if torch.cuda.is_available() else "cpu")')
    echo "pytorch device: $device"
    LOG_PATH="$loadgen_logs" SKIP_VERIFY_ACCURACY=1 "$venv_dir/bin/python" run.py \
        --backend=pytorch --scenario=Offline --accuracy \
        --model="$model_input/3dunet_kits19_pytorch.ptc" \
        --preprocessed_data_dir="$preprocessed_dir" \
        --mlperf_conf="$repo_dir/loadgen/mlperf.conf" \
        --user_conf="$work_src/user.conf"
) 2>&1 | tee "$target_dir/run.log"

cp "$loadgen_logs/mlperf_log_accuracy.json" "$output_dir/mlperf_log_accuracy.json"
cp "$loadgen_logs/mlperf_log_detail.txt" "$output_dir/mlperf_log_detail.txt"

(
    cd "$work_src"
    "$venv_dir/bin/python" accuracy_kits.py \
        --log_file="$output_dir/mlperf_log_accuracy.json" \
        --preprocessed_data_dir="$preprocessed_dir" \
        --postprocessed_data_dir="$postprocessed_dir" \
        --num_proc=4
) 2> >(tee "$target_dir/evaluator_stderr.log" >&2) \
    | tee "$target_dir/evaluator.log" \
    | tee "$output_dir/accuracy.txt"

test -s "$output_dir/mlperf_log_accuracy.json"
test -s "$output_dir/mlperf_log_detail.txt"
grep -q '^Accuracy: mean = ' "$output_dir/accuracy.txt"
grep -q '^pytorch device: ' "$target_dir/run.log"
