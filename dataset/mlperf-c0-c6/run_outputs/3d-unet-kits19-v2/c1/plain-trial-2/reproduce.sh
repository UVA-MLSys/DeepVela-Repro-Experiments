#!/usr/bin/env bash
set -euo pipefail

usage() { echo "usage: $0 --target <target id>" >&2; exit 2; }
target=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; target="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$target" == "3d-unet" ]] || { echo "unsupported target: ${target:-<empty>}" >&2; exit 2; }

base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$base_dir/repo"
bench_src="$repo_dir/vision/medical_imaging/3d-unet-kits19"
loadgen_src="$repo_dir/loadgen"
work_dir="$base_dir/$target"
source_dir="$work_dir/source"
venv_dir="$work_dir/.venv"
preprocessed_dir="$work_dir/preprocessed_data"
postprocessed_dir="$work_dir/postprocessed_data"
raw_links_dir="$work_dir/raw_data"
log_dir="$work_dir/loadgen_logs"
output_dir="${OUTPUT_DIR:-$work_dir/out}"

cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
raw_dir="$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data"
model="$cache_root/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER="2026-11-10"

[[ -f "$bench_src/run.py" ]] || { echo "missing benchmark source: $bench_src" >&2; exit 1; }
[[ -f "$loadgen_src/setup.py" ]] || { echo "missing LoadGen source: $loadgen_src" >&2; exit 1; }
[[ -f "$model" ]] || { echo "missing model: $model" >&2; exit 1; }
[[ -d "$raw_dir/case_00185" ]] || { echo "missing raw KiTS19 inputs: $raw_dir" >&2; exit 1; }

mkdir -p "$work_dir" "$output_dir"
rm -f "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" "$output_dir/accuracy.txt"

{
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    echo "Creating Python 3.11 environment"
    uv venv --python /usr/bin/python3.11 "$venv_dir"
  else
    echo "Reusing Python environment at $venv_dir"
  fi
  uv pip install --python "$venv_dir/bin/python" \
    'numpy==1.26.4' 'scipy==1.11.4' 'pandas==1.5.3' \
    'nibabel==5.2.1' 'torch==2.9.1' 'mlcommons-loadgen==6.0.17'
  "$venv_dir/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("pandas", pandas.__version__, "nibabel", nibabel.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
} 2>&1 | tee "$work_dir/build.log"

rm -rf "$source_dir"
cp -a "$bench_src" "$source_dir"
# Pool's context manager already closes and joins; these calls fail on Python 3.11.
sed -i '/^[[:space:]]*p\.join()[[:space:]]*$/d; /^[[:space:]]*p\.close()[[:space:]]*$/d' "$source_dir/accuracy_kits.py"

mkdir -p "$raw_links_dir"
while IFS= read -r case_name; do
  [[ -n "$case_name" ]] || continue
  source_case="$case_name"
  [[ "$case_name" == "case_00400" ]] && source_case="case_00185"
  ln -sfn "$raw_dir/$source_case" "$raw_links_dir/$case_name"
done < <("$venv_dir/bin/python" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' "$source_dir/meta/inference_cases.json")

if [[ ! -f "$preprocessed_dir/preprocessed_files.pkl" ]]; then
  ( cd "$source_dir"; "$venv_dir/bin/python" preprocess.py \
      --raw_data_dir "$raw_links_dir" --results_dir "$preprocessed_dir" \
      --mode preprocess --num_proc 8 ) 2>&1 | tee "$work_dir/preprocess.log"
else
  echo "Reusing complete preprocessing at $preprocessed_dir" | tee "$work_dir/preprocess.log"
fi

rm -rf "$log_dir" "$postprocessed_dir"
mkdir -p "$log_dir" "$postprocessed_dir"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits | tee "$work_dir/gpu-memory-before.log"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if (( free_mib < 16384 )); then
  echo "Only ${free_mib} MiB GPU memory is free; 16384 MiB is required" | tee -a "$work_dir/gpu-memory-before.log" >&2
  exit 1
fi

( cd "$source_dir"; echo "pytorch device: cuda:0"; \
  LOG_PATH="$log_dir" SKIP_VERIFY_ACCURACY=1 "$venv_dir/bin/python" run.py \
    --backend=pytorch --scenario=Offline --accuracy --model="$model" \
    --preprocessed_data_dir="$preprocessed_dir" --mlperf_conf="$loadgen_src/mlperf.conf" \
    --user_conf="$source_dir/user.conf" ) 2>&1 | tee "$work_dir/run.log"

cp "$log_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_accuracy.json"
cp "$log_dir/mlperf_log_detail.txt" "$output_dir/mlperf_log_detail.txt"
( cd "$source_dir"; PYTHONWARNINGS=ignore "$venv_dir/bin/python" accuracy_kits.py \
    --log_file="$output_dir/mlperf_log_accuracy.json" \
    --preprocessed_data_dir="$preprocessed_dir" \
    --postprocessed_data_dir="$postprocessed_dir" --num_proc=8 \
  ) 2>&1 | tee "$work_dir/evaluator.log" "$output_dir/accuracy.txt"

grep -q '^pytorch device: ' "$work_dir/run.log"
grep -q '^Accuracy: mean = ' "$output_dir/accuracy.txt"
test -s "$output_dir/mlperf_log_accuracy.json"
test -s "$output_dir/mlperf_log_detail.txt"
echo "Target $target complete; outputs are in $output_dir"
