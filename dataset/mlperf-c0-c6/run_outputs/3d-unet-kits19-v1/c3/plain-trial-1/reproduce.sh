#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --target <target id>" >&2
  exit 2
}

[[ ${1:-} == "--target" && $# -eq 2 ]] || usage
target=$2
[[ "$target" == "3d-unet" ]] || { echo "Unknown target: $target" >&2; exit 2; }

root=$(cd "$(dirname "$0")" && pwd)
repo="$root/repo"
target_dir="$root/$target"
output_dir=${OUTPUT_DIR:-"$target_dir/out"}
cache_root=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
raw_input="$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data"
model_input="$cache_root/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
uv_cache="$cache_root/uv"
work="$target_dir/work"
app="$work/3d-unet-kits19"
venv="$work/venv"

mkdir -p "$target_dir" "$output_dir" "$work"
output_dir=$(cd "$output_dir" && pwd)
exec > >(tee "$target_dir/reproduce.log") 2>&1

echo "target: $target"
echo "OUTPUT_DIR: $output_dir"
echo "Preparing private source tree"
rm -rf "$app"
cp -a "$repo/vision/medical_imaging/3d-unet-kits19" "$app"
mkdir -p "$app/build/raw_data" "$app/build/preprocessed_data" \
  "$app/build/postprocessed_data" "$app/build/model"
cp "$repo/loadgen/mlperf.conf" "$app/build/mlperf.conf"

# Keep raw inputs read-only. The official case_00400 is a documented duplicate of
# case_00185, so represent it with a second symlink in the private raw-data view.
while IFS= read -r case_name; do
  source_case=$case_name
  [[ "$case_name" == "case_00400" ]] && source_case=case_00185
  ln -sfn "$raw_input/$source_case" "$app/build/raw_data/$case_name"
done < <(/usr/bin/python3.11 -c 'import json; print("\n".join(json.load(open("'"$app"'/meta/inference_cases.json"))))')
ln -sfn "$model_input" "$app/build/model/3dunet_kits19_pytorch.ptc"

# This revision records self.device but does not print it. Add the requested
# runtime evidence immediately after the TorchScript model has been loaded.
sed -i '/self.model = torch.jit.load/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$app/pytorch_SUT.py"

export UV_CACHE_DIR="$uv_cache"
export UV_EXCLUDE_NEWER=2026-11-10
if [[ ! -x "$venv/bin/python" ]]; then
  echo "Creating Python environment"
  uv venv --python /usr/bin/python3.11 "$venv"
fi

echo "Installing pinned runtime and building LoadGen"
uv pip install --python "$venv/bin/python" \
  'numpy==1.26.4' 'scipy==1.10.1' 'pandas==1.5.3' 'nibabel==5.2.1' \
  'torch==2.1.2' 2>&1 | tee "$target_dir/build.log"
uv pip install --python "$venv/bin/python" 'mlcommons-loadgen==6.0.17' 2>&1 | tee -a "$target_dir/build.log"

echo "Preprocessing 43 KiTS19 accuracy cases"
(
  cd "$app"
  "$venv/bin/python" preprocess.py \
    --raw_data_dir build/raw_data --results_dir build/preprocessed_data \
    --mode preprocess
) 2>&1 | tee "$target_dir/preprocess.log"

echo "GPU memory immediately before inference:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader | tee "$target_dir/gpu-before-run.log"

echo "Running requested Offline PyTorch accuracy command"
(
  cd "$app"
  export LOG_PATH="$output_dir"
  export SKIP_VERIFY_ACCURACY=1
  export CUDA_VISIBLE_DEVICES=0
  "$venv/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
    --model="$model_input" \
    --preprocessed_data_dir=build/preprocessed_data \
    --mlperf_conf=build/mlperf.conf --user_conf=user.conf
) 2>&1 | tee "$target_dir/run.log"

echo "Evaluating LoadGen accuracy output"
(
  cd "$app"
  "$venv/bin/python" accuracy_kits.py \
    --log_file="$output_dir/mlperf_log_accuracy.json" \
    --preprocessed_data_dir=build/preprocessed_data \
    --postprocessed_data_dir=build/postprocessed_data
) 2>&1 | tee "$output_dir/accuracy.txt" "$target_dir/evaluator.log"

test -s "$output_dir/mlperf_log_accuracy.json"
test -s "$output_dir/mlperf_log_detail.txt"
test -s "$output_dir/accuracy.txt"
grep -q '^pytorch device: ' "$target_dir/run.log"
echo "Completed target $target"
