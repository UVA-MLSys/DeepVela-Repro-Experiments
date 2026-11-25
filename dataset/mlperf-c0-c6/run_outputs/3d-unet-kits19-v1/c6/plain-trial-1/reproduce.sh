#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != "--target" || ${2:-} != "3d-unet" || $# -ne 2 ]]; then
  echo "usage: $0 --target 3d-unet" >&2
  exit 2
fi
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

ROOT=$(cd "$(dirname "$0")" && pwd)
WORK="$ROOT/3d-unet"
SRC="$WORK/repo"
APP="$SRC/vision/medical_imaging/3d-unet-kits19"
VENV="$WORK/.venv-c6"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
PYTHON311=$(command -v python3.11)

mkdir -p "$WORK" "$OUTPUT_DIR"
if [[ ! -d "$SRC" ]]; then
  cp -a "$ROOT/repo" "$SRC"
fi

{
  date -Is
  uv venv --clear --python "$PYTHON311" "$VENV"
  uv pip install --offline --python "$VENV/bin/python" \
    'torch==2.9.1' 'numpy==1.26.4' 'scipy==1.12.0' \
    'nibabel==5.3.2' 'pandas==1.5.3' 'pybind11==3.0.1' setuptools wheel
  uv pip install --offline --no-build-isolation --python "$VENV/bin/python" "$SRC/loadgen"
  "$VENV/bin/python" -c 'import torch,numpy,scipy,nibabel,pandas,mlperf_loadgen; print("torch",torch.__version__,"cuda",torch.version.cuda); print("numpy",numpy.__version__,"scipy",scipy.__version__,"nibabel",nibabel.__version__,"pandas",pandas.__version__); print("loadgen",mlperf_loadgen.__file__)'
} >"$WORK/build-c6.log" 2>&1

mkdir -p "$APP/build/raw_data" "$APP/build/model" "$APP/build/preprocessed_data" "$APP/build/logs" "$APP/build/postprocessed_data"
while IFS= read -r case_name; do
  ln -sfn "$RAW/$case_name" "$APP/build/raw_data/$case_name"
done < <("$VENV/bin/python" -c 'import json,sys; print(*json.load(open(sys.argv[1])), sep="\n")' "$APP/meta/inference_cases.json")
# case_00400 is the documented duplicate of case_00185 and is absent from the read-only input.
ln -sfn "$RAW/case_00185" "$APP/build/raw_data/case_00400"
ln -sfn "$MODEL" "$APP/build/model/3dunet_kits19_pytorch.ptc"
cp "$SRC/loadgen/mlperf.conf" "$APP/build/mlperf.conf"

cd "$APP"
if [[ ! -s build/preprocessed_data/preprocessed_files.pkl ]]; then
  "$VENV/bin/python" preprocess.py \
    --raw_data_dir build/raw_data --results_dir build/preprocessed_data \
    --mode preprocess --num_proc 4 >"$WORK/preprocess-c6.log" 2>&1
else
  echo "Using complete existing preprocessing output" >"$WORK/preprocess-c6.log"
fi

{
  date -Is
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
} >"$WORK/gpu-memory-c6.log"
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if (( free_mib < 20480 )); then
  echo "Insufficient free GPU memory: ${free_mib} MiB (need 20480 MiB)" | tee -a "$WORK/gpu-memory-c6.log" >&2
  exit 1
fi

export LOG_PATH="$APP/build/logs"
export SKIP_VERIFY_ACCURACY=1
{
  echo "pytorch device: cuda:0"
  "$VENV/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
    --model="$MODEL" --preprocessed_data_dir=build/preprocessed_data \
    --mlperf_conf=build/mlperf.conf --user_conf=user.conf
} >"$WORK/run-c6.log" 2>&1

"$VENV/bin/python" accuracy_kits.py \
  --preprocessed_data_dir=build/preprocessed_data \
  --postprocessed_data_dir=build/postprocessed_data \
  --log_file=build/logs/mlperf_log_accuracy.json \
  --num_proc=4 >"$WORK/evaluator-c6.log" 2>"$WORK/evaluator-stderr-c6.log"

cp build/logs/mlperf_log_accuracy.json "$OUTPUT_DIR/mlperf_log_accuracy.json"
cp build/logs/mlperf_log_detail.txt "$OUTPUT_DIR/mlperf_log_detail.txt"
cp "$WORK/evaluator-c6.log" "$OUTPUT_DIR/accuracy.txt"
