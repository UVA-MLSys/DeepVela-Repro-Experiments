#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != "--target" || ${2:-} != "3d-unet" || $# -ne 2 ]]; then
  echo "usage: $0 --target 3d-unet" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "$0")" && pwd)
TARGET="$ROOT/3d-unet"
WORK="$TARGET/work"
APP="$WORK/app"
VENV="$WORK/.venv"
OUT=${OUTPUT_DIR:-"$TARGET/out"}
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_SOURCE="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$TARGET" "$WORK" "$OUT"
OUT=$(cd "$OUT" && pwd)

rm -rf "$APP" "$WORK/loadgen" "$WORK/raw_data" "$WORK/preprocessed_data" "$WORK/postprocessed_data"
cp -a "$ROOT/repo/vision/medical_imaging/3d-unet-kits19" "$APP"
cp -a "$ROOT/repo/loadgen" "$WORK/loadgen"
mkdir -p "$WORK/raw_data" "$WORK/preprocessed_data" "$WORK/postprocessed_data"
ln -s "$RAW_SOURCE"/case_* "$WORK/raw_data/"
ln -s "$RAW_SOURCE/case_00185" "$WORK/raw_data/case_00400"

{
  uv venv --python 3.11 "$VENV"
  uv pip install --python "$VENV/bin/python" --offline \
    'torch==2.4.1' 'numpy==1.26.4' 'scipy==1.10.1' \
    'nibabel==5.3.2' 'pandas==3.0.6'
} 2>&1 | tee "$TARGET/build.log"

# This commit predates pandas 2 and does not print the required device evidence.
sed -i '/import pandas as pd/a\
if not hasattr(pd.DataFrame, "append"):\
    pd.DataFrame.append = lambda self, other, ignore_index=False: pd.concat([self, pd.DataFrame([other])], ignore_index=ignore_index)' \
  "$APP/accuracy_kits.py"
sed -i 's/float(final\["composite"\])/float(final["composite"].iloc[0])/; s/float(final\["kidney"\])/float(final["kidney"].iloc[0])/; s/float(final\["tumor"\])/float(final["tumor"].iloc[0])/' \
  "$APP/accuracy_kits.py"
sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
  "$APP/pytorch_SUT.py"

(
  cd "$APP"
  "$VENV/bin/python" preprocess.py \
    --raw_data_dir="$WORK/raw_data" \
    --results_dir="$WORK/preprocessed_data" \
    --mode=preprocess --num_proc=16
) > "$TARGET/preprocess.log" 2>&1 &
PREPROCESS_PID=$!

set +e
uv pip install --python "$VENV/bin/python" --offline "$WORK/loadgen" \
  2>&1 | tee -a "$TARGET/build.log"
BUILD_STATUS=${PIPESTATUS[0]}
set -e
wait "$PREPROCESS_PID"
if [[ $BUILD_STATUS -ne 0 ]]; then
  exit "$BUILD_STATUS"
fi

{
  echo "GPU memory immediately before run:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used \
    --format=csv,noheader
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if (( FREE_MIB < 16000 )); then
    echo "Insufficient free GPU memory: ${FREE_MIB} MiB (need at least 16000 MiB)" >&2
    exit 1
  fi
  echo "python3 run.py --backend=pytorch --scenario=Offline --accuracy"
  cd "$APP"
  LOG_PATH="$OUT" SKIP_VERIFY_ACCURACY=1 "$VENV/bin/python" run.py \
    --backend=pytorch --scenario=Offline --accuracy \
    --model="$MODEL" \
    --preprocessed_data_dir="$WORK/preprocessed_data" \
    --mlperf_conf="$WORK/loadgen/mlperf.conf" \
    --user_conf="$APP/user.conf"
} 2>&1 | tee "$TARGET/run.log"

(
  cd "$APP"
  "$VENV/bin/python" accuracy_kits.py \
    --log_file="$OUT/mlperf_log_accuracy.json" \
    --preprocessed_data_dir="$WORK/preprocessed_data" \
    --postprocessed_data_dir="$WORK/postprocessed_data" \
    --num_proc=16
) 2>&1 | tee "$OUT/accuracy.txt" "$TARGET/evaluator.log"

test -s "$OUT/mlperf_log_accuracy.json"
test -s "$OUT/mlperf_log_detail.txt"
test -s "$OUT/accuracy.txt"
