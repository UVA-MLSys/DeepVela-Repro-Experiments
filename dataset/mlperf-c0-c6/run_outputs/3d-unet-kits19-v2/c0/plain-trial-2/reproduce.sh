#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != "--target" || ${2:-} != "3d-unet" || $# -ne 2 ]]; then
  echo "usage: $0 --target 3d-unet" >&2
  exit 2
fi

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
TARGET_ID=$2
WORK_DIR="$ROOT_DIR/$TARGET_ID"
OUT_DIR=${OUTPUT_DIR:-"$WORK_DIR/out"}
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_SOURCE="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_SOURCE="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$WORK_DIR" "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd)
exec > >(tee "$WORK_DIR/reproduce.log") 2>&1

echo "target: $TARGET_ID"
echo "work directory: $WORK_DIR"
echo "output directory: $OUT_DIR"

if [[ ! -f "$MODEL_SOURCE" || ! -d "$RAW_SOURCE" ]]; then
  echo "Required cached model or KiTS19 raw data is missing" >&2
  exit 1
fi

if [[ ! -x "$WORK_DIR/.venv/bin/python" ]]; then
  uv venv --python /usr/bin/python3.11 "$WORK_DIR/.venv"
fi

if [[ ! -f "$WORK_DIR/.environment-ready" ]]; then
  {
    uv pip install --python "$WORK_DIR/.venv/bin/python" \
      'numpy==1.26.4' 'scipy==1.10.1' 'pandas==1.5.3' \
      'nibabel==5.2.1' 'torch==2.4.1' 'pybind11==2.11.1' \
      'setuptools==70.3.0' 'wheel==0.44.0'
    uv pip install --python "$WORK_DIR/.venv/bin/python" 'mlcommons-loadgen==6.0.17'
  } 2>&1 | tee "$WORK_DIR/build.log"
  touch "$WORK_DIR/.environment-ready"
fi

BENCH_DIR="$WORK_DIR/benchmark"
if [[ ! -f "$BENCH_DIR/.prepared" ]]; then
  rm -rf "$BENCH_DIR"
  cp -a "$ROOT_DIR/repo/vision/medical_imaging/3d-unet-kits19" "$BENCH_DIR"
  mkdir -p "$BENCH_DIR/build" "$WORK_DIR/raw_data"
  cp "$ROOT_DIR/repo/loadgen/mlperf.conf" "$BENCH_DIR/build/mlperf.conf"
  sed -i '/self.model = torch.jit.load/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
    "$BENCH_DIR/pytorch_SUT.py"
  touch "$BENCH_DIR/.prepared"
fi

while IFS= read -r case_name; do
  if [[ "$case_name" == "case_00400" ]]; then
    source_case=case_00185
  else
    source_case=$case_name
  fi
  ln -sfn "$RAW_SOURCE/$source_case" "$WORK_DIR/raw_data/$case_name"
done < <("$WORK_DIR/.venv/bin/python" -c 'import json; print(*json.load(open("'"$BENCH_DIR/meta/inference_cases.json"'")), sep="\n")')

cd "$BENCH_DIR"

if [[ ! -f "$WORK_DIR/preprocessed/preprocessed_files.pkl" ]]; then
  mkdir -p "$WORK_DIR/preprocessed"
  "$WORK_DIR/.venv/bin/python" preprocess.py \
    --raw_data_dir "$WORK_DIR/raw_data" \
    --results_dir "$WORK_DIR/preprocessed" \
    --mode preprocess --num_proc "${PREPROCESS_JOBS:-8}" \
    2>&1 | tee "$WORK_DIR/preprocess.log"
fi

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
echo "GPU free memory before run: ${free_mib} MiB"
if (( free_mib < 24000 )); then
  echo "Insufficient free GPU memory: need at least 24000 MiB" >&2
  exit 1
fi

rm -f "$OUT_DIR/mlperf_log_accuracy.json" "$OUT_DIR/mlperf_log_detail.txt" \
  "$OUT_DIR/mlperf_log_summary.txt" "$OUT_DIR/accuracy.txt"
rm -rf "$WORK_DIR/postprocessed"
mkdir -p "$WORK_DIR/postprocessed"

export LOG_PATH="$OUT_DIR"
export SKIP_VERIFY_ACCURACY=1
"$WORK_DIR/.venv/bin/python" run.py \
  --backend=pytorch --scenario=Offline --accuracy \
  --model="$MODEL_SOURCE" \
  --preprocessed_data_dir="$WORK_DIR/preprocessed" \
  --mlperf_conf="$BENCH_DIR/build/mlperf.conf" \
  --user_conf="$BENCH_DIR/user.conf" \
  2>&1 | tee "$WORK_DIR/run.log"

PYTHONWARNINGS='ignore::FutureWarning' "$WORK_DIR/.venv/bin/python" accuracy_kits.py \
  --log_file="$OUT_DIR/mlperf_log_accuracy.json" \
  --preprocessed_data_dir="$WORK_DIR/preprocessed" \
  --postprocessed_data_dir="$WORK_DIR/postprocessed" \
  --num_proc "${EVALUATOR_JOBS:-8}" \
  2>"$WORK_DIR/evaluator_stderr.log" \
  | tee "$OUT_DIR/accuracy.txt" "$WORK_DIR/evaluator.log"

test -s "$OUT_DIR/mlperf_log_accuracy.json"
test -s "$OUT_DIR/mlperf_log_detail.txt"
test -s "$OUT_DIR/accuracy.txt"
grep -F 'pytorch device: ' "$WORK_DIR/run.log"
grep -E 'Accuracy: mean = [0-9.]+, kidney = [0-9.]+, tumor = [0-9.]+' "$OUT_DIR/accuracy.txt"
