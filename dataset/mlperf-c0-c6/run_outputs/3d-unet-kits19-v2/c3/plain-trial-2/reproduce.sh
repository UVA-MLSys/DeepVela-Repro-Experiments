#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target 3d-unet" >&2
    exit 2
}

[[ $# -eq 2 && "$1" == "--target" ]] || usage
TARGET_ID=$2
[[ "$TARGET_ID" == "3d-unet" ]] || usage
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
TARGET_DIR="$SCRIPT_DIR/$TARGET_ID"
WORK_DIR="$TARGET_DIR/work"
SRC_DIR="$WORK_DIR/src"
BENCH_DIR="$SRC_DIR/vision/medical_imaging/3d-unet-kits19"
VENV_DIR="$WORK_DIR/venv"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$TARGET_DIR" "$WORK_DIR" "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd)
exec 3>&1

{
    echo "Preparing isolated working copy"
    test -f "$SCRIPT_DIR/repo/vision/medical_imaging/3d-unet-kits19/run.py"
    test -f "$MODEL_INPUT"
    test -d "$RAW_INPUT"
    mkdir -p "$SRC_DIR/vision/medical_imaging"
    if [[ ! -d "$BENCH_DIR" ]]; then
        cp -a "$SCRIPT_DIR/repo/vision/medical_imaging/3d-unet-kits19" "$BENCH_DIR"
    fi
    if [[ ! -d "$SRC_DIR/loadgen" ]]; then
        cp -a "$SCRIPT_DIR/repo/loadgen" "$SRC_DIR/loadgen"
    fi

    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        uv venv --python 3.11 "$VENV_DIR"
    fi
    uv pip install --python "$VENV_DIR/bin/python" --offline \
        torch numpy scipy nibabel pandas pybind11 setuptools wheel
    if ! "$VENV_DIR/bin/python" -c 'import mlperf_loadgen' 2>/dev/null; then
        uv pip install --python "$VENV_DIR/bin/python" --offline \
            --no-build-isolation "$SRC_DIR/loadgen"
    fi
    "$VENV_DIR/bin/python" - <<'PY'
import nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("nibabel", nibabel.__version__, "pandas", pandas.__version__)
PY
} 2>&1 | tee "$TARGET_DIR/build.log" >&3

# Compatibility adjustments are made only in the disposable working copy.
sed -i 's/signal\.gaussian(/signal.windows.gaussian(/' "$BENCH_DIR/inference_utils.py"
if ! grep -q '_reproduce_append_compat' "$BENCH_DIR/accuracy_kits.py"; then
    sed -i '/import pandas as pd/a\# _reproduce_append_compat\npd.DataFrame.append = lambda self, row, ignore_index=False: pd.concat([self, pd.DataFrame([row])], ignore_index=ignore_index)' "$BENCH_DIR/accuracy_kits.py"
fi
sed -i 's/float(final\["composite"\])/float(final["composite"].iloc[0])/; s/float(final\["kidney"\])/float(final["kidney"].iloc[0])/; s/float(final\["tumor"\])/float(final["tumor"].iloc[0])/' "$BENCH_DIR/accuracy_kits.py"

BUILD_DIR="$BENCH_DIR/build"
PREPROCESSED_DIR="$BUILD_DIR/preprocessed_data"
RAW_WORK="$BUILD_DIR/raw_data"
mkdir -p "$BUILD_DIR" "$RAW_WORK" "$BUILD_DIR/model" "$BUILD_DIR/logs" "$BUILD_DIR/postprocessed_data"
cp "$SRC_DIR/loadgen/mlperf.conf" "$BUILD_DIR/mlperf.conf"
ln -sfn "$MODEL_INPUT" "$BUILD_DIR/model/3dunet_kits19_pytorch.ptc"

while IFS= read -r case_name; do
    if [[ "$case_name" == "case_00400" ]]; then
        ln -sfn "$RAW_INPUT/case_00185" "$RAW_WORK/$case_name"
    else
        ln -sfn "$RAW_INPUT/$case_name" "$RAW_WORK/$case_name"
    fi
done < <("$VENV_DIR/bin/python" -c 'import json,sys; print(*json.load(open(sys.argv[1])), sep="\n")' "$BENCH_DIR/meta/inference_cases.json")

if [[ ! -f "$PREPROCESSED_DIR/preprocessed_files.pkl" ]]; then
    (
        cd "$BENCH_DIR"
        "$VENV_DIR/bin/python" -c 'import preprocess; args=preprocess.parse_args(); preprocess.preprocess_with_multiproc(args)' \
            --raw_data_dir "$RAW_WORK" \
            --results_dir "$PREPROCESSED_DIR" \
            --mode preprocess --num_proc 8
    ) 2>&1 | tee "$TARGET_DIR/preprocess.log" >&3
else
    echo "Reusing completed preprocessing at $PREPROCESSED_DIR" | tee "$TARGET_DIR/preprocess.log" >&3
fi

{
    date -Is
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
} 2>&1 | tee "$TARGET_DIR/gpu.log" >&3

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if [[ ! "$FREE_MIB" =~ ^[0-9]+$ || "$FREE_MIB" -lt 20000 ]]; then
    echo "Insufficient free GPU memory before inference: ${FREE_MIB:-unknown} MiB" | tee -a "$TARGET_DIR/gpu.log" >&2
    exit 1
fi

rm -f "$BUILD_DIR/logs"/mlperf_log_*.txt "$BUILD_DIR/logs/mlperf_log_accuracy.json"
rm -rf "$BUILD_DIR/postprocessed_data"
mkdir -p "$BUILD_DIR/postprocessed_data"

(
    cd "$BENCH_DIR"
    export LOG_PATH="$BUILD_DIR/logs"
    export SKIP_VERIFY_ACCURACY=1
    "$VENV_DIR/bin/python" - <<'PY'
import torch
model = torch.jit.load("build/model/3dunet_kits19_pytorch.ptc", map_location="cuda:0")
try:
    device = next(model.parameters()).device
except StopIteration:
    device = torch.device("cuda:0")
print(f"pytorch device: {device}")
del model
torch.cuda.empty_cache()
PY
    "$VENV_DIR/bin/python" run.py \
        --backend=pytorch --scenario=Offline --accuracy \
        --model="$BUILD_DIR/model/3dunet_kits19_pytorch.ptc" \
        --preprocessed_data_dir="$PREPROCESSED_DIR" \
        --mlperf_conf="$BUILD_DIR/mlperf.conf" \
        --user_conf="$BENCH_DIR/user.conf"
) 2>&1 | tee "$TARGET_DIR/run.log" >&3

cp "$BUILD_DIR/logs/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
cp "$BUILD_DIR/logs/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"

(
    cd "$BENCH_DIR"
    PYTHONWARNINGS=ignore "$VENV_DIR/bin/python" accuracy_kits.py \
        --preprocessed_data_dir="$PREPROCESSED_DIR" \
        --postprocessed_data_dir="$BUILD_DIR/postprocessed_data" \
        --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
        --num_proc=8
) 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$TARGET_DIR/evaluator.log" >&3

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
grep -q '^pytorch device: ' "$TARGET_DIR/run.log"
