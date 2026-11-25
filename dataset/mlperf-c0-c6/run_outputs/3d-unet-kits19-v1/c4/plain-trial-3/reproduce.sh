#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target 3d-unet" >&2
    exit 2
}

[[ $# -eq 2 && "$1" == "--target" ]] || usage
TARGET_ID=$2
[[ "$TARGET_ID" == "3d-unet" ]] || usage

ROOT=$(cd "$(dirname "$0")" && pwd)
REPO="$ROOT/repo"
TARGET_DIR="$ROOT/$TARGET_ID"
WORK="$TARGET_DIR/work"
OUTPUT_DIR=${OUTPUT_DIR:-"$TARGET_DIR/out"}
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
fi
CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
KITS19_RAW_DIR=${KITS19_RAW_DIR:-"$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"}
MODEL_DIR=${MODEL_DIR:-"$CACHE_ROOT/3d-unet-kits19/model"}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$CACHE_ROOT/uv"}
export UV_EXCLUDE_NEWER=2026-11-10

[[ -f "$REPO/vision/medical_imaging/3d-unet-kits19/run.py" ]] || {
    echo "missing repository beside reproduce.sh: $REPO" >&2
    exit 1
}
[[ -f "$MODEL_DIR/3dunet_kits19_pytorch.ptc" ]] || {
    echo "missing model: $MODEL_DIR/3dunet_kits19_pytorch.ptc" >&2
    exit 1
}
[[ -d "$KITS19_RAW_DIR/case_00185" ]] || {
    echo "missing KiTS19 inputs: $KITS19_RAW_DIR" >&2
    exit 1
}

mkdir -p "$TARGET_DIR" "$WORK" "$OUTPUT_DIR"

if [[ ! -d "$WORK/benchmark" ]]; then
    cp -a "$REPO/vision/medical_imaging/3d-unet-kits19" "$WORK/benchmark"
fi
if [[ ! -d "$WORK/loadgen" ]]; then
    cp -a "$REPO/loadgen" "$WORK/loadgen"
fi

# The task requires this device line; the upstream backend selects CUDA but does
# not report where the loaded model parameters reside.
if ! grep -q 'pytorch device:' "$WORK/benchmark/pytorch_SUT.py"; then
    sed -i '/self\.model\.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
        "$WORK/benchmark/pytorch_SUT.py"
fi
# This source revision reads a summary row named "mean" but omits creating it.
if ! grep -q 'df.loc\["mean"\]' "$WORK/benchmark/accuracy_kits.py"; then
    sed -i '/df\.set_index("case", inplace=True)/a\    df.loc["mean"] = df.mean()' \
        "$WORK/benchmark/accuracy_kits.py"
fi

{
    uv venv --python python3.11 "$WORK/.venv"
    uv pip install --python "$WORK/.venv/bin/python" \
        'numpy==1.26.4' 'scipy==1.12.0' 'pandas==1.5.3' \
        'nibabel==5.4.2' 'torch==2.15.0' 'pybind11==3.1.0' \
        'setuptools==84.2.1' 'wheel==0.48.0'
    uv pip install --python "$WORK/.venv/bin/python" --no-build-isolation "$WORK/loadgen"
    "$WORK/.venv/bin/python" -c \
        'import torch, mlperf_loadgen; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
} 2>&1 | tee "$TARGET_DIR/build.log"

mkdir -p "$WORK/raw_data"
while IFS= read -r case_dir; do
    case_name=$(basename "$case_dir")
    [[ -e "$WORK/raw_data/$case_name" ]] || ln -s "$case_dir" "$WORK/raw_data/$case_name"
done < <(find "$KITS19_RAW_DIR" -mindepth 1 -maxdepth 1 -type d -name 'case_*' | sort)
[[ -e "$WORK/raw_data/case_00400" ]] || ln -s "$KITS19_RAW_DIR/case_00185" "$WORK/raw_data/case_00400"

if [[ ! -f "$WORK/preprocessed_data/preprocessed_files.pkl" ]]; then
    (
        cd "$WORK/benchmark"
        ../.venv/bin/python preprocess.py \
            --raw_data_dir ../raw_data \
            --results_dir ../preprocessed_data \
            --mode preprocess --num_proc 6
    ) 2>&1 | tee "$TARGET_DIR/preprocess.log"
else
    echo "Reusing complete preprocessing in $WORK/preprocessed_data" | tee "$TARGET_DIR/preprocess.log"
fi

rm -rf "$WORK/loadgen_logs" "$WORK/postprocessed_data"
mkdir -p "$WORK/loadgen_logs" "$WORK/postprocessed_data"

# This implementation needs substantially less than 12 GiB, but refusing to
# launch below that margin avoids colliding with another session on the A100.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits | tee "$TARGET_DIR/gpu-memory.log"
if [[ ! "$FREE_MIB" =~ ^[0-9]+$ || "$FREE_MIB" -lt 12288 ]]; then
    echo "insufficient free GPU memory immediately before inference: ${FREE_MIB:-unknown} MiB" | tee -a "$TARGET_DIR/run.log" >&2
    exit 1
fi

(
    cd "$WORK/benchmark"
    LOG_PATH="$WORK/loadgen_logs" SKIP_VERIFY_ACCURACY=1 \
        ../.venv/bin/python run.py \
        --backend=pytorch --scenario=Offline --accuracy \
        --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
        --preprocessed_data_dir="$WORK/preprocessed_data" \
        --mlperf_conf="$WORK/loadgen/mlperf.conf" \
        --user_conf="$WORK/benchmark/user.conf"
) 2>&1 | tee "$TARGET_DIR/run.log"

cp "$WORK/loadgen_logs/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
cp "$WORK/loadgen_logs/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"

(
    cd "$WORK/benchmark"
    ../.venv/bin/python accuracy_kits.py \
        --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
        --preprocessed_data_dir="$WORK/preprocessed_data" \
        --postprocessed_data_dir="$WORK/postprocessed_data" \
        --num_proc=6
) > >(tee "$OUTPUT_DIR/accuracy.txt" "$TARGET_DIR/evaluator.log") \
  2> >(tee "$TARGET_DIR/evaluator-stderr.log" >&2)
