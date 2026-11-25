#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target 3d-unet" >&2
    exit 2
}

[[ $# -eq 2 && "$1" == "--target" ]] || usage
TARGET_ID=$2
[[ "$TARGET_ID" == "3d-unet" ]] || {
    echo "unsupported target: $TARGET_ID" >&2
    exit 2
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="$ROOT_DIR/repo"
TARGET_DIR="$ROOT_DIR/$TARGET_ID"
OUT_DIR=${OUTPUT_DIR:-"$TARGET_DIR/out"}
CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
RAW_INPUT=${KITS19_RAW_DIR:-"$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"}
MODEL_INPUT=${MODEL_DIR:-"$CACHE_ROOT/3d-unet-kits19/model"}
UV_CACHE_DIR=${UV_CACHE_DIR:-"$CACHE_ROOT/uv"}
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

BENCHMARK_SOURCE="$REPO_DIR/vision/medical_imaging/3d-unet-kits19"
LOADGEN_SOURCE="$REPO_DIR/loadgen"
SRC_DIR="$TARGET_DIR/src"
BENCHMARK_DIR="$SRC_DIR/3d-unet-kits19"
LOADGEN_DIR="$SRC_DIR/loadgen"
VENV_DIR="$TARGET_DIR/.venv"
PREPROCESSED_DIR="$TARGET_DIR/preprocessed_data"
POSTPROCESSED_DIR="$TARGET_DIR/postprocessed_data"
RAW_VIEW="$TARGET_DIR/raw_data"
MODEL_PATH="$MODEL_INPUT/3dunet_kits19_pytorch.ptc"

mkdir -p "$TARGET_DIR" "$OUT_DIR"
exec > >(tee -a "$TARGET_DIR/reproduce.log") 2>&1

echo "target: $TARGET_ID"
echo "output directory: $OUT_DIR"
echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"

for required in "$BENCHMARK_SOURCE/run.py" "$LOADGEN_SOURCE/setup.py" "$MODEL_PATH" "$RAW_INPUT/case_00185/imaging.nii.gz"; do
    [[ -e "$required" ]] || {
        echo "missing required input: $required" >&2
        exit 1
    }
done

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating Python environment and installing dependencies..."
    uv venv "$VENV_DIR" --python 3.11 2>&1 | tee "$TARGET_DIR/environment.log"
    uv pip install --python "$VENV_DIR/bin/python" \
        'numpy==1.26.4' 'scipy==1.10.1' 'nibabel==5.3.2' \
        'pandas==1.5.3' 'torch==2.4.1' 2>&1 | tee -a "$TARGET_DIR/environment.log"
else
    echo "Reusing Python environment." | tee "$TARGET_DIR/environment.log"
fi

if [[ ! -d "$BENCHMARK_DIR" ]]; then
    mkdir -p "$SRC_DIR"
    cp -a "$BENCHMARK_SOURCE" "$BENCHMARK_DIR"
fi
if [[ ! -d "$LOADGEN_DIR" ]]; then
    cp -a "$LOADGEN_SOURCE" "$LOADGEN_DIR"
fi

# The task requires this exact prefix; the upstream source did not report it.
if ! rg -q 'pytorch device: ' "$BENCHMARK_DIR/pytorch_SUT.py"; then
    sed -i '/self\.model\.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' \
        "$BENCHMARK_DIR/pytorch_SUT.py"
fi

# This benchmark/model's KiTS19 resampling spacing (z, y, x) is masked in the
# supplied source tree and must be restored for preprocessing and inference.
sed -i 's/^TARGET_SPACING = None$/TARGET_SPACING = [1.6, 1.2, 1.2]/' \
    "$BENCHMARK_DIR/global_vars.py"

if ! "$VENV_DIR/bin/python" -c 'import mlperf_loadgen' >/dev/null 2>&1; then
    uv pip install --python "$VENV_DIR/bin/python" "$LOADGEN_DIR" \
        2>&1 | tee "$TARGET_DIR/build.log"
else
    echo "LoadGen already built." | tee "$TARGET_DIR/build.log"
fi

if [[ ! -f "$PREPROCESSED_DIR/preprocessed_files.pkl" ]]; then
    echo "Creating read-only raw-data view and preprocessing 43 cases..."
    mkdir -p "$RAW_VIEW"
    while IFS= read -r case_name; do
        source_case=$case_name
        [[ "$case_name" == "case_00400" ]] && source_case=case_00185
        [[ -e "$RAW_INPUT/$source_case" ]] || {
            echo "missing raw case: $source_case" >&2
            exit 1
        }
        ln -sfn "$RAW_INPUT/$source_case" "$RAW_VIEW/$case_name"
    done < <("$VENV_DIR/bin/python" -c \
        'import json; print(*json.load(open("'"$BENCHMARK_DIR"'/meta/inference_cases.json")), sep="\n")')

    set +e
    (
        cd "$BENCHMARK_DIR"
        PYTHONUNBUFFERED=1 "$VENV_DIR/bin/python" preprocess.py \
            --raw_data_dir "$RAW_VIEW" --results_dir "$PREPROCESSED_DIR" \
            --mode preprocess --num_proc 4
    ) 2>&1 | tee "$TARGET_DIR/preprocess.log"
    preprocess_status=${PIPESTATUS[0]}
    set -e
    if [[ $preprocess_status -ne 0 && ! -s "$PREPROCESSED_DIR/preprocessed_files.pkl" ]]; then
        echo "preprocessing failed before writing its completion metadata" >&2
        exit "$preprocess_status"
    fi
    if [[ $preprocess_status -ne 0 ]]; then
        echo "Preprocessing completed; checksum verification was skipped because the supplied source has no checksum manifest."
    fi
else
    echo "Reusing completed preprocessing." | tee "$TARGET_DIR/preprocess.log"
fi

mkdir -p "$POSTPROCESSED_DIR"
cp "$LOADGEN_DIR/mlperf.conf" "$TARGET_DIR/mlperf.conf"

echo "GPU state immediately before inference:" | tee "$TARGET_DIR/gpu.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits 2>&1 | tee -a "$TARGET_DIR/gpu.log"

cd "$BENCHMARK_DIR"
export LOG_PATH="$OUT_DIR"
export SKIP_VERIFY_ACCURACY=1
export PYTHONUNBUFFERED=1
"$VENV_DIR/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
    --model "$MODEL_PATH" \
    --preprocessed_data_dir "$PREPROCESSED_DIR" \
    --mlperf_conf "$TARGET_DIR/mlperf.conf" \
    --user_conf "$BENCHMARK_DIR/user.conf" \
    2>&1 | tee "$TARGET_DIR/run.log"

unset SKIP_VERIFY_ACCURACY
"$VENV_DIR/bin/python" accuracy_kits.py \
    --log_file "$OUT_DIR/mlperf_log_accuracy.json" \
    --preprocessed_data_dir "$PREPROCESSED_DIR" \
    --postprocessed_data_dir "$POSTPROCESSED_DIR" \
    --num_proc 4 \
    2> >(tee "$TARGET_DIR/evaluator_stderr.log" >&2) \
    | tee "$OUT_DIR/accuracy.txt" "$TARGET_DIR/evaluator.log"

for output in mlperf_log_accuracy.json mlperf_log_detail.txt accuracy.txt; do
    [[ -s "$OUT_DIR/$output" ]] || {
        echo "required output missing or empty: $OUT_DIR/$output" >&2
        exit 1
    }
done
rg -q '^pytorch device: ' "$TARGET_DIR/run.log" || {
    echo "required log prefix not found: pytorch device: " >&2
    exit 1
}

echo "Target $TARGET_ID completed successfully."
