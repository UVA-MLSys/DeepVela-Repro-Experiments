#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != "--target" || ${2:-} != "3d-unet" || $# -ne 2 ]]; then
    echo "usage: $0 --target 3d-unet" >&2
    exit 2
fi
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

ROOT=$(pwd)
TARGET=3d-unet
TARGET_DIR="$ROOT/$TARGET"
WORK_DIR="$TARGET_DIR/work"
BENCH_DIR="$WORK_DIR/benchmark"
OUT_DIR=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)
CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
RAW_SOURCE=${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}
MODEL_SOURCE=${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
export UV_EXCLUDE_NEWER=2026-11-10
export UV_PYTHON_INSTALL_DIR="$TARGET_DIR/python"

mkdir -p "$TARGET_DIR" "$WORK_DIR" "$BENCH_DIR" "$WORK_DIR/raw_data" \
    "$WORK_DIR/preprocessed_data" "$WORK_DIR/postprocessed_data" "$WORK_DIR/logs"
: >"$TARGET_DIR/build.log"
: >"$TARGET_DIR/preprocess.log"
: >"$TARGET_DIR/run.log"
: >"$TARGET_DIR/evaluator.log"

{
    echo "Installing target-local Python and dependencies"
    uv python install 3.11
    PYTHON_BASE=$(find "$UV_PYTHON_INSTALL_DIR" -type f -path '*/bin/python3.11' -print -quit)
    test -n "$PYTHON_BASE"
    uv venv --python "$PYTHON_BASE" "$TARGET_DIR/venv"
    uv pip install --python "$TARGET_DIR/venv/bin/python" \
        'numpy<2' 'scipy<1.14' 'nibabel<6' 'pandas>=2,<3' \
        'torch==2.7.1' setuptools wheel pybind11
    cp -a "$ROOT/repo/loadgen/." "$WORK_DIR/loadgen/"
    uv pip install --python "$TARGET_DIR/venv/bin/python" "$WORK_DIR/loadgen"
} 2>&1 | tee -a "$TARGET_DIR/build.log"

cp -a "$ROOT/repo/vision/medical_imaging/3d-unet-kits19/." "$BENCH_DIR/"
cp "$ROOT/repo/loadgen/mlperf.conf" "$BENCH_DIR/mlperf.conf"

# The source predates pandas 2, and the task requires an explicit device log line.
"$TARGET_DIR/venv/bin/python" - "$BENCH_DIR" <<'PY'
from pathlib import Path
import sys

bench = Path(sys.argv[1])
sut = bench / "pytorch_SUT.py"
text = sut.read_text()
needle = "        self.model.eval()\n"
replacement = needle + '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n'
if "pytorch device:" not in text:
    assert needle in text
    sut.write_text(text.replace(needle, replacement, 1))

accuracy = bench / "accuracy_kits.py"
text = accuracy.read_text()
old = '''        df = df.append(
            {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite},
            ignore_index=True,
        )'''
new = '''        df = pd.concat(
            [df, pd.DataFrame([{
                "case": case,
                "kidney": kidney,
                "tumor": tumor,
                "composite": composite,
            }])],
            ignore_index=True,
        )'''
if old in text:
    accuracy.write_text(text.replace(old, new, 1))

inference = bench / "inference_utils.py"
text = inference.read_text()
inference.write_text(text.replace("signal.gaussian(n, std)",
                                  "signal.windows.gaussian(n, std)"))

text = accuracy.read_text()
for column in ("composite", "kidney", "tumor"):
    text = text.replace(f'float(final["{column}"])',
                        f'float(final["{column}"].iloc[0])')
accuracy.write_text(text)
PY

for case_dir in "$RAW_SOURCE"/case_*; do
    ln -s "$case_dir" "$WORK_DIR/raw_data/$(basename "$case_dir")" 2>/dev/null || true
done
ln -s "$RAW_SOURCE/case_00185" "$WORK_DIR/raw_data/case_00400" 2>/dev/null || true

(
    cd "$BENCH_DIR"
    "$TARGET_DIR/venv/bin/python" preprocess.py \
        --raw_data_dir "$WORK_DIR/raw_data" \
        --results_dir "$WORK_DIR/preprocessed_data" \
        --mode preprocess --num_proc 8
) 2>&1 | tee -a "$TARGET_DIR/preprocess.log"

while true; do
    GPU_LINE=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits | head -n 1)
    echo "GPU memory before run (index, name, total MiB, used MiB, free MiB): $GPU_LINE" \
        | tee -a "$TARGET_DIR/run.log"
    GPU_FREE=$(awk -F, '{gsub(/ /, "", $5); print $5}' <<<"$GPU_LINE")
    if [[ "$GPU_FREE" =~ ^[0-9]+$ ]] && (( GPU_FREE >= 20480 )); then
        break
    fi
    echo "Waiting 15 seconds for at least 20480 MiB free GPU memory" | tee -a "$TARGET_DIR/run.log"
    sleep 15
done

# Match the benchmark's documented container launch setting for LoadGen's
# callback thread; the system default 8 MiB stack is too small for this model.
ulimit -s 65536

(
    cd "$BENCH_DIR"
    export LOG_PATH="$WORK_DIR/logs"
    export SKIP_VERIFY_ACCURACY=1
    "$TARGET_DIR/venv/bin/python" run.py \
        --backend=pytorch --scenario=Offline --accuracy \
        --model="$MODEL_SOURCE/3dunet_kits19_pytorch.ptc" \
        --preprocessed_data_dir="$WORK_DIR/preprocessed_data" \
        --mlperf_conf="$BENCH_DIR/mlperf.conf" \
        --user_conf="$BENCH_DIR/user.conf"
) 2>&1 | tee -a "$TARGET_DIR/run.log"

cp "$WORK_DIR/logs/mlperf_log_accuracy.json" "$OUT_DIR/mlperf_log_accuracy.json"
cp "$WORK_DIR/logs/mlperf_log_detail.txt" "$OUT_DIR/mlperf_log_detail.txt"

(
    cd "$BENCH_DIR"
    "$TARGET_DIR/venv/bin/python" accuracy_kits.py \
        --preprocessed_data_dir="$WORK_DIR/preprocessed_data" \
        --postprocessed_data_dir="$WORK_DIR/postprocessed_data" \
        --log_file="$OUT_DIR/mlperf_log_accuracy.json" --num_proc 8
) 2>&1 | tee "$OUT_DIR/accuracy.txt" "$TARGET_DIR/evaluator.log"
