#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target <target-id>" >&2
    exit 2
}

TARGET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) [[ $# -ge 2 ]] || usage; TARGET="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ "$TARGET" == "3d-unet" ]] || { echo "unsupported target: $TARGET" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$ROOT_DIR/repo"
TARGET_DIR="$ROOT_DIR/$TARGET"
WORK_DIR="$TARGET_DIR/work"
BENCH_DIR="$WORK_DIR/3d-unet-kits19"
VENV_DIR="$TARGET_DIR/venv"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    if [[ "$OUTPUT_DIR" = /* ]]; then
        OUT_DIR="$OUTPUT_DIR"
    else
        OUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
    fi
else
    OUT_DIR="$TARGET_DIR/out"
fi
CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER="2026-11-10"
export PYTHONUNBUFFERED=1

mkdir -p "$TARGET_DIR" "$WORK_DIR" "$OUT_DIR"
exec > >(tee "$TARGET_DIR/reproduce.log") 2>&1

echo "Preparing private source tree"
rm -rf "$BENCH_DIR" "$WORK_DIR/loadgen"
cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" "$BENCH_DIR"
cp -a "$REPO_DIR/loadgen" "$WORK_DIR/loadgen"

# Compatibility with modern SciPy/Pandas while preserving the reference math.
sed -i 's/from scipy import signal/from scipy import signal\nfrom scipy.signal import windows/' "$BENCH_DIR/inference_utils.py"
sed -i 's/signal\.gaussian(/windows.gaussian(/' "$BENCH_DIR/inference_utils.py"
sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$BENCH_DIR/pytorch_SUT.py"
python3 - "$BENCH_DIR/accuracy_kits.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = '''        df = df.append(
            {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite},
            ignore_index=True,
        )'''
new = '''        if len(df.columns) == 0:
            df = pd.DataFrame(columns=["case", "kidney", "tumor", "composite"])
        df.loc[len(df)] = {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite}'''
if old not in s:
    raise SystemExit("expected pandas compatibility site not found")
p.write_text(s.replace(old, new))
PY

echo "Building Python environment"
uv venv --offline --python 3.11 "$VENV_DIR" 2>&1 | tee "$TARGET_DIR/build-venv.log"
uv pip install --offline --python "$VENV_DIR/bin/python" \
    'numpy==1.26.4' 'scipy==1.11.4' 'pandas==2.2.3' 'nibabel==5.3.2' \
    'torch==2.5.1' 'setuptools<81' wheel 'pybind11==2.11.1' \
    2>&1 | tee "$TARGET_DIR/build-packages.log"
uv pip install --offline --python "$VENV_DIR/bin/python" --no-build-isolation "$WORK_DIR/loadgen" \
    2>&1 | tee "$TARGET_DIR/build-loadgen.log"

echo "Preparing read-only input links"
mkdir -p "$WORK_DIR/raw_data"
find "$RAW_INPUT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | while read -r case_name; do
    ln -sfn "$RAW_INPUT/$case_name" "$WORK_DIR/raw_data/$case_name"
done
ln -sfn "$RAW_INPUT/case_00185" "$WORK_DIR/raw_data/case_00400"
mkdir -p "$BENCH_DIR/build/model"
ln -sfn "$MODEL_INPUT" "$BENCH_DIR/build/model/3dunet_kits19_pytorch.ptc"
cp "$REPO_DIR/loadgen/mlperf.conf" "$BENCH_DIR/build/mlperf.conf"

cd "$BENCH_DIR"
echo "Preprocessing KiTS19 inference cases"
"$VENV_DIR/bin/python" preprocess.py \
    --raw_data_dir "$WORK_DIR/raw_data" \
    --results_dir build/preprocessed_data \
    --mode preprocess --num_proc "${PREPROCESS_PROCS:-8}" \
    2>&1 | tee "$TARGET_DIR/preprocess.log"

echo "GPU state immediately before inference"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader 2>&1 | tee "$TARGET_DIR/gpu-before-run.log"

export LOG_PATH="$OUT_DIR"
export SKIP_VERIFY_ACCURACY=1
"$VENV_DIR/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
    2>&1 | tee "$TARGET_DIR/run.log"
grep -q '^pytorch device: ' "$TARGET_DIR/run.log"

"$VENV_DIR/bin/python" accuracy_kits.py \
    --preprocessed_data_dir build/preprocessed_data \
    --postprocessed_data_dir build/postprocessed_data \
    --log_file "$OUT_DIR/mlperf_log_accuracy.json" \
    --num_proc "${EVAL_PROCS:-4}" \
    2>&1 | tee "$OUT_DIR/accuracy.txt" "$TARGET_DIR/evaluator.log"

test -s "$OUT_DIR/mlperf_log_accuracy.json"
test -s "$OUT_DIR/mlperf_log_detail.txt"
test -s "$OUT_DIR/accuracy.txt"
echo "Target $TARGET completed"
