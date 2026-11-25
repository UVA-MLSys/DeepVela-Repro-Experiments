#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--target" || "${2:-}" != "3d-unet" || $# -ne 2 ]]; then
  echo "usage: $0 --target 3d-unet" >&2
  exit 2
fi
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

ROOT=$(cd "$(dirname "$0")" && pwd)
OUTPUT_DIR=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)
TARGET_DIR="$ROOT/3d-unet"
WORK_DIR="$TARGET_DIR/work"
BENCH_DIR="$WORK_DIR/benchmark"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$TARGET_DIR" "$WORK_DIR/raw_data" "$WORK_DIR/preprocessed" \
  "$WORK_DIR/logs" "$WORK_DIR/postprocessed"
cp -a "$ROOT/repo/vision/medical_imaging/3d-unet-kits19" "$BENCH_DIR"
cp "$ROOT/repo/loadgen/mlperf.conf" "$BENCH_DIR/mlperf.conf"

python3 - "$RAW_INPUT" "$WORK_DIR/raw_data" "$BENCH_DIR/meta/inference_cases.json" <<'PY' \
  2>&1 | tee "$TARGET_DIR/setup.log"
import json
import pathlib
import sys

source_root, destination, case_file = map(pathlib.Path, sys.argv[1:])
for case in json.loads(case_file.read_text()):
    source = source_root / ("case_00185" if case == "case_00400" else case)
    link = destination / case
    if not source.is_dir():
        raise FileNotFoundError(source)
    link.symlink_to(source, target_is_directory=True)
PY

python3 - "$BENCH_DIR" <<'PY' 2>&1 | tee "$TARGET_DIR/patch.log"
from pathlib import Path
import sys

root = Path(sys.argv[1])
p = root / "pytorch_SUT.py"
s = p.read_text()
s = s.replace(
    "        self.model.eval()\n",
    "        self.model.eval()\n"
    "        print(\"pytorch device: {}\".format(next(self.model.parameters()).device))\n",
)
p.write_text(s)

p = root / "inference_utils.py"
p.write_text(p.read_text().replace("signal.gaussian(n, std)", "signal.windows.gaussian(n, std)"))

p = root / "accuracy_kits.py"
s = p.read_text()
s = s.replace(
    '        df = df.append(\n'
    '            {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite},\n'
    '            ignore_index=True,\n'
    '        )',
    '        df = pd.concat([df, pd.DataFrame([{"case": case, "kidney": kidney, '
    '"tumor": tumor, "composite": composite}])], ignore_index=True)',
)
s = s.replace('float(final["composite"])', 'float(final["composite"].iloc[0])')
s = s.replace('float(final["kidney"])', 'float(final["kidney"].iloc[0])')
s = s.replace('float(final["tumor"])', 'float(final["tumor"].iloc[0])')
s = s.replace(
    '    df.set_index("case", inplace=True)\n',
    '    df.set_index("case", inplace=True)\n    df.loc["mean"] = df.mean()\n',
)
p.write_text(s)
PY

uv venv --python /usr/bin/python3.11 "$WORK_DIR/venv" \
  2>&1 | tee "$TARGET_DIR/build.log"
uv pip install --python "$WORK_DIR/venv/bin/python" --offline \
  torch numpy scipy nibabel pandas pybind11 setuptools wheel mlcommons-loadgen \
  2>&1 | tee -a "$TARGET_DIR/build.log"

set +e
(cd "$BENCH_DIR" && "$WORK_DIR/venv/bin/python" preprocess.py \
  --raw_data_dir ../raw_data --results_dir ../preprocessed --mode preprocess --num_proc 8) \
  2>&1 | tee "$TARGET_DIR/preprocess.log"
PREPROCESS_RC=${PIPESTATUS[0]}
set -e
if [[ ! -s "$WORK_DIR/preprocessed/preprocessed_files.pkl" ]]; then
  echo "preprocessing failed with status $PREPROCESS_RC and produced no metadata" >&2
  exit "$PREPROCESS_RC"
fi

nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free \
  --format=csv,noheader 2>&1 | tee "$TARGET_DIR/gpu_memory_before_run.log"

(cd "$BENCH_DIR" && \
  LOG_PATH=../logs SKIP_VERIFY_ACCURACY=1 PYTHONUNBUFFERED=1 \
  "$WORK_DIR/venv/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
    --model="$MODEL" --preprocessed_data_dir=../preprocessed \
    --mlperf_conf=mlperf.conf --user_conf=user.conf) \
  2>&1 | tee "$TARGET_DIR/run.log"

(cd "$BENCH_DIR" && "$WORK_DIR/venv/bin/python" accuracy_kits.py \
  --log_file=../logs/mlperf_log_accuracy.json \
  --preprocessed_data_dir=../preprocessed \
  --postprocessed_data_dir=../postprocessed --num_proc 4) \
  2> >(tee "$TARGET_DIR/evaluator_stderr.log" >&2) \
  | tee "$TARGET_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"

cp "$WORK_DIR/logs/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
cp "$WORK_DIR/logs/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"
