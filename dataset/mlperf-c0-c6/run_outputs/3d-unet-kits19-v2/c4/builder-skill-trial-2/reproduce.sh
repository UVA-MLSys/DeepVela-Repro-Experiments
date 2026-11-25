#!/usr/bin/env bash
set -euo pipefail

TARGET=""
ACTION="all"
START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) ACTION="stage"; START_STAGE="${2:-}"; shift 2 ;;
    --from) ACTION="from"; START_STAGE="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "3d-unet" ]] || { echo "supported target: 3d-unet" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$ROOT_DIR/repo"
TARGET_DIR="$ROOT_DIR/$TARGET"
OUTPUT_DIR="${OUTPUT_DIR:-$TARGET_DIR/out}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
WORK_DIR="$TARGET_DIR/work"
SOURCE_DIR="$WORK_DIR/source"
LOADGEN_DIR="$WORK_DIR/loadgen"
VENV_DIR="$TARGET_DIR/venv"
PREPROCESSED_DIR="$WORK_DIR/preprocessed_data"
POSTPROCESSED_DIR="$WORK_DIR/postprocessed_data"
RAW_VIEW_DIR="$WORK_DIR/raw_data"
MODEL_PATH="${MODEL_DIR:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache/3d-unet-kits19/model}/3dunet_kits19_pytorch.ptc"
KITS19_RAW_DIR="${KITS19_RAW_DIR:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache/3d-unet-kits19/kits19_raw_data/kits19/data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache/uv}"
export UV_EXCLUDE_NEWER=2026-11-10
export UV_LINK_MODE=copy

STAGES=(environment data preflight run evaluate)
marker() { printf '%s/.%s.success' "$TARGET_DIR" "$1"; }

stage_environment() {
  mkdir -p "$WORK_DIR"
  rm -rf "$SOURCE_DIR" "$LOADGEN_DIR"
  cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" "$SOURCE_DIR"
  cp -a "$REPO_DIR/loadgen" "$LOADGEN_DIR"
  python3 - "$SOURCE_DIR" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1])
p = src / "pytorch_SUT.py"
s = p.read_text()
old = "        self.model.eval()\n"
new = old + '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n'
if new not in s:
    if old not in s:
        raise SystemExit("device-log patch anchor missing")
    p.write_text(s.replace(old, new, 1))
p = src / "accuracy_kits.py"
s = p.read_text()
old = '''    df = pd.DataFrame()\n\n    for _s in dice_scores:\n        case, arr = _s\n        kidney = arr[0]\n        tumor = arr[1]\n        composite = np.mean(arr)\n        df = df.append(\n            {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite},\n            ignore_index=True,\n        )\n\n    df.set_index("case", inplace=True)\n\n    df.to_csv(sum_path)\n'''
new = '''    rows = []\n\n    for _s in dice_scores:\n        case, arr = _s\n        kidney = arr[0]\n        tumor = arr[1]\n        composite = np.mean(arr)\n        rows.append(\n+            {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite}\n+        )\n\n    df = pd.DataFrame.from_records(rows).set_index("case")\n    df.loc["mean"] = df.mean(axis=0)\n\n    df.to_csv(sum_path)\n'''.replace("\n+", "\n")
if new not in s:
    if old not in s:
        raise SystemExit("evaluator patch anchor missing")
    p.write_text(s.replace(old, new, 1))
PY
  rm -rf "$VENV_DIR"
  uv venv --python 3.11 "$VENV_DIR"
  uv pip install --python "$VENV_DIR/bin/python" setuptools==80.9.0 wheel==0.45.1 pybind11==2.11.1
  uv pip install --python "$VENV_DIR/bin/python" torch==2.9.1 numpy==1.26.4 scipy==1.12.0 nibabel==5.2.1 pandas==1.5.3
  uv pip install --python "$VENV_DIR/bin/python" --no-build-isolation "$LOADGEN_DIR"
  "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("nibabel", nibabel.__version__, "pandas", pandas.__version__)
print("loadgen import ok")
PY
}

stage_data() {
  [[ -f "$MODEL_PATH" ]] || { echo "model missing: $MODEL_PATH" >&2; exit 1; }
  [[ -d "$KITS19_RAW_DIR/case_00185" ]] || { echo "raw data missing: $KITS19_RAW_DIR" >&2; exit 1; }
  rm -rf "$RAW_VIEW_DIR" "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR"
  mkdir -p "$RAW_VIEW_DIR" "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR"
  "$VENV_DIR/bin/python" - "$SOURCE_DIR/meta/inference_cases.json" "$KITS19_RAW_DIR" "$RAW_VIEW_DIR" <<'PY'
import json, os, sys
manifest, raw, view = sys.argv[1:]
for case in json.load(open(manifest)):
    source = "case_00185" if case == "case_00400" else case
    path = os.path.join(raw, source)
    if not os.path.isdir(path):
        raise SystemExit(f"missing input case: {source}")
    os.symlink(path, os.path.join(view, case), target_is_directory=True)
print("materialized 43-case read-only view; case_00400 duplicates case_00185")
PY
  (cd "$SOURCE_DIR" && "$VENV_DIR/bin/python" preprocess.py \
    --raw_data_dir "$RAW_VIEW_DIR" --results_dir "$PREPROCESSED_DIR" \
    --mode preprocess --num_proc 4)
}

gpu_snapshot() {
  nvidia-smi --query-gpu=timestamp,name,uuid,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
}

stage_preflight() {
  gpu_snapshot
  (cd "$SOURCE_DIR" && "$VENV_DIR/bin/python" - "$MODEL_PATH" <<'PY'
import sys
import numpy as np
import torch
from accuracy_kits import get_dice_score
free, total = torch.cuda.mem_get_info()
print(f"preflight cuda free={free} total={total}")
assert torch.cuda.is_available(), "CUDA is required"
model = torch.jit.load(sys.argv[1], map_location="cuda:0").eval()
print("pytorch device:", next(model.parameters()).device)
x = torch.zeros((1, 1, 128, 128, 128), dtype=torch.float32, device="cuda:0")
with torch.no_grad():
    y = model(x)
print("preflight model output", tuple(y.shape), str(y.dtype), str(y.device))
a = np.zeros((1, 4, 4, 4), dtype=np.uint8)
case, dice = get_dice_score("preflight_empty", a, a)
print("preflight evaluator", case, dice.tolist())
assert np.allclose(dice, [1.0, 1.0])
PY
  )
}

stage_run() {
  gpu_snapshot
  cp "$REPO_DIR/loadgen/mlperf.conf" "$WORK_DIR/mlperf.conf"
  export LOG_PATH="$OUTPUT_DIR"
  export SKIP_VERIFY_ACCURACY=1
  (cd "$SOURCE_DIR" && "$VENV_DIR/bin/python" run.py \
    --backend=pytorch --scenario=Offline --accuracy \
    --model="$MODEL_PATH" \
    --preprocessed_data_dir="$PREPROCESSED_DIR" \
    --mlperf_conf="$WORK_DIR/mlperf.conf" \
    --user_conf="$SOURCE_DIR/user.conf")
  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" ]]
  [[ -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
}

stage_evaluate() {
  rm -rf "$POSTPROCESSED_DIR"
  mkdir -p "$POSTPROCESSED_DIR"
  (cd "$SOURCE_DIR" && "$VENV_DIR/bin/python" accuracy_kits.py \
    --preprocessed_data_dir="$PREPROCESSED_DIR" \
    --postprocessed_data_dir="$POSTPROCESSED_DIR" \
    --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --output_dtype=uint8 --num_proc=4) | tee "$OUTPUT_DIR/accuracy.txt"
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  grep -q '^pytorch device: cuda' "$OUTPUT_DIR/run.log"
}

clear_from() {
  local found=0 stage
  for stage in "${STAGES[@]}"; do
    if [[ "$stage" == "$1" ]]; then found=1; fi
    if [[ $found -eq 1 ]]; then rm -f "$(marker "$stage")"; fi
  done
  [[ $found -eq 1 ]] || { echo "unknown stage: $1" >&2; exit 2; }
  case "$1" in
    environment) rm -rf "$VENV_DIR" "$WORK_DIR"; rm -f "$OUTPUT_DIR"/environment.log "$OUTPUT_DIR"/data.log "$OUTPUT_DIR"/preflight.log "$OUTPUT_DIR"/run.log "$OUTPUT_DIR"/evaluator.log "$OUTPUT_DIR"/accuracy.txt "$OUTPUT_DIR"/mlperf_log_*.json "$OUTPUT_DIR"/mlperf_log_*.txt ;;
    data) rm -rf "$RAW_VIEW_DIR" "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR"; rm -f "$OUTPUT_DIR"/data.log "$OUTPUT_DIR"/preflight.log "$OUTPUT_DIR"/gpu_preflight.log "$OUTPUT_DIR"/run.log "$OUTPUT_DIR"/gpu_run.log "$OUTPUT_DIR"/evaluator.log "$OUTPUT_DIR"/accuracy.txt "$OUTPUT_DIR"/mlperf_log_*.json "$OUTPUT_DIR"/mlperf_log_*.txt ;;
    preflight) rm -f "$OUTPUT_DIR"/preflight.log "$OUTPUT_DIR"/gpu_preflight.log "$OUTPUT_DIR"/run.log "$OUTPUT_DIR"/gpu_run.log "$OUTPUT_DIR"/evaluator.log "$OUTPUT_DIR"/accuracy.txt "$OUTPUT_DIR"/mlperf_log_*.json "$OUTPUT_DIR"/mlperf_log_*.txt ;;
    run) rm -f "$OUTPUT_DIR"/run.log "$OUTPUT_DIR"/gpu_run.log "$OUTPUT_DIR"/evaluator.log "$OUTPUT_DIR"/accuracy.txt "$OUTPUT_DIR"/mlperf_log_*.json "$OUTPUT_DIR"/mlperf_log_*.txt ;;
    evaluate) rm -rf "$POSTPROCESSED_DIR"; rm -f "$OUTPUT_DIR"/evaluator.log "$OUTPUT_DIR"/accuracy.txt ;;
  esac
}

run_stage() {
  local stage="$1" log="$OUTPUT_DIR/$1.log"
  if [[ -f "$(marker "$stage")" ]]; then echo "stage $stage already complete"; return; fi
  echo "[$(date -Ins)] starting stage $stage" | tee "$log"
  local start=$SECONDS
  "stage_$stage" 2>&1 | tee -a "$log"
  local elapsed=$((SECONDS-start))
  echo "[$(date -Ins)] completed stage $stage elapsed_seconds=$elapsed" | tee -a "$log"
  : > "$(marker "$stage")"
}

if [[ "$ACTION" == "from" ]]; then clear_from "$START_STAGE"; fi
if [[ "$ACTION" == "stage" ]]; then run_stage "$START_STAGE"; exit 0; fi

start=0
[[ "$ACTION" == "all" ]] && start=1
for stage in "${STAGES[@]}"; do
  if [[ "$ACTION" == "from" && "$stage" == "$START_STAGE" ]]; then start=1; fi
  [[ $start -eq 1 ]] && run_stage "$stage"
done
