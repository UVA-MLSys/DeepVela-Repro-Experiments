#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT_DEFAULT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
KITS19_RAW_DIR_DEFAULT="$CACHE_ROOT_DEFAULT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR_DEFAULT="$CACHE_ROOT_DEFAULT/3d-unet-kits19/model"
UV_CACHE_DIR_DEFAULT="$CACHE_ROOT_DEFAULT/uv"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=""
MODE=all
START_STAGE=""
ONE_STAGE=""

usage() {
  echo "usage: $0 --target 3d-unet [--stage setup|preprocess|run|evaluate] [--from setup|preprocess|run|evaluate]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET=${2:-}; shift 2 ;;
    --stage) MODE=stage; ONE_STAGE=${2:-}; shift 2 ;;
    --from) MODE=from; START_STAGE=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$TARGET" == 3d-unet ]] || usage
[[ -d "$ROOT/repo/vision/medical_imaging/3d-unet-kits19" ]] || { echo "fresh repository not found at $ROOT/repo" >&2; exit 1; }

OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/$TARGET/out"}
KITS19_RAW_DIR=${KITS19_RAW_DIR:-$KITS19_RAW_DIR_DEFAULT}
MODEL_DIR=${MODEL_DIR:-$MODEL_DIR_DEFAULT}
UV_CACHE_DIR=${UV_CACHE_DIR:-$UV_CACHE_DIR_DEFAULT}
PYTHON_BIN=$(command -v python3.11)
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

if [[ -n "${PREFLIGHT_CASES:-}" ]]; then
  WORK="$ROOT/$TARGET/preflight-work"
else
  WORK="$ROOT/$TARGET/work"
fi
SRC="$WORK/source"
VENV="$WORK/venv"
RAW="$WORK/raw_data"
PRE="$WORK/preprocessed_data"
POST="$WORK/postprocessed_data"
MARKERS="$WORK/markers"
mkdir -p "$OUTPUT_DIR" "$WORK" "$MARKERS"

stages=(setup preprocess run evaluate)
valid_stage() {
  local candidate=$1 stage
  for stage in "${stages[@]}"; do [[ "$candidate" == "$stage" ]] && return 0; done
  return 1
}

if [[ "$MODE" == stage ]]; then valid_stage "$ONE_STAGE" || usage; fi
if [[ "$MODE" == from ]]; then valid_stage "$START_STAGE" || usage; fi

clear_from() {
  local start=$1 seen=0 stage
  for stage in "${stages[@]}"; do
    [[ "$stage" == "$start" ]] && seen=1
    if [[ $seen -eq 1 ]]; then rm -f "$MARKERS/$stage.success"; fi
  done
  case "$start" in
    setup)
      rm -rf "$SRC" "$VENV" "$WORK/loadgen" "$RAW" "$PRE" "$POST"
      rm -f "$OUTPUT_DIR"/*
      ;;
    preprocess)
      rm -rf "$PRE" "$POST"
      rm -f "$OUTPUT_DIR/mlperf_log_"* "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/preprocess.log" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log"
      ;;
    run)
      rm -rf "$POST"
      rm -f "$OUTPUT_DIR/mlperf_log_"* "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log"
      ;;
    evaluate)
      rm -rf "$POST"
      rm -f "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
      ;;
  esac
}
[[ "$MODE" == from ]] && clear_from "$START_STAGE"

setup_stage() {
  [[ -f "$MARKERS/setup.success" ]] && return
  {
    echo "setup: Python 3.11 environment and pinned benchmark dependencies"
    test -d "$KITS19_RAW_DIR"
    test -f "$MODEL_DIR/3dunet_kits19_pytorch.ptc"
    cp -a "$ROOT/repo/vision/medical_imaging/3d-unet-kits19" "$SRC"
    cp "$ROOT/repo/mlperf.conf" "$SRC/mlperf.conf"
    cp -a "$ROOT/repo/loadgen" "$WORK/loadgen"
    mkdir -p "$RAW"
    local manifest="$SRC/meta/inference_cases.json"
    if [[ -n "${PREFLIGHT_CASES:-}" ]]; then
      "$PYTHON_BIN" - "$manifest" "$SRC/meta/checksum_inference.json" "$PREFLIGHT_CASES" <<'PY'
import json, sys
manifest, checksums, raw = sys.argv[1:]
cases = [x.strip() for x in raw.split(',') if x.strip()]
with open(manifest, 'w') as f:
    json.dump(cases, f)
with open(checksums) as f:
    old = json.load(f)
with open(checksums, 'w') as f:
    json.dump({c + '.pkl': old[c + '.pkl'] for c in cases}, f, indent=4, sort_keys=True)
PY
    fi
    while IFS= read -r case_name; do
      source_case=$case_name
      [[ "$case_name" == case_00400 ]] && source_case=case_00185
      test -d "$KITS19_RAW_DIR/$source_case"
      ln -sfn "$KITS19_RAW_DIR/$source_case" "$RAW/$case_name"
    done < <("$PYTHON_BIN" -c 'import json,sys; print(*json.load(open(sys.argv[1])), sep="\n")' "$manifest")
    "$PYTHON_BIN" - "$SRC/pytorch_SUT.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
needle = "        self.model.eval()\n"
replacement = needle + "        print(\"pytorch device: {}\".format(next(self.model.parameters()).device))\n"
assert needle in s
open(p, 'w').write(s.replace(needle, replacement, 1))
PY
    "$PYTHON_BIN" - "$SRC/run.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
needle = '    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)\n'
replacement = '    if not args.accuracy:\n        settings.FromConfig(args.user_conf, "3d-unet", args.scenario)\n'
assert needle in s
open(p, 'w').write(s.replace(needle, replacement, 1))
PY
    uv venv --python "$PYTHON_BIN" "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'numpy==1.26.4' 'scipy==1.11.4' 'pandas==1.5.3' 'nibabel==5.3.2' \
      'torch==2.2.0' "$WORK/loadgen"
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "nibabel", nibabel.__version__)
PY
  } 2>&1 | tee "$OUTPUT_DIR/setup.log"
  touch "$MARKERS/setup.success"
}

preprocess_stage() {
  [[ -f "$MARKERS/preprocess.success" ]] && return
  [[ -f "$MARKERS/setup.success" ]] || setup_stage
  mkdir -p "$PRE"
  (
    cd "$SRC"
    "$VENV/bin/python" preprocess.py --raw_data_dir "$RAW" --results_dir "$PRE" --mode preprocess --num_proc 8
  ) 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  touch "$MARKERS/preprocess.success"
}

run_stage() {
  [[ -f "$MARKERS/run.success" ]] && return
  [[ -f "$MARKERS/preprocess.success" ]] || preprocess_stage
  {
    echo "GPU state immediately before inference:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( free_mib < 16000 )); then
      echo "Insufficient free GPU memory: ${free_mib} MiB (need at least 16000 MiB)" >&2
      exit 1
    fi
    cd "$SRC"
    LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 "$VENV/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
      --preprocessed_data_dir="$PRE" --mlperf_conf="$SRC/mlperf.conf" --user_conf="$SRC/user.conf"
  } 2>&1 | tee "$OUTPUT_DIR/run.log"
  touch "$MARKERS/run.success"
}

evaluate_stage() {
  [[ -f "$MARKERS/evaluate.success" ]] && return
  [[ -f "$MARKERS/run.success" ]] || run_stage
  mkdir -p "$POST"
  (
    cd "$SRC"
    "$VENV/bin/python" accuracy_kits.py \
      --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="$PRE" --postprocessed_data_dir="$POST" --num_proc=8
  ) 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  touch "$MARKERS/evaluate.success"
}

run_named_stage() {
  case "$1" in
    setup) setup_stage ;;
    preprocess) preprocess_stage ;;
    run) run_stage ;;
    evaluate) evaluate_stage ;;
  esac
}

case "$MODE" in
  stage) run_named_stage "$ONE_STAGE" ;;
  all) for stage in "${stages[@]}"; do run_named_stage "$stage"; done ;;
  from)
    started=0
    for stage in "${stages[@]}"; do
      [[ "$stage" == "$START_STAGE" ]] && started=1
      [[ $started -eq 1 ]] && run_named_stage "$stage"
    done
    ;;
esac
