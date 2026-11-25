#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
PREFLIGHT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:?missing target id}"; shift 2 ;;
    --stage) STAGE="${2:?missing stage}"; shift 2 ;;
    --from) FROM="${2:?missing stage}"; shift 2 ;;
    --preflight) PREFLIGHT=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "supported target: 3d-unet" >&2
  exit 2
fi
if [[ -n "$STAGE" && -n "$FROM" ]]; then
  echo "use only one of --stage or --from" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="$SCRIPT_DIR/repo"
CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
KITS19_RAW_DIR="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR="$CACHE_ROOT/3d-unet-kits19/model"
LOADGEN_WHEEL="$CACHE_ROOT/uv/sdists-v10/path/d631ddc5ecfaa9d3/430bt2BwNq64VZYX/mlcommons_loadgen-6.0.17-cp311-cp311-linux_x86_64.whl"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER="2026-11-10"

if [[ $PREFLIGHT -eq 1 ]]; then
  TARGET_DIR="$SCRIPT_DIR/$TARGET/preflight"
  OUT="$TARGET_DIR/out"
else
  TARGET_DIR="$SCRIPT_DIR/$TARGET"
  OUT="${OUTPUT_DIR:-$TARGET_DIR/out}"
fi
WORK_DIR="$TARGET_DIR/work"
WORK_REPO="$WORK_DIR/repo"
APP_DIR="$WORK_REPO/vision/medical_imaging/3d-unet-kits19"
VENV="$WORK_DIR/.venv"
PY="$VENV/bin/python"
MARKERS="$WORK_DIR/markers"

mkdir -p "$OUT"

valid_stage() {
  [[ "$1" == "setup" || "$1" == "preprocess" || "$1" == "run" ]]
}
if [[ -n "$STAGE" ]] && ! valid_stage "$STAGE"; then echo "invalid stage: $STAGE" >&2; exit 2; fi
if [[ -n "$FROM" ]] && ! valid_stage "$FROM"; then echo "invalid stage: $FROM" >&2; exit 2; fi

stage_setup() {
  rm -rf "$WORK_DIR"
  mkdir -p "$WORK_DIR" "$MARKERS"
  cp -a "$SOURCE_REPO" "$WORK_REPO"
  {
    echo "Python environment setup started: $(date --iso-8601=seconds)"
    uv venv --python /usr/bin/python3.11 "$VENV"
    uv pip install --python "$PY" --offline \
      "numpy==1.26.4" "scipy==1.10.1" "pandas==1.5.3" \
      "nibabel==5.3.2" "pybind11==2.11.1" "setuptools<80" wheel
    uv pip install --python "$PY" --offline "torch==2.5.1"
    test -s "$LOADGEN_WHEEL"
    test "$(tr -d '\n' < "$WORK_REPO/loadgen/VERSION.txt")" = "6.0.17"
    uv pip install --python "$PY" --offline "$LOADGEN_WHEEL"
    "$PY" - <<'PY'
import nibabel, numpy, pandas, scipy, torch, mlperf_loadgen
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("pandas", pandas.__version__)
print("nibabel", nibabel.__version__)
print("torch", torch.__version__)
print("torch CUDA runtime", torch.version.cuda)
print("loadgen import", mlperf_loadgen.__file__)
PY
    echo "Python environment setup finished: $(date --iso-8601=seconds)"
  } 2>&1 | tee "$OUT/build.log"

  sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$APP_DIR/pytorch_SUT.py"
  mkdir -p "$APP_DIR/build/model" "$APP_DIR/build/raw_data"
  ln -s "$MODEL_DIR/3dunet_kits19_pytorch.ptc" "$APP_DIR/build/model/3dunet_kits19_pytorch.ptc"
  cp "$WORK_REPO/loadgen/mlperf.conf" "$APP_DIR/build/mlperf.conf"

  if [[ $PREFLIGHT -eq 1 ]]; then
    printf '[\n    "case_00000"\n]\n' > "$APP_DIR/meta/inference_cases.json"
    "$PY" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); json.dump({"case_00000.pkl": d["case_00000.pkl"]}, open(p,"w"), indent=4, sort_keys=True)' "$APP_DIR/meta/checksum_inference.json"
    ln -s "$KITS19_RAW_DIR/case_00000" "$APP_DIR/build/raw_data/case_00000"
  else
    while IFS= read -r case_name; do
      [[ -n "$case_name" ]] || continue
      if [[ "$case_name" == "case_00400" ]]; then
        ln -s "$KITS19_RAW_DIR/case_00185" "$APP_DIR/build/raw_data/case_00400"
      else
        ln -s "$KITS19_RAW_DIR/$case_name" "$APP_DIR/build/raw_data/$case_name"
      fi
    done < <("$PY" -c 'import json; print("\n".join(json.load(open("'"$APP_DIR"'/meta/inference_cases.json"))))')
  fi
  touch "$MARKERS/setup.done"
}

stage_preprocess() {
  [[ -f "$MARKERS/setup.done" ]] || { echo "setup stage has not succeeded" >&2; exit 1; }
  rm -f "$MARKERS/preprocess.done" "$MARKERS/run.done"
  rm -rf "$APP_DIR/build/preprocessed_data" "$APP_DIR/build/postprocessed_data"
  mkdir -p "$APP_DIR/build/preprocessed_data" "$APP_DIR/build/postprocessed_data"
  {
    echo "Preprocessing started: $(date --iso-8601=seconds)"
    cd "$APP_DIR"
    "$PY" preprocess.py --raw_data_dir build/raw_data \
      --results_dir build/preprocessed_data --mode preprocess --num_proc 4
    echo "Preprocessed cases: $(find build/preprocessed_data -maxdepth 1 -name 'case_*.pkl' | wc -l)"
    echo "Preprocessing finished: $(date --iso-8601=seconds)"
  } 2>&1 | tee "$OUT/preprocess.log"
  touch "$MARKERS/preprocess.done"
}

stage_run() {
  [[ -f "$MARKERS/preprocess.done" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  rm -f "$MARKERS/run.done"
  {
    echo "GPU memory immediately before run: $(date --iso-8601=seconds)"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
  } 2>&1 | tee "$OUT/gpu_memory.log"

  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 20000 )); then
    echo "insufficient free GPU memory: ${free_mib:-unknown} MiB; need at least 20000 MiB" | tee -a "$OUT/gpu_memory.log" >&2
    exit 1
  fi

  rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
    "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt" "$OUT/evaluator.log"
  {
    cd "$APP_DIR"
    export LOG_PATH="$OUT"
    export SKIP_VERIFY_ACCURACY=1
    "$PY" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model=build/model/3dunet_kits19_pytorch.ptc \
      --preprocessed_data_dir=build/preprocessed_data \
      --mlperf_conf=build/mlperf.conf --user_conf=user.conf
  } 2>&1 | tee "$OUT/run.log"

  {
    cd "$APP_DIR"
    "$PY" accuracy_kits.py \
      --log_file="$OUT/mlperf_log_accuracy.json" \
      --preprocessed_data_dir=build/preprocessed_data \
      --postprocessed_data_dir=build/postprocessed_data --num_proc 4
  } 2>&1 | tee "$OUT/accuracy.txt" "$OUT/evaluator.log"

  grep -q '^pytorch device: ' "$OUT/run.log"
  grep -q '^Accuracy: mean = ' "$OUT/accuracy.txt"
  test -s "$OUT/mlperf_log_accuracy.json"
  test -s "$OUT/mlperf_log_detail.txt"
  touch "$MARKERS/run.done"
}

run_stage() {
  case "$1" in
    setup) stage_setup ;;
    preprocess) stage_preprocess ;;
    run) stage_run ;;
  esac
}

if [[ -n "$STAGE" ]]; then
  run_stage "$STAGE"
  exit 0
fi

stages=(setup preprocess run)
start=0
if [[ -n "$FROM" ]]; then
  case "$FROM" in setup) start=0;; preprocess) start=1;; run) start=2;; esac
else
  if [[ -f "$MARKERS/setup.done" ]]; then start=1; fi
  if [[ -f "$MARKERS/preprocess.done" ]]; then start=2; fi
  if [[ -f "$MARKERS/run.done" ]]; then exit 0; fi
fi
for ((i=start; i<${#stages[@]}; i++)); do
  run_stage "${stages[$i]}"
done
