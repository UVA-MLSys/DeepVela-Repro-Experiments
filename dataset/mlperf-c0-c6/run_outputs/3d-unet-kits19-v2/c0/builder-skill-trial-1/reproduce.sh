#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
KITS19_RAW_DIR=${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}
MODEL_DIR=${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}
UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
UV_EXCLUDE_NEWER=2026-11-10
export UV_CACHE_DIR UV_EXCLUDE_NEWER

TARGET=""
STAGE=""
FROM=""
PREFLIGHT=0
while (($#)); do
  case "$1" in
    --target) TARGET=${2:?missing target id}; shift 2 ;;
    --stage) STAGE=${2:?missing stage}; shift 2 ;;
    --from) FROM=${2:?missing stage}; shift 2 ;;
    --preflight) PREFLIGHT=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$TARGET" == "3d-unet" ]] || { echo "supported target: 3d-unet" >&2; exit 2; }
[[ -z "$STAGE" || -z "$FROM" ]] || { echo "use only one of --stage or --from" >&2; exit 2; }

TARGET_DIR=$ROOT/$TARGET
VENV=$TARGET_DIR/.venv
if ((PREFLIGHT)); then
  RUN_ROOT=$TARGET_DIR/preflight
  OUTPUT_DIR=$RUN_ROOT/out
else
  RUN_ROOT=$TARGET_DIR
  OUTPUT_DIR=${OUTPUT_DIR:-$TARGET_DIR/out}
fi
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
case "$OUTPUT_DIR/" in
  "$TARGET_DIR/"*) ;;
  *) echo "OUTPUT_DIR must be inside $TARGET_DIR" >&2; exit 2 ;;
esac
WORK=$RUN_ROOT/work
SRC=$WORK/source
PREPROCESSED=$WORK/preprocessed_data
POSTPROCESSED=$WORK/postprocessed_data
RAW_VIEW=$WORK/raw_data
MODEL=$MODEL_DIR/3dunet_kits19_pytorch.ptc
export OUTPUT_DIR

stages=(setup preprocess run)
stage_index() {
  case "$1" in setup) echo 0;; preprocess) echo 1;; run) echo 2;; *) return 1;; esac
}

clear_from() {
  local idx i s
  idx=$(stage_index "$1") || { echo "unknown stage: $1" >&2; exit 2; }
  for ((i=idx; i<${#stages[@]}; i++)); do
    s=${stages[$i]}
    rm -f "$RUN_ROOT/$s.success"
  done
  if ((idx == 0)); then
    rm -rf "$WORK" "$OUTPUT_DIR"
    if ((!PREFLIGHT)); then rm -rf "$VENV"; fi
  elif ((idx == 1)); then
    rm -rf "$PREPROCESSED" "$POSTPROCESSED"
    rm -f "$OUTPUT_DIR"/mlperf_log_* "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log"
  else
    rm -rf "$POSTPROCESSED"
    rm -f "$OUTPUT_DIR"/mlperf_log_* "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log"
  fi
}

setup_stage() {
  mkdir -p "$WORK" "$OUTPUT_DIR" "$TARGET_DIR"
  exec > >(tee -a "$OUTPUT_DIR/build.log") 2>&1
  echo "stage=setup preflight=$PREFLIGHT"
  test -d "$ROOT/repo/vision/medical_imaging/3d-unet-kits19"
  test -f "$MODEL"
  test -d "$KITS19_RAW_DIR"

  rm -rf "$SRC"
  mkdir -p "$SRC"
  cp -a "$ROOT/repo/vision/medical_imaging/3d-unet-kits19/." "$SRC/"
  cp -a "$ROOT/repo/loadgen" "$WORK/loadgen"
  cp "$ROOT/repo/mlperf.conf" "$WORK/mlperf.conf"

  # Required observability only; inference device selection is unchanged.
  sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(self.device))' "$SRC/pytorch_SUT.py"
  # This snapshot omits trace_generator.h and no source includes it; remove only
  # the stale setuptools dependency declaration.
  sed -i '/"trace_generator.h",/d' "$WORK/loadgen/setup.py"

  if ((PREFLIGHT)); then
    python3 - "$SRC/meta/inference_cases.json" "$SRC/meta/checksum_inference.json" <<'PY'
import json, sys
cases_path, sums_path = sys.argv[1:]
case = json.load(open(cases_path))[0]
json.dump([case], open(cases_path, "w"), indent=2)
sums = json.load(open(sums_path))
key = case + ".pkl"
json.dump({key: sums[key]}, open(sums_path, "w"), indent=2, sort_keys=True)
PY
  fi

  if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'numpy==1.26.4' 'scipy==1.11.4' 'pandas==1.5.3' \
      'nibabel==5.2.1' 'torch==2.5.1' 'pybind11==2.13.6' \
      'setuptools==75.6.0' 'wheel==0.45.1'
    uv pip install --python "$VENV/bin/python" --no-build-isolation "$WORK/loadgen"
  fi
  "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "nibabel", nibabel.__version__)
PY
  uv pip freeze --python "$VENV/bin/python" > "$OUTPUT_DIR/environment.freeze.txt"
  touch "$RUN_ROOT/setup.success"
}

populate_and_preprocess() {
  mkdir -p "$OUTPUT_DIR"
  exec > >(tee -a "$OUTPUT_DIR/preprocess.log") 2>&1
  echo "stage=preprocess preflight=$PREFLIGHT"
  test -f "$RUN_ROOT/setup.success"
  rm -rf "$RAW_VIEW" "$PREPROCESSED"
  mkdir -p "$RAW_VIEW" "$PREPROCESSED"
  mapfile -t cases < <("$VENV/bin/python" - "$SRC/meta/inference_cases.json" <<'PY'
import json, sys
print(*json.load(open(sys.argv[1])), sep="\n")
PY
  )
  local case src_case
  for case in "${cases[@]}"; do
    src_case=$case
    [[ "$case" == case_00400 ]] && src_case=case_00185
    test -f "$KITS19_RAW_DIR/$src_case/imaging.nii.gz"
    test -f "$KITS19_RAW_DIR/$src_case/segmentation.nii.gz"
    ln -s "$KITS19_RAW_DIR/$src_case" "$RAW_VIEW/$case"
  done
  echo "input_cases=${#cases[@]}"
  (cd "$SRC" && /usr/bin/time -v "$VENV/bin/python" preprocess.py \
    --raw_data_dir "$RAW_VIEW" --results_dir "$PREPROCESSED" \
    --mode preprocess --num_proc 4)
  test "$(find "$PREPROCESSED" -maxdepth 1 -name 'case_*.pkl' | wc -l)" -eq "${#cases[@]}"
  touch "$RUN_ROOT/preprocess.success"
}

run_stage() {
  mkdir -p "$OUTPUT_DIR" "$POSTPROCESSED"
  test -f "$RUN_ROOT/preprocess.success"
  nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.free,memory.used \
    --format=csv,noheader | tee "$OUTPUT_DIR/gpu-memory-before.log"
  "$VENV/bin/python" - <<'PY' | tee -a "$OUTPUT_DIR/gpu-memory-before.log"
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
free, total = torch.cuda.mem_get_info(0)
print(f"torch cuda free bytes: {free}; total bytes: {total}")
assert free >= 16 * 1024**3, f"need at least 16 GiB free, found {free / 1024**3:.1f} GiB"
PY
  rm -f "$OUTPUT_DIR"/mlperf_log_accuracy.json "$OUTPUT_DIR"/mlperf_log_detail.txt "$OUTPUT_DIR"/mlperf_log_summary.txt
  export LOG_PATH=$OUTPUT_DIR SKIP_VERIFY_ACCURACY=1
  (cd "$SRC" && /usr/bin/time -v "$VENV/bin/python" run.py \
    --backend=pytorch --scenario=Offline --accuracy \
    --model="$MODEL" --preprocessed_data_dir="$PREPROCESSED" \
    --mlperf_conf="$WORK/mlperf.conf" --user_conf="$SRC/user.conf") \
    2>&1 | tee "$OUTPUT_DIR/run.log"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  (cd "$SRC" && "$VENV/bin/python" accuracy_kits.py \
    --preprocessed_data_dir="$PREPROCESSED" \
    --postprocessed_data_dir="$POSTPROCESSED" \
    --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" --num_proc=4) \
    2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  "$VENV/bin/python" - "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/accuracy.txt" "$SRC/meta/inference_cases.json" <<'PY'
import json, re, sys
log_path, accuracy_path, cases_path = sys.argv[1:]
records = json.load(open(log_path))
cases = json.load(open(cases_path))
assert len(records) == len(cases), (len(records), len(cases))
text = open(accuracy_path).read()
assert re.search(r"^Accuracy: mean = [0-9.]+, kidney = [0-9.]+, tumor = [0-9.]+$", text, re.M), text[-1000:]
print(f"validated {len(records)} accuracy records")
PY
  touch "$RUN_ROOT/run.success"
}

run_one() {
  case "$1" in
    setup) [[ -f "$RUN_ROOT/setup.success" ]] || setup_stage ;;
    preprocess) [[ -f "$RUN_ROOT/preprocess.success" ]] || populate_and_preprocess ;;
    run) [[ -f "$RUN_ROOT/run.success" ]] || run_stage ;;
  esac
}

if [[ -n "$FROM" ]]; then clear_from "$FROM"; fi
if [[ -n "$STAGE" ]]; then
  run_one "$STAGE"
else
  start=0
  if [[ -n "$FROM" ]]; then start=$(stage_index "$FROM"); fi
  for ((i=start; i<${#stages[@]}; i++)); do run_one "${stages[$i]}"; done
fi
