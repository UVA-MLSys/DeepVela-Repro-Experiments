#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
KITS19_RAW_DIR="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR="$CACHE_ROOT/3d-unet-kits19/model"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="$ROOT_DIR/repo"
TARGET_ID=
ACTION=all
START_STAGE=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET_ID=${2:?missing target id}; shift 2 ;;
    --stage) ACTION=stage; START_STAGE=${2:?missing stage}; shift 2 ;;
    --from) ACTION=from; START_STAGE=${2:?missing stage}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET_ID" != "3d-unet" ]]; then
  echo "unsupported target: ${TARGET_ID:-<missing>}" >&2
  exit 2
fi
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
TARGET_DIR="$ROOT_DIR/$TARGET_ID"
WORK_DIR="$TARGET_DIR/work"
APP_DIR="$WORK_DIR/app"
LOADGEN_DIR="$WORK_DIR/loadgen"
VENV_DIR="$TARGET_DIR/.venv"
RAW_VIEW="$WORK_DIR/raw_data"
PREPROC_DIR="$WORK_DIR/preprocessed_data"
POSTPROC_DIR="$WORK_DIR/postprocessed_data"
MARKER_DIR="$WORK_DIR/markers"
MODEL_FILE="$MODEL_DIR/3dunet_kits19_pytorch.ptc"
mkdir -p "$WORK_DIR" "$MARKER_DIR"

stages=(setup preprocess run)

stage_setup() {
  exec > >(tee "$OUTPUT_DIR/build.log") 2>&1
  echo "stage setup: start $(date -Is)"
  test -f "$MODEL_FILE"
  test -d "$KITS19_RAW_DIR"
  command -v uv
  rm -rf "$APP_DIR" "$LOADGEN_DIR"
  cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" "$APP_DIR"
  cp -a "$REPO_DIR/loadgen" "$LOADGEN_DIR"
  cp "$REPO_DIR/loadgen/mlperf.conf" "$APP_DIR/mlperf.conf"

  # Compatibility-only runtime-copy repairs for the cached modern packages.
  sed -i 's/signal\.gaussian(/signal.windows.gaussian(/' "$APP_DIR/inference_utils.py"
  sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$APP_DIR/pytorch_SUT.py"

  uv venv --python 3.11 --clear "$VENV_DIR"
  uv pip install --offline --python "$VENV_DIR/bin/python" \
    'setuptools>=42' wheel 'pybind11==2.11.1' \
    'numpy<2' 'scipy<1.12' 'pandas<2' nibabel torch
  uv pip install --offline --no-build-isolation --python "$VENV_DIR/bin/python" "$LOADGEN_DIR"
  "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("python stack:", "torch", torch.__version__, "cuda", torch.version.cuda,
      "numpy", numpy.__version__, "scipy", scipy.__version__,
      "pandas", pandas.__version__, "nibabel", nibabel.__version__)
assert torch.cuda.is_available(), "CUDA is unavailable"
PY
  touch "$MARKER_DIR/setup.success"
  echo "stage setup: complete $(date -Is)"
}

stage_preprocess() {
  exec > >(tee "$OUTPUT_DIR/preprocess.log") 2>&1
  echo "stage preprocess: start $(date -Is)"
  test -f "$MARKER_DIR/setup.success"
  rm -rf "$RAW_VIEW" "$PREPROC_DIR" "$POSTPROC_DIR"
  mkdir -p "$RAW_VIEW" "$PREPROC_DIR" "$POSTPROC_DIR"

  # Restore the full benchmark metadata before an optional bounded preflight.
  cp "$REPO_DIR/vision/medical_imaging/3d-unet-kits19/meta/inference_cases.json" "$APP_DIR/meta/inference_cases.json"
  cp "$REPO_DIR/vision/medical_imaging/3d-unet-kits19/meta/checksum_inference.json" "$APP_DIR/meta/checksum_inference.json"
  cp "$REPO_DIR/loadgen/mlperf.conf" "$APP_DIR/mlperf.conf"
  if [[ -n "${REPRO_CASE_LIMIT:-}" ]]; then
    "$VENV_DIR/bin/python" - "$APP_DIR/meta/inference_cases.json" "$APP_DIR/meta/checksum_inference.json" "$REPRO_CASE_LIMIT" <<'PY'
import json, sys
manifest_path, checksum_path, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(manifest_path) as f:
    cases = json.load(f)[:limit]
with open(checksum_path) as f:
    checksums = json.load(f)
with open(manifest_path, "w") as f:
    json.dump(cases, f, indent=2)
with open(checksum_path, "w") as f:
    json.dump({case + ".pkl": checksums[case + ".pkl"] for case in cases}, f, indent=2, sort_keys=True)
print("bounded preflight cases:", cases)
PY
    sed -i "s/^3d-unet\.Offline\.min_query_count = .*/3d-unet.Offline.min_query_count = $REPRO_CASE_LIMIT/" "$APP_DIR/mlperf.conf"
  fi

  while IFS= read -r case_name; do
    source_case=$case_name
    [[ "$case_name" == case_00400 ]] && source_case=case_00185
    ln -s "$KITS19_RAW_DIR/$source_case" "$RAW_VIEW/$case_name"
  done < <("$VENV_DIR/bin/python" - "$APP_DIR/meta/inference_cases.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print("\n".join(json.load(f)))
PY
  )

  cd "$APP_DIR"
  "$VENV_DIR/bin/python" preprocess.py \
    --raw_data_dir "$RAW_VIEW" --results_dir "$PREPROC_DIR" \
    --mode preprocess --num_proc "${PREPROCESS_PROCS:-8}"
  touch "$MARKER_DIR/preprocess.success"
  echo "stage preprocess: complete $(date -Is)"
}

stage_run() {
  exec > >(tee "$OUTPUT_DIR/run.log") 2>&1
  echo "stage run: start $(date -Is)"
  test -f "$MARKER_DIR/preprocess.success"
  rm -rf "$POSTPROC_DIR"
  mkdir -p "$POSTPROC_DIR"
  rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt"

  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  echo "GPU free memory before run: ${free_mib} MiB"
  if [[ "$free_mib" -lt "${MIN_GPU_FREE_MIB:-24000}" ]]; then
    echo "insufficient free GPU memory: ${free_mib} MiB" >&2
    exit 1
  fi

  cd "$APP_DIR"
  LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 \
    "$VENV_DIR/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_FILE" --preprocessed_data_dir="$PREPROC_DIR" \
      --mlperf_conf="$APP_DIR/mlperf.conf" --user_conf="$APP_DIR/user.conf"

  "$VENV_DIR/bin/python" accuracy_kits.py \
    --preprocessed_data_dir="$PREPROC_DIR" \
    --postprocessed_data_dir="$POSTPROC_DIR" \
    --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --num_proc "${EVAL_PROCS:-8}" | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  touch "$MARKER_DIR/run.success"
  echo "stage run: complete $(date -Is)"
}

run_stage() {
  case "$1" in
    setup) stage_setup ;;
    preprocess) stage_preprocess ;;
    run) stage_run ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}

if [[ "$ACTION" == stage ]]; then
  run_stage "$START_STAGE"
  exit 0
fi

start_index=0
if [[ "$ACTION" == from ]]; then
  found=0
  for i in "${!stages[@]}"; do
    if [[ "${stages[$i]}" == "$START_STAGE" ]]; then start_index=$i; found=1; break; fi
  done
  [[ "$found" == 1 ]] || { echo "unknown stage: $START_STAGE" >&2; exit 2; }
  for ((i=start_index; i<${#stages[@]}; i++)); do
    rm -f "$MARKER_DIR/${stages[$i]}.success"
  done
fi

for i in "${!stages[@]}"; do
  (( i < start_index )) && continue
  stage=${stages[$i]}
  if [[ -f "$MARKER_DIR/$stage.success" ]]; then
    echo "stage $stage: already complete"
  else
    run_stage "$stage"
  fi
done
