#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { echo "--target requires a value" >&2; exit 2; }
      TARGET="$2"; shift 2 ;;
    --stage)
      [[ $# -ge 2 ]] || { echo "--stage requires a value" >&2; exit 2; }
      MODE="stage"; START_STAGE="$2"; shift 2 ;;
    --from)
      [[ $# -ge 2 ]] || { echo "--from requires a value" >&2; exit 2; }
      MODE="from"; START_STAGE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$TARGET" == "3d-unet" ]] || { echo "supported target: 3d-unet" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
TARGET_DIR="$SCRIPT_DIR/$TARGET"
SOURCE_DIR="$TARGET_DIR/source"
VENV_DIR="$TARGET_DIR/venv"
RAW_VIEW="$TARGET_DIR/raw_data"
PREPROCESSED_DIR="$TARGET_DIR/preprocessed_data"
POSTPROCESSED_DIR="$TARGET_DIR/postprocessed_data"
DEFAULT_OUTPUT_DIR="$TARGET_DIR/out"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
UV_CACHE_DIR="$CACHE_ROOT/uv"
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
PYTHON_SOURCE="/home/mi3se/.local/bin/python3.11"
CASE_LIMIT="${REPRO_CASE_LIMIT:-0}"
MIN_FREE_MIB="${REPRO_MIN_FREE_MIB:-20000}"
STAGES=(env data gpu evaluate)

case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR" ;;
esac

mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"

stage_index() {
  local wanted="$1" i
  for i in "${!STAGES[@]}"; do
    [[ "${STAGES[$i]}" == "$wanted" ]] && { echo "$i"; return 0; }
  done
  echo "unknown stage: $wanted (choose env, data, gpu, evaluate)" >&2
  return 2
}

remove_stage_outputs() {
  local first="$1" i stage
  for ((i=first; i<${#STAGES[@]}; i++)); do
    stage="${STAGES[$i]}"
    rm -f "$TARGET_DIR/.${stage}.done"
  done
  if (( first <= 0 )); then
    rm -rf "$SOURCE_DIR" "$VENV_DIR"
    rm -f "$TARGET_DIR/env.log"
  fi
  if (( first <= 1 )); then
    rm -rf "$RAW_VIEW" "$PREPROCESSED_DIR"
    rm -f "$TARGET_DIR/preprocess.log"
  fi
  if (( first <= 2 )); then
    rm -rf "$POSTPROCESSED_DIR"
    rm -f "$TARGET_DIR/run.log" "$TARGET_DIR/gpu-before.log"
    rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt"
  fi
  if (( first <= 3 )); then
    rm -f "$TARGET_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  fi
}

run_env() {
  {
    echo "stage=env started=$(date -Is)"
    test -d "$REPO_DIR/vision/medical_imaging/3d-unet-kits19"
    test -d "$REPO_DIR/loadgen"
    test -d "$RAW_INPUT"
    test -s "$MODEL_INPUT"
    test -d "$UV_CACHE_DIR"
    test -x "$PYTHON_SOURCE"

    cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" "$SOURCE_DIR"
    cp -a "$REPO_DIR/loadgen" "$SOURCE_DIR/loadgen"
    cp "$REPO_DIR/loadgen/mlperf.conf" "$SOURCE_DIR/build_mlperf.conf"

    # Required observability repair on the target-owned source copy.
    sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(self.device))' \
      "$SOURCE_DIR/pytorch_SUT.py"
    "$PYTHON_SOURCE" - "$SOURCE_DIR/run.py" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = '    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)\n'
assert text.count(old) == 1
path.write_text(text.replace(old, ""), encoding="utf-8")
PY

    if [[ "$CASE_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
      "$PYTHON_SOURCE" - "$SOURCE_DIR/meta/inference_cases.json" \
        "$SOURCE_DIR/meta/checksum_inference.json" "$CASE_LIMIT" <<'PY'
import json, sys
manifest, checksums, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
cases = json.load(open(manifest, encoding="utf-8"))[:count]
with open(manifest, "w", encoding="utf-8") as f:
    json.dump(cases, f, indent=2)
    f.write("\n")
all_hashes = json.load(open(checksums, encoding="utf-8"))
subset = {f"{case}.pkl": all_hashes[f"{case}.pkl"] for case in cases}
with open(checksums, "w", encoding="utf-8") as f:
    json.dump(subset, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"bounded manifest entries={len(cases)}")
PY
    fi

    UV_CACHE_DIR="$UV_CACHE_DIR" UV_EXCLUDE_NEWER=2026-11-10 \
      uv venv --python "$PYTHON_SOURCE" "$VENV_DIR"
    UV_CACHE_DIR="$UV_CACHE_DIR" UV_EXCLUDE_NEWER=2026-11-10 \
      uv pip install --python "$VENV_DIR/bin/python" --offline \
      torch==2.7.1 numpy==1.26.4 scipy==1.11.4 pandas==1.5.3 \
      nibabel==5.2.1 pybind11==2.11.1 setuptools==84.2.1 wheel==0.48.0
    UV_CACHE_DIR="$UV_CACHE_DIR" UV_EXCLUDE_NEWER=2026-11-10 \
      uv pip install --python "$VENV_DIR/bin/python" --offline --no-build-isolation \
      "$SOURCE_DIR/loadgen"
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("python dependency import check: ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "nibabel", nibabel.__version__)
PY
    echo "stage=env finished=$(date -Is)"
  } 2>&1 | tee "$TARGET_DIR/env.log"
  touch "$TARGET_DIR/.env.done"
}

run_data() {
  {
    echo "stage=data started=$(date -Is)"
    mkdir -p "$RAW_VIEW" "$PREPROCESSED_DIR"
    "$VENV_DIR/bin/python" - "$SOURCE_DIR/meta/inference_cases.json" "$RAW_INPUT" "$RAW_VIEW" <<'PY'
import json, os, pathlib, sys
manifest, source_root, view_root = map(pathlib.Path, sys.argv[1:])
cases = json.load(open(manifest, encoding="utf-8"))
for case in cases:
    source_case = "case_00185" if case == "case_00400" else case
    source = source_root / source_case
    assert (source / "imaging.nii.gz").is_file(), source
    assert (source / "segmentation.nii.gz").is_file(), source
    destination = view_root / case
    if not destination.exists():
        os.symlink(source, destination, target_is_directory=True)
print(f"raw view entries={len(cases)}")
PY
    cd "$SOURCE_DIR"
    "$VENV_DIR/bin/python" preprocess.py \
      --raw_data_dir "$RAW_VIEW" --results_dir "$PREPROCESSED_DIR" \
      --mode preprocess --num_proc 8
    echo "stage=data finished=$(date -Is)"
  } 2>&1 | tee "$TARGET_DIR/preprocess.log"
  touch "$TARGET_DIR/.data.done"
}

run_gpu() {
  {
    echo "stage=gpu started=$(date -Is)"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,driver_version \
      --format=csv,noheader | tee "$TARGET_DIR/gpu-before.log"
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "could not parse free GPU memory" >&2; exit 1; }
    if (( free_mib < MIN_FREE_MIB )); then
      echo "GPU free memory ${free_mib} MiB is below required ${MIN_FREE_MIB} MiB" >&2
      exit 1
    fi
    mkdir -p "$POSTPROCESSED_DIR" "$OUTPUT_DIR" "$SOURCE_DIR/build"
    "$VENV_DIR/bin/python" - "$SOURCE_DIR/build_mlperf.conf" "$SOURCE_DIR/user.conf" \
      "$SOURCE_DIR/build/mlperf.conf" <<'PY'
import pathlib, sys
base, user, output = map(pathlib.Path, sys.argv[1:])
output.write_text(base.read_text(encoding="utf-8") + "\n" + user.read_text(encoding="utf-8"), encoding="utf-8")
PY
    rm -f "$SOURCE_DIR/audit.conf"
    cd "$SOURCE_DIR"
    CUDA_VISIBLE_DEVICES=0 LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 \
      "$VENV_DIR/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model "$MODEL_INPUT" --preprocessed_data_dir "$PREPROCESSED_DIR" \
      --mlperf_conf "$SOURCE_DIR/build/mlperf.conf" --user_conf "$SOURCE_DIR/user.conf" \
      --audit_conf "$SOURCE_DIR/audit.conf"
    test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
    test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
    grep -F 'pytorch device: ' "$TARGET_DIR/run.log" >/dev/null || true
    echo "stage=gpu finished=$(date -Is)"
  } 2>&1 | tee "$TARGET_DIR/run.log"
  grep -F 'pytorch device: ' "$TARGET_DIR/run.log" >/dev/null
  touch "$TARGET_DIR/.gpu.done"
}

run_evaluate() {
  {
    echo "stage=evaluate started=$(date -Is)"
    cd "$SOURCE_DIR"
    "$VENV_DIR/bin/python" accuracy_kits.py \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --preprocessed_data_dir "$PREPROCESSED_DIR" \
      --postprocessed_data_dir "$POSTPROCESSED_DIR" --num_proc 8
    echo "stage=evaluate finished=$(date -Is)"
  } 2>&1 | tee "$TARGET_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  grep -E '^Accuracy: mean = [0-9.]+, kidney = [0-9.]+, tumor = [0-9.]+$' \
    "$OUTPUT_DIR/accuracy.txt" >/dev/null
  touch "$TARGET_DIR/.evaluate.done"
}

if [[ "$MODE" == "all" ]]; then
  FIRST=0
  remove_stage_outputs "$FIRST"
elif [[ "$MODE" == "from" ]]; then
  FIRST="$(stage_index "$START_STAGE")"
  remove_stage_outputs "$FIRST"
else
  FIRST="$(stage_index "$START_STAGE")"
fi

for i in "${!STAGES[@]}"; do
  (( i < FIRST )) && continue
  stage="${STAGES[$i]}"
  if [[ "$MODE" == "stage" && "$stage" != "$START_STAGE" ]]; then
    continue
  fi
  marker="$TARGET_DIR/.${stage}.done"
  if [[ -f "$marker" ]]; then
    echo "stage=$stage already complete"
    continue
  fi
  case "$stage" in
    env) run_env ;;
    data) test -f "$TARGET_DIR/.env.done"; run_data ;;
    gpu) test -f "$TARGET_DIR/.data.done"; run_gpu ;;
    evaluate) test -f "$TARGET_DIR/.gpu.done"; run_evaluate ;;
  esac
done
