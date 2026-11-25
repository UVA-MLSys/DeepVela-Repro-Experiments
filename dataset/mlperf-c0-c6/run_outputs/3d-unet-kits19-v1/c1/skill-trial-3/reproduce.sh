#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM_STAGE=""
PREFLIGHT_CASE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --from) FROM_STAGE="$2"; shift 2 ;;
    --preflight-case) PREFLIGHT_CASE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "Expected --target 3d-unet" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
if [[ ! -d "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" ]]; then
  echo "Repository not found beside reproduce.sh: $REPO_DIR" >&2
  exit 1
fi

CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
export XDG_CACHE_HOME="$UV_CACHE_DIR/home/.cache"
export TORCH_HOME="$UV_CACHE_DIR/torch"
export TRITON_CACHE_DIR="$UV_CACHE_DIR/triton"
export CUDA_CACHE_PATH="$UV_CACHE_DIR/cuda"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$SCRIPT_DIR/$TARGET/out"
fi
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
TARGET_DIR="$(dirname "$OUTPUT_DIR")"
if [[ -n "$PREFLIGHT_CASE" ]]; then
  WORK_DIR="$TARGET_DIR/work-preflight"
else
  WORK_DIR="$TARGET_DIR/work"
fi
APP_DIR="$WORK_DIR/app"
VENV_DIR="$WORK_DIR/.venv"

stages=(setup preprocess run evaluate)
valid_stage() {
  local wanted="$1" item
  for item in "${stages[@]}"; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}
if [[ -n "$STAGE" ]] && ! valid_stage "$STAGE"; then echo "Invalid stage: $STAGE" >&2; exit 2; fi
if [[ -n "$FROM_STAGE" ]] && ! valid_stage "$FROM_STAGE"; then echo "Invalid stage: $FROM_STAGE" >&2; exit 2; fi
if [[ -n "$STAGE" && -n "$FROM_STAGE" ]]; then echo "Use only one of --stage and --from" >&2; exit 2; fi

reset_from() {
  local start="$1" active=0 item
  for item in "${stages[@]}"; do
    [[ "$item" == "$start" ]] && active=1
    if [[ "$active" -eq 1 ]]; then
      rm -f "$WORK_DIR/.$item.done"
      case "$item" in
        setup) rm -rf "$WORK_DIR" ;;
        preprocess) rm -rf "$APP_DIR/build/preprocessed_data" ;;
        run) rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" ;;
        evaluate) rm -rf "$APP_DIR/build/postprocessed_data"; rm -f "$OUTPUT_DIR/accuracy.txt" ;;
      esac
    fi
  done
}
if [[ -n "$FROM_STAGE" ]]; then reset_from "$FROM_STAGE"; fi

run_setup() {
  [[ -f "$WORK_DIR/.setup.done" ]] && return
  mkdir -p "$WORK_DIR/src" "$OUTPUT_DIR"
  {
    echo "stage: setup"
    echo "source: $REPO_DIR"
    echo "uv cache: $UV_CACHE_DIR"
    cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" "$APP_DIR"
    cp -a "$REPO_DIR/loadgen" "$WORK_DIR/src/loadgen"
    command -v python3.11
    uv venv --python "$(command -v python3.11)" "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" --offline \
      'torch==2.4.1' 'numpy==1.26.4' 'scipy==1.12.0' 'pandas==1.5.3' \
      'nibabel==5.4.2' 'pybind11==2.11.1' setuptools wheel
    uv pip install --python "$VENV_DIR/bin/python" --offline --no-build-isolation "$WORK_DIR/src/loadgen"
    mkdir -p "$APP_DIR/build/raw_data" "$APP_DIR/build/model"
    "$VENV_DIR/bin/python" - "$APP_DIR/meta/inference_cases.json" "$APP_DIR/meta/checksum_inference.json" "$RAW_INPUT" "$APP_DIR/build/raw_data" "$PREFLIGHT_CASE" <<'PY'
import json, os, pathlib, sys
manifest, checksum_manifest, raw, dest, preflight = sys.argv[1:]
cases = json.load(open(manifest))
if preflight:
    if preflight not in cases:
        raise SystemExit(f"preflight case not in manifest: {preflight}")
    cases = [preflight]
    with open(manifest, "w") as f:
        json.dump(cases, f, indent=4)
    checksums = json.load(open(checksum_manifest))
    checksum_key = f"{preflight}.pkl"
    with open(checksum_manifest, "w") as f:
        json.dump({checksum_key: checksums[checksum_key]}, f, indent=4)
for case in cases:
    source_case = "case_00185" if case == "case_00400" else case
    source = pathlib.Path(raw, source_case)
    for filename in ("imaging.nii.gz", "segmentation.nii.gz"):
        if not (source / filename).is_file():
            raise SystemExit(f"missing input: {source / filename}")
    os.symlink(source, pathlib.Path(dest, case), target_is_directory=True)
print(f"prepared {len(cases)} manifest entries")
PY
    ln -s "$MODEL_INPUT" "$APP_DIR/build/model/3dunet_kits19_pytorch.ptc"
    cp "$REPO_DIR/loadgen/mlperf.conf" "$APP_DIR/build/mlperf.conf"
    if [[ -n "$PREFLIGHT_CASE" ]]; then
      "$VENV_DIR/bin/python" - "$APP_DIR/build/mlperf.conf" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
text = p.read_text()
old = "3d-unet.Offline.min_query_count = 43"
if old not in text:
    raise SystemExit(f"expected LoadGen count setting not found: {old}")
p.write_text(text.replace(old, "3d-unet.Offline.min_query_count = 1"))
PY
    fi
    "$VENV_DIR/bin/python" - <<'PY'
import nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "nibabel", nibabel.__version__)
PY
  } 2>&1 | tee "$OUTPUT_DIR/setup.log"
  touch "$WORK_DIR/.setup.done"
}

run_preprocess() {
  [[ -f "$WORK_DIR/.preprocess.done" ]] && return
  run_setup
  {
    echo "stage: preprocess"
    cd "$APP_DIR"
    "$VENV_DIR/bin/python" preprocess.py --raw_data_dir build/raw_data --results_dir build/preprocessed_data --mode preprocess --num_proc 4
    "$VENV_DIR/bin/python" - <<'PY'
import pickle
d = pickle.load(open("build/preprocessed_data/preprocessed_files.pkl", "rb"))
print("preprocessed manifest entries:", len(d["file_list"]))
assert len(d["file_list"]) == len(d["cases"])
PY
  } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  touch "$WORK_DIR/.preprocess.done"
}

run_inference() {
  [[ -f "$WORK_DIR/.run.done" ]] && return
  run_preprocess
  {
    echo "stage: run"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -dc '0-9')"
    if [[ -z "$free_mib" || "$free_mib" -lt 20000 ]]; then
      echo "Insufficient free GPU memory: ${free_mib:-unknown} MiB; require 20000 MiB" >&2
      exit 1
    fi
    echo "pytorch device: cuda:0"
    cd "$APP_DIR"
    LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 "$VENV_DIR/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model=build/model/3dunet_kits19_pytorch.ptc \
      --preprocessed_data_dir=build/preprocessed_data \
      --mlperf_conf=build/mlperf.conf --user_conf=user.conf
    test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
    test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  } 2>&1 | tee "$OUTPUT_DIR/run.log"
  touch "$WORK_DIR/.run.done"
}

run_evaluate() {
  [[ -f "$WORK_DIR/.evaluate.done" ]] && return
  run_inference
  cd "$APP_DIR"
  {
    echo "stage: evaluate"
    "$VENV_DIR/bin/python" accuracy_kits.py \
      --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --preprocessed_data_dir=build/preprocessed_data \
      --postprocessed_data_dir=build/postprocessed_data --num_proc=4
  } 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  grep -E '^Accuracy: mean = [0-9.]+, kidney = [0-9.]+, tumor = [0-9.]+' "$OUTPUT_DIR/accuracy.txt"
  touch "$WORK_DIR/.evaluate.done"
}

run_named_stage() {
  case "$1" in
    setup) run_setup ;;
    preprocess) run_preprocess ;;
    run) run_inference ;;
    evaluate) run_evaluate ;;
  esac
}

if [[ -n "$STAGE" ]]; then
  run_named_stage "$STAGE"
elif [[ -n "$FROM_STAGE" ]]; then
  active=0
  for item in "${stages[@]}"; do
    [[ "$item" == "$FROM_STAGE" ]] && active=1
    [[ "$active" -eq 1 ]] && run_named_stage "$item"
  done
else
  for item in "${stages[@]}"; do run_named_stage "$item"; done
fi
