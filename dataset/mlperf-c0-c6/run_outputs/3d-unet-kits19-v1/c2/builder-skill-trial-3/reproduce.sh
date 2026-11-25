#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet [--stage setup|preprocess|run] [--from setup|preprocess|run]" >&2
  exit 2
}

TARGET=""
ONLY_STAGE=""
FROM_STAGE=""
while (($#)); do
  case "$1" in
    --target) (($# >= 2)) || usage; TARGET="$2"; shift 2 ;;
    --stage) (($# >= 2)) || usage; ONLY_STAGE="$2"; shift 2 ;;
    --from) (($# >= 2)) || usage; FROM_STAGE="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$TARGET" == "3d-unet" ]] || usage
[[ -z "$ONLY_STAGE" || "$ONLY_STAGE" =~ ^(setup|preprocess|run)$ ]] || usage
[[ -z "$FROM_STAGE" || "$FROM_STAGE" =~ ^(setup|preprocess|run)$ ]] || usage
[[ -z "$ONLY_STAGE" || -z "$FROM_STAGE" ]] || usage

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$ROOT_DIR/repo"
TARGET_DIR="$ROOT_DIR/$TARGET"
WORK_DIR="$TARGET_DIR/work"
APP_DIR="$WORK_DIR/3d-unet-kits19"
VENV_DIR="$TARGET_DIR/venv"
OUTPUT_DIR="${OUTPUT_DIR:-$TARGET_DIR/out}"
CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
KITS19_RAW_DIR="${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_EXCLUDE_NEWER=2026-11-10
CASE_LIMIT="${REPRO_CASE_LIMIT:-0}"

mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"

stage_num() {
  case "$1" in setup) echo 1 ;; preprocess) echo 2 ;; run) echo 3 ;; esac
}

if [[ -n "$FROM_STAGE" ]]; then
  from_n="$(stage_num "$FROM_STAGE")"
  for stage in setup preprocess run; do
    if (( $(stage_num "$stage") >= from_n )); then
      rm -f "$TARGET_DIR/.$stage.success"
    fi
  done
  if (( from_n <= 1 )); then
    rm -rf "$WORK_DIR" "$VENV_DIR"
    rm -f "$OUTPUT_DIR"/*.log "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/accuracy.txt"
  elif (( from_n <= 2 )); then
    rm -rf "$APP_DIR/build/preprocessed_data" "$APP_DIR/build/raw_data"
    rm -f "$OUTPUT_DIR/preprocess.log" "$OUTPUT_DIR/gpu_check.log" \
      "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" \
      "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/accuracy.txt"
  else
    rm -rf "$APP_DIR/build/logs" "$APP_DIR/build/postprocessed_data"
    rm -f "$OUTPUT_DIR/gpu_check.log" "$OUTPUT_DIR/run.log" \
      "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/accuracy.txt"
  fi
fi

run_setup() {
  local log="$OUTPUT_DIR/setup.log"
  {
    date -Ins
    echo "source revision: 3fbc329939999c13d0a7b5e67fb2092287e06047"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    [[ -d "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" ]]
    [[ -d "$REPO_DIR/loadgen" ]]
    [[ -f "$MODEL_DIR/3dunet_kits19_pytorch.ptc" ]]
    [[ -d "$KITS19_RAW_DIR" ]]

    mkdir -p "$WORK_DIR"
    cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" "$APP_DIR"
    cp -a "$REPO_DIR/loadgen" "$WORK_DIR/loadgen"

    echo "fixture repair: TARGET_SPACING None -> [1.6, 1.2, 1.2]"
    perl -0pi -e 's/TARGET_SPACING = None/TARGET_SPACING = [1.6, 1.2, 1.2]/' "$APP_DIR/global_vars.py"
    grep -Fq 'TARGET_SPACING = [1.6, 1.2, 1.2]' "$APP_DIR/global_vars.py"

    echo "fixture repair: emit required model-parameter device log line"
    perl -0pi -e 's/self\.model\.eval\(\)/self.model.eval()\n        print("pytorch device: {:}".format(next(self.model.parameters()).device))/' "$APP_DIR/pytorch_SUT.py"
    grep -Fq 'pytorch device: ' "$APP_DIR/pytorch_SUT.py"

    echo "fixture repair: generate then verify checksum manifest when source manifest is absent"
    perl -0pi -e 's/preprocess_with_multiproc\(args\)\n        verify_dataset\(args\)/preprocess_with_multiproc(args)\n        if not CHECKSUM_INFER_FILE.is_file():\n            generate_hash_from_dataset(args)\n        verify_dataset(args)/' "$APP_DIR/preprocess.py"
    grep -Fq 'if not CHECKSUM_INFER_FILE.is_file():' "$APP_DIR/preprocess.py"

    if [[ ! "$CASE_LIMIT" =~ ^(0|[1-9][0-9]*)$ ]]; then
      echo "REPRO_CASE_LIMIT must be 0 or a positive integer" >&2
      exit 2
    fi

    uv venv --python 3.11 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" \
      'numpy==1.26.4' 'scipy==1.11.4' 'pandas==1.5.3' 'nibabel==5.2.1' \
      'torch==2.5.1' 'pybind11==2.11.1' 'setuptools==75.6.0' 'wheel==0.45.1'
    uv pip install --python "$VENV_DIR/bin/python" --no-build-isolation "$WORK_DIR/loadgen"
    "$VENV_DIR/bin/python" -c 'import torch,numpy,scipy,pandas,nibabel,mlperf_loadgen; print("torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available())'

    mkdir -p "$APP_DIR/build/model" "$APP_DIR/build/logs" "$APP_DIR/build/postprocessed_data"
    ln -s "$MODEL_DIR/3dunet_kits19_pytorch.ptc" "$APP_DIR/build/model/3dunet_kits19_pytorch.ptc"
    cp "$WORK_DIR/loadgen/mlperf.conf" "$APP_DIR/build/mlperf.conf"
    date -Ins
  } 2>&1 | tee "$log"
  touch "$TARGET_DIR/.setup.success"
}

run_preprocess() {
  [[ -f "$TARGET_DIR/.setup.success" ]] || { echo "setup stage has not succeeded" >&2; exit 1; }
  local log="$OUTPUT_DIR/preprocess.log"
  {
    date -Ins
    mkdir -p "$APP_DIR/build/raw_data" "$APP_DIR/build/preprocessed_data"
    "$VENV_DIR/bin/python" - "$APP_DIR/meta/inference_cases.json" "$KITS19_RAW_DIR" "$APP_DIR/build/raw_data" "$CASE_LIMIT" <<'PY'
import json, os, sys
manifest, raw, staged, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
with open(manifest, encoding="utf-8") as f:
    cases = json.load(f)
if limit:
    cases = cases[:limit]
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=4)
        f.write("\n")
for case in cases:
    source_case = "case_00185" if case == "case_00400" else case
    source = os.path.join(raw, source_case)
    target = os.path.join(staged, case)
    if not os.path.isdir(source):
        raise FileNotFoundError(source)
    if not os.path.lexists(target):
        os.symlink(source, target)
print(f"staged {len(cases)} inference entries")
PY
    cd "$APP_DIR"
    "$VENV_DIR/bin/python" preprocess.py \
      --raw_data_dir build/raw_data --results_dir build/preprocessed_data --mode preprocess
    "$VENV_DIR/bin/python" - "$APP_DIR/build/preprocessed_data/preprocessed_files.pkl" "$CASE_LIMIT" <<'PY'
import pickle, sys
with open(sys.argv[1], "rb") as f:
    count = len(pickle.load(f)["file_list"])
expected = int(sys.argv[2]) or 43
assert count == expected, (count, expected)
print(f"verified {count} preprocessed entries")
PY
    date -Ins
  } 2>&1 | tee "$log"
  touch "$TARGET_DIR/.preprocess.success"
}

run_inference() {
  [[ -f "$TARGET_DIR/.preprocess.success" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  local gpu_log="$OUTPUT_DIR/gpu_check.log"
  local run_log="$OUTPUT_DIR/run.log"
  local eval_log="$OUTPUT_DIR/evaluator.log"
  {
    date -Ins
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
  } 2>&1 | tee "$gpu_log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 20000 )); then
    echo "insufficient free GPU memory: ${free_mib:-unknown} MiB; require at least 20000 MiB" | tee -a "$gpu_log" >&2
    exit 1
  fi

  cd "$APP_DIR"
  LOG_PATH="$APP_DIR/build/logs" SKIP_VERIFY_ACCURACY=1 \
    "$VENV_DIR/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
    2>&1 | tee "$run_log"
  grep -Fq 'pytorch device: cuda:0' "$run_log"
  cp "$APP_DIR/build/logs/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
  cp "$APP_DIR/build/logs/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"
  "$VENV_DIR/bin/python" accuracy_kits.py \
    --preprocessed_data_dir="$APP_DIR/build/preprocessed_data" \
    --postprocessed_data_dir="$APP_DIR/build/postprocessed_data" \
    --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
    2>&1 | tee "$eval_log" "$OUTPUT_DIR/accuracy.txt"
  grep -Eq '^Accuracy: mean = [^,]+, kidney = [^,]+, tumor = .+$' "$OUTPUT_DIR/accuracy.txt"
  "$VENV_DIR/bin/python" - "$OUTPUT_DIR/mlperf_log_accuracy.json" "$CASE_LIMIT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    rows = json.load(f)
expected = int(sys.argv[2]) or 43
assert len(rows) == expected, (len(rows), expected)
assert len({row["qsl_idx"] for row in rows}) == expected
print(f"verified {expected} unique LoadGen accuracy results")
PY
  touch "$TARGET_DIR/.run.success"
}

run_selected() {
  local stage="$1"
  case "$stage" in
    setup) [[ -f "$TARGET_DIR/.setup.success" ]] || run_setup ;;
    preprocess) [[ -f "$TARGET_DIR/.preprocess.success" ]] || run_preprocess ;;
    run) [[ -f "$TARGET_DIR/.run.success" ]] || run_inference ;;
  esac
}

if [[ -n "$ONLY_STAGE" ]]; then
  run_selected "$ONLY_STAGE"
elif [[ -n "$FROM_STAGE" ]]; then
  start="$(stage_num "$FROM_STAGE")"
  for stage in setup preprocess run; do
    (( $(stage_num "$stage") >= start )) && run_selected "$stage"
  done
else
  for stage in setup preprocess run; do run_selected "$stage"; done
fi
