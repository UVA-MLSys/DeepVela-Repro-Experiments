#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --from) FROM="${2:-}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "Expected --target 3d-unet" >&2
  exit 2
fi
if [[ -n "$STAGE" && -n "$FROM" ]]; then
  echo "Use only one of --stage or --from" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/repo"
BENCH_REL="vision/medical_imaging/3d-unet-kits19"
TARGET_DIR="$ROOT/$TARGET"
WORK="$TARGET_DIR/work"
OUT="${OUTPUT_DIR:-$TARGET_DIR/out}"
if [[ "$OUT" != /* ]]; then
  OUT="$(pwd)/$OUT"
fi
VENV="$WORK/.venv"
SRC="$WORK/src"
BUILD="$WORK/build"
RAW="$WORK/raw_data"
DONE="$WORK/.done"

CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
KITS19_RAW_DIR="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR="$CACHE_ROOT/3d-unet-kits19/model"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$OUT" "$WORK" "$DONE"
stages=(environment preprocess run evaluate)

stage_index() {
  local sought="$1" i
  for i in "${!stages[@]}"; do
    [[ "${stages[$i]}" == "$sought" ]] && { echo "$i"; return 0; }
  done
  echo "Unknown stage: $sought" >&2
  return 2
}

if [[ -n "$FROM" ]]; then
  from_i="$(stage_index "$FROM")"
  for ((i=from_i; i<${#stages[@]}; i++)); do
    rm -f "$DONE/${stages[$i]}"
  done
  case "$FROM" in
    environment)
      rm -rf "$VENV" "$SRC" "$BUILD" "$RAW"
      rm -f "$OUT/environment.log" "$OUT/preprocess.log" "$OUT/run.log" \
        "$OUT/evaluator.log" "$OUT/accuracy.txt" "$OUT/mlperf_log_accuracy.json" \
        "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" ;;
    preprocess)
      rm -rf "$SRC" "$BUILD" "$RAW"
      rm -f "$OUT/preprocess.log" "$OUT/run.log" "$OUT/evaluator.log" \
        "$OUT/accuracy.txt" "$OUT/mlperf_log_accuracy.json" \
        "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" ;;
    run)
      rm -rf "$BUILD/logs" "$BUILD/postprocessed_data"
      rm -f "$OUT/run.log" "$OUT/evaluator.log" "$OUT/accuracy.txt" \
        "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
        "$OUT/mlperf_log_summary.txt" ;;
    evaluate)
      rm -rf "$BUILD/postprocessed_data"
      rm -f "$OUT/evaluator.log" "$OUT/accuracy.txt" ;;
  esac
fi

want_stage() {
  local candidate="$1"
  if [[ -n "$STAGE" ]]; then
    [[ "$candidate" == "$STAGE" ]]
  elif [[ -n "$FROM" ]]; then
    [[ "$(stage_index "$candidate")" -ge "$(stage_index "$FROM")" ]]
  else
    return 0
  fi
}

run_environment() {
  {
    echo "Creating Python environment with UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'numpy==1.26.4' 'scipy==1.10.1' 'pandas==1.5.3' \
      'nibabel==5.2.1' 'torch==2.5.1' 'mlcommons-loadgen==6.0.17'
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("pandas", pandas.__version__, "nibabel", nibabel.__version__)
print("mlperf_loadgen imported")
PY
  } 2>&1 | tee "$OUT/environment.log"
  touch "$DONE/environment"
}

run_preprocess() {
  [[ -f "$DONE/environment" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
  {
    rm -rf "$SRC" "$BUILD" "$RAW"
    mkdir -p "$SRC" "$BUILD/preprocessed_data" "$RAW"
    cp -a "$REPO/$BENCH_REL/." "$SRC/"
    cp "$REPO/loadgen/mlperf.conf" "$BUILD/mlperf.conf"

    "$VENV/bin/python" - "$SRC" "$BUILD/mlperf.conf" "${PREFLIGHT_CASES:-0}" <<'PY'
import json, pathlib, sys
src = pathlib.Path(sys.argv[1])
conf_path = pathlib.Path(sys.argv[2])
n = int(sys.argv[3])
if n:
    cases_path = src / "meta/inference_cases.json"
    checksum_path = src / "meta/checksum_inference.json"
    cases = json.loads(cases_path.read_text())[:n]
    checksums = json.loads(checksum_path.read_text())
    checksums = {f"{case}.pkl": checksums[f"{case}.pkl"] for case in cases}
    cases_path.write_text(json.dumps(cases, indent=2) + "\n")
    checksum_path.write_text(json.dumps(checksums, indent=4, sort_keys=True) + "\n")
    conf = conf_path.read_text()
    conf = conf.replace("3d-unet.Offline.min_query_count = 43",
                        f"3d-unet.Offline.min_query_count = {n}")
    conf_path.write_text(conf)
    print(f"Bounded preflight manifest: {len(cases)} case(s)")
else:
    cases = json.loads((src / "meta/inference_cases.json").read_text())
    print(f"Full manifest: {len(cases)} entries")
print("\n".join(cases))
PY

    while IFS= read -r case_name; do
      [[ -n "$case_name" ]] || continue
      source_case="$case_name"
      [[ "$case_name" == "case_00400" ]] && source_case="case_00185"
      [[ -f "$KITS19_RAW_DIR/$source_case/imaging.nii.gz" ]]
      [[ -f "$KITS19_RAW_DIR/$source_case/segmentation.nii.gz" ]]
      ln -s "$KITS19_RAW_DIR/$source_case" "$RAW/$case_name"
    done < <("$VENV/bin/python" -c \
      'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' \
      "$SRC/meta/inference_cases.json")

    cd "$SRC"
    /usr/bin/time -v "$VENV/bin/python" preprocess.py \
      --raw_data_dir "$RAW" --results_dir "$BUILD/preprocessed_data" \
      --mode preprocess --num_proc 8
  } 2>&1 | tee "$OUT/preprocess.log"
  touch "$DONE/preprocess"
}

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '
}

run_inference() {
  [[ -f "$DONE/preprocess" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  {
    free_mib="$(gpu_free_mib)"
    echo "GPU memory before run: ${free_mib} MiB free"
    if (( free_mib < 20000 )); then
      echo "Need at least 20000 MiB free for this fp32 run" >&2
      exit 1
    fi
    cd "$SRC"
    device="$($VENV/bin/python - "$MODEL_DIR/3dunet_kits19_pytorch.ptc" <<'PY'
import sys, torch
assert torch.cuda.is_available(), "CUDA is unavailable"
model = torch.jit.load(sys.argv[1], map_location="cuda:0")
print(next(model.parameters()).device)
PY
)"
    echo "pytorch device: $device"
    mkdir -p "$BUILD/logs" "$BUILD/postprocessed_data"
    export LOG_PATH="$BUILD/logs"
    export SKIP_VERIFY_ACCURACY=1
    /usr/bin/time -v "$VENV/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
      --preprocessed_data_dir="$BUILD/preprocessed_data" \
      --mlperf_conf="$BUILD/mlperf.conf" --user_conf="$SRC/user.conf"
    cp "$BUILD/logs/mlperf_log_accuracy.json" "$OUT/mlperf_log_accuracy.json"
    cp "$BUILD/logs/mlperf_log_detail.txt" "$OUT/mlperf_log_detail.txt"
    if [[ -f "$BUILD/logs/mlperf_log_summary.txt" ]]; then
      cp "$BUILD/logs/mlperf_log_summary.txt" "$OUT/mlperf_log_summary.txt"
    fi
  } 2>&1 | tee "$OUT/run.log"
  touch "$DONE/run"
}

run_evaluate() {
  [[ -f "$DONE/run" ]] || { echo "run stage has not succeeded" >&2; exit 1; }
  : > "$OUT/evaluator.log"
  : > "$OUT/evaluator.stderr.log"
  if ! {
    cd "$SRC"
    rm -rf "$BUILD/postprocessed_data"
    mkdir -p "$BUILD/postprocessed_data"
    /usr/bin/time -v "$VENV/bin/python" accuracy_kits.py \
      --log_file="$OUT/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="$BUILD/preprocessed_data" \
      --postprocessed_data_dir="$BUILD/postprocessed_data" --num_proc=8
  } > "$OUT/accuracy.txt" 2> "$OUT/evaluator.stderr.log"; then
    cat "$OUT/accuracy.txt" "$OUT/evaluator.stderr.log" > "$OUT/evaluator.log"
    cat "$OUT/evaluator.log"
    return 1
  fi
  cat "$OUT/accuracy.txt" "$OUT/evaluator.stderr.log" > "$OUT/evaluator.log"
  cat "$OUT/evaluator.log"
  touch "$DONE/evaluate"
}

if [[ -n "$STAGE" ]]; then
  stage_index "$STAGE" >/dev/null
fi

for candidate in "${stages[@]}"; do
  want_stage "$candidate" || continue
  if [[ -f "$DONE/$candidate" ]]; then
    echo "Skipping completed stage: $candidate"
    continue
  fi
  case "$candidate" in
    environment) run_environment ;;
    preprocess) run_preprocess ;;
    run) run_inference ;;
    evaluate) run_evaluate ;;
  esac
done
