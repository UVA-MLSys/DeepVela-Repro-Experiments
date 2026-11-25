#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET=${2:?missing target id}; shift 2 ;;
    --stage) STAGE=${2:?missing stage name}; shift 2 ;;
    --from) FROM=${2:?missing stage name}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "--target must be 3d-unet" >&2
  exit 2
fi
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

ROOT=$(cd "$(dirname "$0")" && pwd)
REPO="$ROOT/repo"
OUT=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)
WORK=$(cd "$OUT/.." && pwd)/work
SRC="$WORK/src"
ENV_DIR="$WORK/venv"
PREP="$WORK/preprocessed_data"
POST="$WORK/postprocessed_data"
RAW="$WORK/raw_data"
MODEL=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc
RAW_INPUT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache/3d-unet-kits19/kits19_raw_data/kits19/data
UV_CACHE_DIR=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache/uv
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

stages=(environment data run evaluate)
valid_stage() {
  local wanted=$1 item
  for item in "${stages[@]}"; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}
if [[ -n "$STAGE" ]] && ! valid_stage "$STAGE"; then echo "invalid stage: $STAGE" >&2; exit 2; fi
if [[ -n "$FROM" ]] && ! valid_stage "$FROM"; then echo "invalid stage: $FROM" >&2; exit 2; fi
if [[ -n "$STAGE" && -n "$FROM" ]]; then echo "use only one of --stage and --from" >&2; exit 2; fi

clear_from() {
  local start=$1 active=0 item
  for item in "${stages[@]}"; do
    [[ "$item" == "$start" ]] && active=1
    if [[ $active -eq 1 ]]; then
      rm -f "$WORK/.${item}.done"
      case "$item" in
        environment) rm -rf "$SRC" "$ENV_DIR" ;;
        data) rm -rf "$PREP" "$RAW" ;;
        run) rm -rf "$POST"; rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json" ;;
        evaluate) rm -rf "$POST"; rm -f "$OUT/accuracy.txt" "$OUT/evaluator.log" ;;
      esac
    fi
  done
}

stage_environment() {
  [[ -f "$WORK/.environment.done" ]] && return
  mkdir -p "$WORK"
  {
    echo "Creating Python 3.11 environment from shared package cache"
    rm -rf "$SRC" "$ENV_DIR"
    cp -a "$REPO/vision/medical_imaging/3d-unet-kits19" "$SRC"
    cp "$REPO/mlperf.conf" "$SRC/mlperf.conf"
    sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$SRC/pytorch_SUT.py"
    uv venv --python 3.11 "$ENV_DIR"
    uv pip install --offline --python "$ENV_DIR/bin/python" \
      'torch==2.4.1' 'numpy==1.26.4' 'scipy==1.11.4' \
      'pandas==1.5.3' 'nibabel==5.2.1' "$REPO/loadgen"
    "$ENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("pandas", pandas.__version__, "nibabel", nibabel.__version__)
PY
  } 2>&1 | tee "$OUT/build.log"
  touch "$WORK/.environment.done"
}

stage_data() {
  [[ -f "$WORK/.data.done" ]] && return
  [[ -f "$WORK/.environment.done" ]] || stage_environment
  (cd "$SRC" && {
    rm -rf "$PREP" "$RAW"
    mkdir -p "$PREP" "$RAW"
    cp "$REPO/vision/medical_imaging/3d-unet-kits19/meta/inference_cases.json" meta/inference_cases.json
    cp "$REPO/vision/medical_imaging/3d-unet-kits19/meta/checksum_inference.json" meta/checksum_inference.json
    local case_name
    if [[ -n "${PREFLIGHT_CASES:-}" ]]; then
      IFS=, read -ra selected <<<"$PREFLIGHT_CASES"
    else
      mapfile -t selected < <("$ENV_DIR/bin/python" -c 'import json; print(*json.load(open("meta/inference_cases.json")), sep="\n")' < /dev/null)
    fi
    for case_name in "${selected[@]}"; do
      if [[ "$case_name" == "case_00400" ]]; then
        ln -s "$RAW_INPUT/case_00185" "$RAW/case_00400"
      else
        ln -s "$RAW_INPUT/$case_name" "$RAW/$case_name"
      fi
    done
    if [[ -n "${PREFLIGHT_CASES:-}" ]]; then
      SELECTED_CASES="$PREFLIGHT_CASES" "$ENV_DIR/bin/python" - <<'PY'
import json, os
cases = os.environ["SELECTED_CASES"].split(",")
with open("meta/inference_cases.json", "w") as f:
    json.dump(cases, f)
with open("meta/checksum_inference.json") as f:
    checksums = json.load(f)
with open("meta/checksum_inference.json", "w") as f:
    json.dump({c + ".pkl": checksums[c + ".pkl"] for c in cases}, f, indent=4, sort_keys=True)
PY
    fi
    "$ENV_DIR/bin/python" preprocess.py --raw_data_dir "$RAW" --results_dir "$PREP" --mode preprocess --num_proc "${PREPROCESS_PROCS:-16}"
  }) 2>&1 | tee "$OUT/preprocess.log"
  touch "$WORK/.data.done"
}

stage_run() {
  [[ -f "$WORK/.run.done" ]] && return
  [[ -f "$WORK/.data.done" ]] || stage_data
  (cd "$SRC" && {
    echo "GPU memory immediately before inference:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 16384 )); then
      echo "Need at least 16384 MiB free GPU memory; found ${free_mib:-unknown}" >&2
      exit 1
    fi
    rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json"
    LOG_PATH="$OUT" SKIP_VERIFY_ACCURACY=1 "$ENV_DIR/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy --model="$MODEL" \
      --preprocessed_data_dir="$PREP" --mlperf_conf=mlperf.conf --user_conf=user.conf
  }) 2>&1 | tee "$OUT/run.log"
  test -s "$OUT/mlperf_log_accuracy.json"
  test -s "$OUT/mlperf_log_detail.txt"
  touch "$WORK/.run.done"
}

stage_evaluate() {
  [[ -f "$WORK/.evaluate.done" ]] && return
  [[ -f "$WORK/.run.done" ]] || stage_run
  rm -rf "$POST"
  mkdir -p "$POST"
  (cd "$SRC" && "$ENV_DIR/bin/python" accuracy_kits.py \
    --log_file="$OUT/mlperf_log_accuracy.json" \
    --preprocessed_data_dir="$PREP" --postprocessed_data_dir="$POST" \
    --num_proc "${EVALUATE_PROCS:-16}") 2>&1 | tee "$OUT/accuracy.txt" "$OUT/evaluator.log"
  rg -q '^Accuracy: mean = [0-9.]+, kidney = [0-9.]+, tumor = [0-9.]+' "$OUT/accuracy.txt"
  touch "$WORK/.evaluate.done"
}

mkdir -p "$WORK"
if [[ -n "$FROM" ]]; then clear_from "$FROM"; fi
if [[ -n "$STAGE" ]]; then
  "stage_${STAGE}"
else
  start_index=0
  if [[ -n "$FROM" ]]; then
    for i in "${!stages[@]}"; do [[ "${stages[$i]}" == "$FROM" ]] && start_index=$i; done
  fi
  for ((i=start_index; i<${#stages[@]}; i++)); do "stage_${stages[$i]}"; done
fi
