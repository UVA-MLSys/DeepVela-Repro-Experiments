#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
PREFLIGHT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET=${2:?missing target id}; shift 2 ;;
    --stage) STAGE=${2:?missing stage name}; shift 2 ;;
    --from) FROM=${2:?missing stage name}; shift 2 ;;
    --preflight) PREFLIGHT=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "unsupported target: $TARGET" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH="$ROOT/repo/vision/medical_imaging/3d-unet-kits19"
LOADGEN="$ROOT/repo/loadgen"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

if [[ $PREFLIGHT -eq 1 ]]; then
  WORK="$ROOT/3d-unet/preflight"
  OUT="$ROOT/3d-unet/preflight/out"
else
  # Keep private intermediates separate from concurrent reproductions sharing the workspace.
  WORK="$ROOT/3d-unet/.codex-reproduce-work-3fbc329"
  OUT=${OUTPUT_DIR:-"$ROOT/3d-unet/out"}
  [[ "$OUT" = /* ]] || OUT="$ROOT/$OUT"
fi
SRC="$WORK/source"
VENV="$WORK/venv"
PREP="$WORK/preprocessed_data"
POST="$WORK/postprocessed_data"
RAW="$WORK/raw_data"
LOGS="$OUT"
mkdir -p "$OUT" "$WORK"

stages=(environment preprocess inference evaluate)

run_environment() {
  if [[ ! -f "$WORK/.environment.ok" ]]; then
    rm -rf "$SRC" "$VENV" "$WORK/loadgen-source"
    cp -a "$BENCH" "$SRC"
    cp -a "$LOADGEN" "$WORK/loadgen-source"
    # The task source intentionally masks the documented KiTS19 common spacing.
    sed -i 's/^TARGET_SPACING = None$/TARGET_SPACING = [1.6, 1.2, 1.2]/' "$SRC/global_vars.py"
    # Required task evidence: report the selected parameter device.
    sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$SRC/pytorch_SUT.py"
    # LoadGen 6 rejects multiple FromConfig calls; retain both files' values in one.
    sed -i '/settings.FromConfig(args.user_conf, "3d-unet", args.scenario)/d' "$SRC/run.py"
    cp "$WORK/loadgen-source/mlperf.conf" "$WORK/combined.conf"
    cat "$SRC/user.conf" >> "$WORK/combined.conf"
    uv venv --python python3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'numpy==1.26.4' 'scipy==1.11.4' 'nibabel==5.3.2' \
      'pandas==1.5.3' 'torch==2.7.1'
    uv pip install --python "$VENV/bin/python" "$WORK/loadgen-source"
    touch "$WORK/.environment.ok"
  fi
}

run_preprocess() {
  if [[ ! -f "$WORK/.preprocess.ok" ]]; then
    rm -rf "$PREP" "$RAW"
    mkdir -p "$PREP" "$RAW"
    if [[ $PREFLIGHT -eq 1 ]]; then
      printf '["case_00061"]\n' > "$SRC/meta/inference_cases.json"
      cases=(case_00061)
    else
      mapfile -t cases < <(cd "$SRC" && "$VENV/bin/python" -c 'import json; print(*json.load(open("meta/inference_cases.json")), sep="\n")')
    fi
    for case_id in "${cases[@]}"; do
      source_id=$case_id
      [[ "$case_id" == case_00400 ]] && source_id=case_00185
      ln -s "$RAW_INPUT/$source_id" "$RAW/$case_id"
    done
    (cd "$SRC" && \
      "$VENV/bin/python" -c \
        'import preprocess; args=preprocess.parse_args(); preprocess.preprocess_with_multiproc(args)' \
        --raw_data_dir "$RAW" --results_dir "$PREP" --mode preprocess --num_proc 8 && \
      "$VENV/bin/python" preprocess.py \
        --raw_data_dir "$RAW" --results_dir "$PREP" --mode gen_hash --num_proc 8 && \
      "$VENV/bin/python" preprocess.py \
        --raw_data_dir "$RAW" --results_dir "$PREP" --mode verify --num_proc 8)
    touch "$WORK/.preprocess.ok"
  fi
}

run_inference() {
  if [[ ! -f "$WORK/.inference.ok" ]]; then
    rm -rf "$POST"
    mkdir -p "$POST"
    rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
      "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt"
    gpu_state=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader)
    echo "GPU before run: $gpu_state"
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( free_mib < 20000 )); then
      echo "insufficient free GPU memory: ${free_mib} MiB (need at least 20000 MiB)" >&2
      exit 1
    fi
    (cd "$SRC" && \
      LOG_PATH="$OUT" SKIP_VERIFY_ACCURACY=1 CUDA_VISIBLE_DEVICES=0 \
      "$VENV/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
        --model="$MODEL_INPUT" --preprocessed_data_dir="$PREP" \
        --mlperf_conf="$WORK/combined.conf" --user_conf=user.conf)
    test -s "$OUT/mlperf_log_accuracy.json"
    test -s "$OUT/mlperf_log_detail.txt"
    touch "$WORK/.inference.ok"
  fi
}

run_evaluate() {
  if [[ ! -f "$WORK/.evaluate.ok" ]]; then
    rm -rf "$POST"
    mkdir -p "$POST"
    (cd "$SRC" && "$VENV/bin/python" accuracy_kits.py \
      --preprocessed_data_dir="$PREP" \
      --postprocessed_data_dir="$POST" \
      --log_file="$OUT/mlperf_log_accuracy.json" --num_proc=8) | tee "$OUT/accuracy.txt"
    touch "$WORK/.evaluate.ok"
  fi
}

run_one() {
  case "$1" in
    environment) run_environment 2>&1 | tee -a "$LOGS/build.log" ;;
    preprocess) run_preprocess 2>&1 | tee -a "$LOGS/preprocess.log" ;;
    inference) run_inference 2>&1 | tee -a "$LOGS/run.log" ;;
    evaluate) run_evaluate 2>&1 | tee -a "$LOGS/evaluator.log" ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}

if [[ -n "$STAGE" ]]; then
  run_one "$STAGE"
  exit 0
fi

start=0
if [[ -n "$FROM" ]]; then
  found=0
  for i in "${!stages[@]}"; do
    if [[ "${stages[$i]}" == "$FROM" ]]; then start=$i; found=1; fi
  done
  [[ $found -eq 1 ]] || { echo "unknown stage: $FROM" >&2; exit 2; }
  for ((i=start; i<${#stages[@]}; i++)); do
    rm -f "$WORK/.${stages[$i]}.ok"
  done
  case "$FROM" in
    environment)
      : > "$LOGS/build.log"
      rm -f "$LOGS/preprocess.log" "$LOGS/run.log" "$LOGS/evaluator.log" \
        "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
        "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt"
      ;;
    preprocess)
      : > "$LOGS/preprocess.log"
      rm -f "$LOGS/run.log" "$LOGS/evaluator.log" \
        "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
        "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt"
      ;;
    inference)
      : > "$LOGS/run.log"
      rm -f "$LOGS/evaluator.log" "$OUT/mlperf_log_accuracy.json" \
        "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt"
      ;;
    evaluate)
      : > "$LOGS/evaluator.log"
      rm -f "$OUT/accuracy.txt"
      ;;
  esac
fi

for ((i=start; i<${#stages[@]}; i++)); do
  run_one "${stages[$i]}"
done
