#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
KITS19_RAW_DIR="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR="$CACHE_ROOT/3d-unet-kits19/model"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

TARGET=""
ONE_STAGE=""
FROM_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) ONE_STAGE="${2:-}"; shift 2 ;;
    --from) FROM_STAGE="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "3d-unet" ]] || { echo "supported target: 3d-unet" >&2; exit 2; }
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "OUTPUT_DIR must be set" >&2; exit 2; }

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="$ROOT_DIR/repo"
OUT_DIR=$(realpath -m "$OUTPUT_DIR")
TARGET_DIR=$(dirname "$OUT_DIR")
WORK_DIR="$TARGET_DIR/work"
SRC_DIR="$WORK_DIR/src"
VENV_DIR="$WORK_DIR/.venv"
PREPROC_DIR="$WORK_DIR/preprocessed_data"
POSTPROC_DIR="$WORK_DIR/postprocessed_data"
RAW_LINK_DIR="$WORK_DIR/raw_data"

[[ -d "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" ]] || { echo "fresh repo source is missing" >&2; exit 1; }
[[ -f "$MODEL_DIR/3dunet_kits19_pytorch.ptc" ]] || { echo "fp32 model is missing" >&2; exit 1; }
[[ -d "$KITS19_RAW_DIR" ]] || { echo "KiTS19 raw input is missing" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$WORK_DIR"

stage_rank() {
  case "$1" in environment) echo 1;; preprocess) echo 2;; gpu) echo 3;; evaluate) echo 4;; *) return 1;; esac
}

if [[ -n "$ONE_STAGE" ]]; then stage_rank "$ONE_STAGE" >/dev/null || { echo "bad stage: $ONE_STAGE" >&2; exit 2; }; fi
if [[ -n "$FROM_STAGE" ]]; then
  FROM_RANK=$(stage_rank "$FROM_STAGE") || { echo "bad stage: $FROM_STAGE" >&2; exit 2; }
  case "$FROM_STAGE" in
    environment)
      rm -rf "$WORK_DIR"
      mkdir -p "$WORK_DIR" "$OUT_DIR"
      rm -f "$OUT_DIR"/mlperf_log_accuracy.json "$OUT_DIR"/mlperf_log_detail.txt "$OUT_DIR"/accuracy.txt "$OUT_DIR"/*.log
      ;;
    preprocess)
      rm -rf "$PREPROC_DIR" "$POSTPROC_DIR" "$RAW_LINK_DIR"
      rm -f "$WORK_DIR"/.preprocess.success "$WORK_DIR"/.gpu.success "$WORK_DIR"/.evaluate.success
      rm -f "$OUT_DIR"/mlperf_log_accuracy.json "$OUT_DIR"/mlperf_log_detail.txt "$OUT_DIR"/accuracy.txt "$OUT_DIR"/preprocess.log "$OUT_DIR"/run.log "$OUT_DIR"/evaluator.log
      ;;
    gpu)
      rm -rf "$POSTPROC_DIR"
      rm -f "$WORK_DIR"/.gpu.success "$WORK_DIR"/.evaluate.success
      rm -f "$OUT_DIR"/mlperf_log_accuracy.json "$OUT_DIR"/mlperf_log_detail.txt "$OUT_DIR"/accuracy.txt "$OUT_DIR"/run.log "$OUT_DIR"/evaluator.log
      ;;
    evaluate)
      rm -rf "$POSTPROC_DIR"
      rm -f "$WORK_DIR"/.evaluate.success "$OUT_DIR"/accuracy.txt "$OUT_DIR"/evaluator.log
      ;;
  esac
else
  FROM_RANK=1
fi

want_stage() {
  local name=$1 rank
  rank=$(stage_rank "$name")
  if [[ -n "$ONE_STAGE" ]]; then [[ "$ONE_STAGE" == "$name" ]]; else (( rank >= FROM_RANK )); fi
}

environment_stage() {
  if [[ -f "$WORK_DIR/.environment.success" ]]; then return; fi
  mkdir -p "$OUT_DIR" "$WORK_DIR"
  {
    date -Ins
    echo "Creating Python environment from shared cache"
    uv venv --python /usr/bin/python3.12 "$VENV_DIR"
    uv pip install --offline --python "$VENV_DIR/bin/python" \
      'torch==2.5.1' 'numpy==1.26.4' 'scipy==1.14.1' \
      'nibabel==5.3.2' 'pandas==2.2.3'
    loadgen_wheel=$(find "$UV_CACHE_DIR" -type f -name 'mlcommons_loadgen-6.0.17-cp312-cp312-linux_x86_64.whl' -print -quit)
    [[ -n "$loadgen_wheel" ]] || { echo "cached CPython 3.12 LoadGen wheel not found" >&2; exit 1; }
    uv pip install --offline --python "$VENV_DIR/bin/python" "$loadgen_wheel"
    "$VENV_DIR/bin/python" - <<'PY'
import torch, numpy, scipy, nibabel, pandas, mlperf_loadgen
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('numpy', numpy.__version__, 'scipy', scipy.__version__)
print('nibabel', nibabel.__version__, 'pandas', pandas.__version__)
print('cuda_available', torch.cuda.is_available())
PY
  } 2>&1 | tee "$OUT_DIR/environment.log"

  cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" "$SRC_DIR"
  cp "$REPO_DIR/loadgen/mlperf.conf" "$WORK_DIR/mlperf.conf"
  sed -i 's/signal\.gaussian(/signal.windows.gaussian(/' "$SRC_DIR/inference_utils.py"
  sed -i '/import pandas as pd/a\
if not hasattr(pd.DataFrame, "append"):\
    pd.DataFrame.append = lambda self, row, ignore_index=False: pd.concat([self, pd.DataFrame([row])], ignore_index=ignore_index)' "$SRC_DIR/accuracy_kits.py"
  sed -i '/self\.model\.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$SRC_DIR/pytorch_SUT.py"
  sed -i 's/    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)/    if not args.accuracy:\n        settings.FromConfig(args.user_conf, "3d-unet", args.scenario)/' "$SRC_DIR/run.py"

  if [[ -n "${PREFLIGHT_CASES:-}" ]]; then
    PREFLIGHT_CASES="$PREFLIGHT_CASES" SRC_DIR="$SRC_DIR" "$VENV_DIR/bin/python" - <<'PY'
import json, os
root=os.environ['SRC_DIR']
cases=[x for x in os.environ['PREFLIGHT_CASES'].split(',') if x]
with open(os.path.join(root,'meta','inference_cases.json'),'w') as f: json.dump(cases,f,indent=2)
p=os.path.join(root,'meta','checksum_inference.json')
old=json.load(open(p))
keep={f'{c}.pkl': old[f'{c}.pkl'] for c in cases}
with open(p,'w') as f: json.dump(keep,f,indent=2,sort_keys=True)
PY
  fi
  touch "$WORK_DIR/.environment.success"
}

preprocess_stage() {
  if [[ -f "$WORK_DIR/.preprocess.success" ]]; then return; fi
  mkdir -p "$RAW_LINK_DIR" "$PREPROC_DIR"
  local cases
  cases=$(SRC_DIR="$SRC_DIR" "$VENV_DIR/bin/python" - <<'PY'
import json,os
print(' '.join(json.load(open(os.path.join(os.environ['SRC_DIR'],'meta','inference_cases.json')))))
PY
)
  for case_id in $cases; do
    source_id=$case_id
    [[ "$case_id" == case_00400 ]] && source_id=case_00185
    ln -sfn "$KITS19_RAW_DIR/$source_id" "$RAW_LINK_DIR/$case_id"
  done
  (
    cd "$SRC_DIR"
    "$VENV_DIR/bin/python" preprocess.py --raw_data_dir "$RAW_LINK_DIR" \
      --results_dir "$PREPROC_DIR" --mode preprocess --num_proc 4
  ) 2>&1 | tee "$OUT_DIR/preprocess.log"
  touch "$WORK_DIR/.preprocess.success"
}

gpu_stage() {
  if [[ -f "$WORK_DIR/.gpu.success" ]]; then return; fi
  echo "GPU state immediately before run:" | tee "$OUT_DIR/run.log"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | tee -a "$OUT_DIR/run.log"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  (( free_mib >= 24000 )) || { echo "insufficient free GPU memory: ${free_mib} MiB" | tee -a "$OUT_DIR/run.log"; exit 1; }
  mkdir -p "$POSTPROC_DIR"
  (
    cd "$SRC_DIR"
    LOG_PATH="$OUT_DIR" SKIP_VERIFY_ACCURACY=1 "$VENV_DIR/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
      --preprocessed_data_dir="$PREPROC_DIR" \
      --mlperf_conf="$WORK_DIR/mlperf.conf" --user_conf="$SRC_DIR/user.conf"
  ) 2>&1 | tee -a "$OUT_DIR/run.log"
  [[ -s "$OUT_DIR/mlperf_log_accuracy.json" ]] || { echo "missing LoadGen accuracy log" >&2; exit 1; }
  [[ -s "$OUT_DIR/mlperf_log_detail.txt" ]] || { echo "missing LoadGen detail log" >&2; exit 1; }
  rg -q '^pytorch device: ' "$OUT_DIR/run.log" || { echo "required device log line missing" >&2; exit 1; }
  touch "$WORK_DIR/.gpu.success"
}

evaluate_stage() {
  if [[ -f "$WORK_DIR/.evaluate.success" ]]; then return; fi
  (
    cd "$SRC_DIR"
    "$VENV_DIR/bin/python" accuracy_kits.py \
      --log_file="$OUT_DIR/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="$PREPROC_DIR" \
      --postprocessed_data_dir="$POSTPROC_DIR" --num_proc 4
  ) 2>&1 | tee "$OUT_DIR/accuracy.txt" "$OUT_DIR/evaluator.log"
  rg -q '^Accuracy: mean = [0-9]+\.[0-9]{5}, kidney = [0-9]+\.[0-9]{4}, tumor = [0-9]+\.[0-9]{4}$' "$OUT_DIR/accuracy.txt" || { echo "formatted accuracy result missing" >&2; exit 1; }
  touch "$WORK_DIR/.evaluate.success"
}

if want_stage environment; then environment_stage; fi
if want_stage preprocess; then environment_stage; preprocess_stage; fi
if want_stage gpu; then environment_stage; preprocess_stage; gpu_stage; fi
if want_stage evaluate; then environment_stage; preprocess_stage; gpu_stage; evaluate_stage; fi
