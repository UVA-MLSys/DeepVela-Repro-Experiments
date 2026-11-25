#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START="env"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE="stage"; START="${2:-}"; shift 2 ;;
    --from) MODE="from"; START="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "3d-unet" ]] || { echo "supported target: 3d-unet" >&2; exit 2; }
case "$START" in env|data|run|eval) ;; *) echo "invalid stage: $START" >&2; exit 2 ;; esac

ROOT=$(cd "$(dirname "$0")" && pwd)
REPO="$ROOT/repo"
CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
KITS19_RAW_DIR=${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}
MODEL_DIR=${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}
UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

RUN_TAG=${REPRO_RUN_TAG:-full}
TARGET_ROOT="$ROOT/3d-unet"
WORK="$TARGET_ROOT/work-$RUN_TAG"
STATE="$WORK/state"
OUTPUT_DIR=${OUTPUT_DIR:-$TARGET_ROOT/out}
OUT=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)
VENV="$WORK/venv"
PY="$VENV/bin/python"
BENCH="$WORK/src/3d-unet-kits19"
RAW="$WORK/raw_data"
PRE="$WORK/preprocessed_data"
POST="$WORK/postprocessed_data"
MODEL="$MODEL_DIR/3dunet_kits19_pytorch.ptc"

require_inputs() {
  [[ -d "$REPO/loadgen" && -d "$REPO/vision/medical_imaging/3d-unet-kits19" ]] || { echo "fresh repository not found beside reproduce.sh" >&2; exit 1; }
  [[ -d "$KITS19_RAW_DIR" && -f "$MODEL" ]] || { echo "required cache inputs missing" >&2; exit 1; }
  [[ -d "$UV_CACHE_DIR" ]] || { echo "UV cache missing: $UV_CACHE_DIR" >&2; exit 1; }
}

reset_from() {
  local s=$1
  case "$s" in
    env)
      rm -rf "$WORK"
      mkdir -p "$STATE"
      rm -f "$OUT"/*.log "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt"
      ;;
    data)
      rm -rf "$RAW" "$PRE" "$POST"
      rm -f "$STATE/data.ok" "$STATE/run.ok" "$STATE/eval.ok"
      rm -f "$OUT/preprocess.log" "$OUT/run.log" "$OUT/evaluator.log" "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt"
      ;;
    run)
      rm -rf "$POST"
      rm -f "$STATE/run.ok" "$STATE/eval.ok"
      rm -f "$OUT/run.log" "$OUT/evaluator.log" "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt"
      ;;
    eval)
      rm -rf "$POST"
      rm -f "$STATE/eval.ok" "$OUT/evaluator.log" "$OUT/accuracy.txt"
      ;;
  esac
}

stage_env() {
  [[ -f "$STATE/env.ok" ]] && return
  mkdir -p "$WORK/src" "$STATE"
  cp -a "$REPO/loadgen" "$WORK/src/loadgen"
  cp -a "$REPO/vision/medical_imaging/3d-unet-kits19" "$BENCH"
  cp "$REPO/loadgen/mlperf.conf" "$BENCH/mlperf.conf"

  # Current pandas removed DataFrame.append; preserve the evaluator's row semantics.
  sed -i '/import pandas as pd/a\
if not hasattr(pd.DataFrame, "append"):\
    def _df_append(self, row, ignore_index=False):\
        return pd.concat([self, pd.DataFrame([row])], ignore_index=ignore_index)\
    pd.DataFrame.append = _df_append' "$BENCH/accuracy_kits.py"
  sed -i 's/float(final\["composite"\])/float(final["composite"].iloc[0])/; s/float(final\["kidney"\])/float(final["kidney"].iloc[0])/; s/float(final\["tumor"\])/float(final["tumor"].iloc[0])/' "$BENCH/accuracy_kits.py"
  # The target contract requires the actual parameter device in the run log.
  sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$BENCH/pytorch_SUT.py"

  if [[ -n "${PREFLIGHT_CASES:-}" ]]; then
    "$({ command -v python3.11 || command -v python3; })" - "$BENCH" "$PREFLIGHT_CASES" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cases = [x for x in sys.argv[2].split(',') if x]
all_cases = json.loads((root/'meta/inference_cases.json').read_text())
if not cases or any(x not in all_cases for x in cases):
    raise SystemExit('invalid PREFLIGHT_CASES')
(root/'meta/inference_cases.json').write_text(json.dumps(cases, indent=4) + '\n')
checks = json.loads((root/'meta/checksum_inference.json').read_text())
(root/'meta/checksum_inference.json').write_text(json.dumps({f'{x}.pkl': checks[f'{x}.pkl'] for x in cases}, indent=4) + '\n')
PY
    case_count=$(awk -F, '{print NF}' <<<"$PREFLIGHT_CASES")
    sed -i "s/^3d-unet\.Offline\.min_query_count = .*/3d-unet.Offline.min_query_count = $case_count/" "$BENCH/mlperf.conf"
  fi

  {
    date -Is
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv --python 3.11 --offline "$VENV"
    uv pip install --python "$PY" --offline \
      'setuptools==69.5.1' wheel 'pybind11==2.11.1' \
      'numpy==1.26.4' 'scipy==1.10.1' 'nibabel==5.3.2' 'pandas==3.0.6' 'torch==2.4.1'
    uv pip install --python "$PY" --offline --no-build-isolation "$WORK/src/loadgen"
    "$PY" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'nibabel', nibabel.__version__, 'pandas', pandas.__version__)
PY
  } 2>&1 | tee "$OUT/build.log"
  touch "$STATE/env.ok"
}

stage_data() {
  [[ -f "$STATE/data.ok" ]] && return
  [[ -f "$STATE/env.ok" ]] || { echo "env stage not complete" >&2; exit 1; }
  mkdir -p "$RAW" "$PRE"
  "$PY" - "$BENCH/meta/inference_cases.json" "$KITS19_RAW_DIR" "$RAW" <<'PY'
import json, os, pathlib, sys
cases = json.loads(pathlib.Path(sys.argv[1]).read_text())
source, dest = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
for case in cases:
    actual = 'case_00185' if case == 'case_00400' else case
    src = source/actual
    if not (src/'imaging.nii.gz').is_file() or not (src/'segmentation.nii.gz').is_file():
        raise SystemExit(f'missing raw case {actual}')
    os.symlink(src, dest/case, target_is_directory=True)
print(f'prepared {len(cases)} raw case entries')
PY
  (
    cd "$BENCH"
    /usr/bin/time -v "$PY" preprocess.py --raw_data_dir "$RAW" --results_dir "$PRE" --mode preprocess --num_proc "${PREPROCESS_PROCS:-8}"
  ) 2>&1 | tee "$OUT/preprocess.log"
  touch "$STATE/data.ok"
}

stage_run() {
  [[ -f "$STATE/run.ok" ]] && return
  [[ -f "$STATE/data.ok" ]] || { echo "data stage not complete" >&2; exit 1; }
  mkdir -p "$POST"
  {
    date -Is
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
  } | tee "$OUT/gpu-check.log"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ && "$free_mib" -ge "${MIN_GPU_FREE_MIB:-20000}" ]] || { echo "insufficient free GPU memory: ${free_mib:-unknown} MiB" | tee -a "$OUT/gpu-check.log"; exit 1; }
  (
    cd "$BENCH"
    LOG_PATH="$OUT" SKIP_VERIFY_ACCURACY=1 /usr/bin/time -v "$PY" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL" --preprocessed_data_dir="$PRE" \
      --mlperf_conf="$BENCH/mlperf.conf" --user_conf="$BENCH/user.conf"
  ) 2>&1 | tee "$OUT/run.log"
  [[ -s "$OUT/mlperf_log_accuracy.json" && -s "$OUT/mlperf_log_detail.txt" ]]
  grep -q '^pytorch device: ' "$OUT/run.log"
  touch "$STATE/run.ok"
}

stage_eval() {
  [[ -f "$STATE/eval.ok" ]] && return
  [[ -f "$STATE/run.ok" ]] || { echo "run stage not complete" >&2; exit 1; }
  (
    cd "$BENCH"
    /usr/bin/time -v "$PY" accuracy_kits.py --log_file "$OUT/mlperf_log_accuracy.json" \
      --preprocessed_data_dir "$PRE" --postprocessed_data_dir "$POST" --num_proc "${EVAL_PROCS:-8}"
  ) 2>&1 | tee "$OUT/evaluator.log" | tee "$OUT/accuracy.txt"
  grep -Eq '^Accuracy: mean = [0-9]+\.[0-9]{5}, kidney = [0-9]+\.[0-9]{4}, tumor = [0-9]+\.[0-9]{4}$' "$OUT/accuracy.txt"
  touch "$STATE/eval.ok"
}

run_stage() {
  case "$1" in env) stage_env;; data) stage_data;; run) stage_run;; eval) stage_eval;; esac
}

require_inputs
stages=(env data run eval)
if [[ "$MODE" == "all" ]]; then
  reset_from env
  for s in "${stages[@]}"; do run_stage "$s"; done
elif [[ "$MODE" == "stage" ]]; then
  run_stage "$START"
else
  reset_from "$START"
  active=0
  for s in "${stages[@]}"; do
    [[ "$s" == "$START" ]] && active=1
    [[ $active -eq 1 ]] && run_stage "$s"
  done
fi
