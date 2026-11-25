#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --stage) MODE="stage"; START_STAGE="$2"; shift 2 ;;
    --from) MODE="from"; START_STAGE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "target must be 3d-unet" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="$SCRIPT_DIR/repo"
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/$TARGET/out"}
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR" ;;
esac
WORK_DIR="$OUTPUT_DIR/work"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$OUTPUT_DIR"

stages=(build preprocess run evaluate)
stage_index() {
  local wanted=$1 i
  for i in "${!stages[@]}"; do
    [[ "${stages[$i]}" == "$wanted" ]] && { echo "$i"; return 0; }
  done
  return 1
}

if [[ "$MODE" == "from" ]]; then
  first=$(stage_index "$START_STAGE") || { echo "invalid stage: $START_STAGE" >&2; exit 2; }
  for ((i=first; i<${#stages[@]}; i++)); do
    rm -f "$OUTPUT_DIR/.${stages[$i]}.done"
  done
  if [[ "$START_STAGE" == "build" ]]; then
    rm -rf "$WORK_DIR"
    rm -f "$OUTPUT_DIR/build.log" "$OUTPUT_DIR/preprocess.log" "$OUTPUT_DIR/run.log" \
      "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt" \
      "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt"
  elif [[ "$START_STAGE" == "preprocess" ]]; then
    rm -rf "$WORK_DIR/preprocessed" "$WORK_DIR/postprocessed"
    rm -f "$OUTPUT_DIR/preprocess.log" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" \
      "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt"
  elif [[ "$START_STAGE" == "run" ]]; then
    rm -rf "$WORK_DIR/postprocessed"
    rm -f "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt" \
      "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt"
  else
    rm -rf "$WORK_DIR/postprocessed"
    rm -f "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  fi
fi

want_stage() {
  local name=$1
  if [[ "$MODE" == "stage" ]]; then
    [[ "$name" == "$START_STAGE" ]]
  elif [[ "$MODE" == "from" ]]; then
    local current start
    current=$(stage_index "$name")
    start=$(stage_index "$START_STAGE")
    (( current >= start ))
  else
    return 0
  fi
}

run_build() {
  [[ -f "$OUTPUT_DIR/.build.done" ]] && return 0
  {
    echo "building isolated Python environment"
    test -f "$REPO_DIR/vision/medical_imaging/3d-unet-kits19/run.py"
    test -f "$MODEL_INPUT"
    test -d "$RAW_INPUT"
    mkdir -p "$WORK_DIR"
    cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" "$WORK_DIR/benchmark"
    cp -a "$REPO_DIR/loadgen" "$WORK_DIR/loadgen"

    # Restore the fixed KiTS19 preprocessing spacing and add the requested
    # device diagnostic. The source snapshot lacks its checksum manifest, so
    # only the unavailable verification call is omitted.
    sed -i 's/^TARGET_SPACING = None$/TARGET_SPACING = [1.6, 1.2, 1.2]/' \
      "$WORK_DIR/benchmark/global_vars.py"
    sed -i '/self.device = torch.device(/,/"cuda:0" if torch.cuda.is_available() else "cpu")/ {
      /"cuda:0" if torch.cuda.is_available() else "cpu")/ a\        print("pytorch device: {}".format(self.device))
    }' "$WORK_DIR/benchmark/pytorch_SUT.py"
    sed -i 's/^    if args.mode == "verify":$/    if args.mode == "verify": pass/' \
      "$WORK_DIR/benchmark/preprocess.py"
    sed -i '/^        verify_dataset(args)$/d' "$WORK_DIR/benchmark/preprocess.py"

    if [[ -n "${PREFLIGHT_CASES:-}" ]]; then
      PREFLIGHT_CASES="$PREFLIGHT_CASES" BENCH_DIR="$WORK_DIR/benchmark" \
        python3 - <<'PY'
import json, os
path = os.path.join(os.environ["BENCH_DIR"], "meta", "inference_cases.json")
cases = [x for x in os.environ["PREFLIGHT_CASES"].split(",") if x]
with open(path, "w") as f:
    json.dump(cases, f, indent=2)
PY
    fi

    uv venv --python 3.11 --seed "$WORK_DIR/venv"
    uv pip install --python "$WORK_DIR/venv/bin/python" \
      'torch==2.7.1' 'numpy==1.24.4' 'scipy==1.10.1' \
      'nibabel==5.2.1' 'pandas==1.5.3' 'pybind11==2.13.6'
    uv pip install --python "$WORK_DIR/venv/bin/python" "$WORK_DIR/loadgen"
    "$WORK_DIR/venv/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("dependencies imported")
PY
  } 2>&1 | tee "$OUTPUT_DIR/build.log"
  touch "$OUTPUT_DIR/.build.done"
}

run_preprocess() {
  [[ -f "$OUTPUT_DIR/.preprocess.done" ]] && return 0
  {
    mkdir -p "$WORK_DIR/raw_data" "$WORK_DIR/preprocessed"
    BENCH_DIR="$WORK_DIR/benchmark" RAW_INPUT="$RAW_INPUT" RAW_LINKS="$WORK_DIR/raw_data" \
      "$WORK_DIR/venv/bin/python" - <<'PY'
import json, os
bench = os.environ["BENCH_DIR"]
with open(os.path.join(bench, "meta", "inference_cases.json")) as f:
    cases = json.load(f)
for case in cases:
    source = "case_00185" if case == "case_00400" else case
    src = os.path.join(os.environ["RAW_INPUT"], source)
    dst = os.path.join(os.environ["RAW_LINKS"], case)
    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    if not os.path.lexists(dst):
        os.symlink(src, dst)
print("resolved raw cases:", len(cases))
PY
    cd "$WORK_DIR/benchmark"
    "$WORK_DIR/venv/bin/python" preprocess.py \
      --raw_data_dir "$WORK_DIR/raw_data" \
      --results_dir "$WORK_DIR/preprocessed" --mode preprocess --num_proc 8
    BENCH_DIR="$WORK_DIR/benchmark" PREPROCESSED="$WORK_DIR/preprocessed" \
      "$WORK_DIR/venv/bin/python" - <<'PY'
import json, os
with open(os.path.join(os.environ["BENCH_DIR"], "meta", "inference_cases.json")) as f:
    cases = json.load(f)
for case in cases:
    for path in (os.path.join(os.environ["PREPROCESSED"], case + ".pkl"),
                 os.path.join(os.environ["PREPROCESSED"], "nifti", case, "segmentation.nii.gz")):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
print("validated preprocessed cases:", len(cases))
PY
  } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  touch "$OUTPUT_DIR/.preprocess.done"
}

run_inference() {
  [[ -f "$OUTPUT_DIR/.run.done" ]] && return 0
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader | tee "$OUTPUT_DIR/gpu-before-run.log"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if (( free_mib < 12000 )); then
    echo "insufficient free GPU memory: ${free_mib} MiB" | tee -a "$OUTPUT_DIR/run.log" >&2
    return 1
  fi
  {
    cd "$WORK_DIR/benchmark"
    mkdir -p "$WORK_DIR/postprocessed"
    cp "$REPO_DIR/loadgen/mlperf.conf" "$WORK_DIR/mlperf.conf"
    export LOG_PATH="$OUTPUT_DIR"
    export SKIP_VERIFY_ACCURACY=1
    "$WORK_DIR/venv/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_INPUT" --preprocessed_data_dir="$WORK_DIR/preprocessed" \
      --mlperf_conf="$WORK_DIR/mlperf.conf" --user_conf=user.conf
  } 2>&1 | tee "$OUTPUT_DIR/run.log"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  touch "$OUTPUT_DIR/.run.done"
}

run_evaluate() {
  [[ -f "$OUTPUT_DIR/.evaluate.done" ]] && return 0
  {
    cd "$WORK_DIR/benchmark"
    mkdir -p "$WORK_DIR/postprocessed"
    "$WORK_DIR/venv/bin/python" accuracy_kits.py \
      --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="$WORK_DIR/preprocessed" \
      --postprocessed_data_dir="$WORK_DIR/postprocessed" --num_proc=8
  } 2>&1 | tee "$OUTPUT_DIR/accuracy.txt"
  cp "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  touch "$OUTPUT_DIR/.evaluate.done"
}

if want_stage build; then run_build; fi
if want_stage preprocess; then run_preprocess; fi
if want_stage run; then run_inference; fi
if want_stage evaluate; then run_evaluate; fi

