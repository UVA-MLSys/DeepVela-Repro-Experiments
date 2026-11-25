#!/usr/bin/env bash
set -euo pipefail

TARGET=""
ACTION="all"
START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) ACTION="stage"; START_STAGE="${2:-}"; shift 2 ;;
    --from) ACTION="from"; START_STAGE="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "supported target: 3d-unet" >&2
  exit 2
fi
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
: "${CACHE_ROOT:=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
: "${KITS19_RAW_DIR:=$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}"
: "${MODEL_DIR:=$CACHE_ROOT/3d-unet-kits19/model}"
: "${UV_CACHE_DIR:=$CACHE_ROOT/uv}"
export CACHE_ROOT KITS19_RAW_DIR MODEL_DIR UV_CACHE_DIR
export UV_EXCLUDE_NEWER=2026-11-10

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="$ROOT_DIR/repo"
TARGET_DIR="$ROOT_DIR/3d-unet"
RUN_LABEL="${REPRO_RUN_LABEL:-full}"
STATE_DIR="$TARGET_DIR/$RUN_LABEL"
WORK_REPO="$STATE_DIR/repo-copy"
BENCH_DIR="$WORK_REPO/vision/medical_imaging/3d-unet-kits19"
VENV="$STATE_DIR/.venv"
RAW_VIEW="$STATE_DIR/raw-data"
PREPROCESSED="$STATE_DIR/preprocessed-data"
POSTPROCESSED="$STATE_DIR/postprocessed-data"
LOG_DIR="$TARGET_DIR/logs"
MARKER_DIR="$STATE_DIR/markers"
MODEL_FILE="$MODEL_DIR/3dunet_kits19_pytorch.ptc"
STAGES=(env preprocess infer evaluate)

mkdir -p "$TARGET_DIR" "$LOG_DIR" "$MARKER_DIR"

stage_index() {
  local wanted="$1" i
  for i in "${!STAGES[@]}"; do
    [[ "${STAGES[$i]}" == "$wanted" ]] && { echo "$i"; return 0; }
  done
  echo "invalid stage: $wanted" >&2
  return 2
}

clear_from() {
  local start="$1" idx i stage
  idx="$(stage_index "$start")"
  for ((i=idx; i<${#STAGES[@]}; i++)); do
    stage="${STAGES[$i]}"
    rm -f "$MARKER_DIR/$stage.ok"
  done
  if (( idx <= 0 )); then
    rm -rf "$VENV" "$WORK_REPO"
  fi
  if (( idx <= 1 )); then
    rm -rf "$RAW_VIEW" "$PREPROCESSED"
  fi
  if (( idx <= 2 )); then
    rm -rf "$POSTPROCESSED" "$STATE_DIR/loadgen-logs"
  fi
  if (( idx <= 3 )); then
    rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/accuracy.txt"
  fi
}

write_backend_repair() {
  python3 - "$BENCH_DIR" <<'PY'
from pathlib import Path
import sys

d = Path(sys.argv[1])
(d / "base_SUT.py").write_text(r'''import array
import mlperf_loadgen as lg

from kits_QSL import get_kits_QSL


class BaseSUT:
    def __init__(self, preprocessed_data_dir, performance_count):
        self.qsl = get_kits_QSL(preprocessed_data_dir, performance_count)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        for query in query_samples:
            prediction = self.infer(query.index)
            response_array = array.array("B", prediction.tobytes())
            address, size = response_array.buffer_info()
            response = lg.QuerySampleResponse(query.id, address, size)
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass
''')
(d / "pytorch_SUT.py").write_text(r'''import numpy as np
import scipy.signal as signal
import torch

if not hasattr(signal, "gaussian"):
    signal.gaussian = signal.windows.gaussian

from base_SUT import BaseSUT
from inference_utils import finalize, get_slice_for_sliding_window, prepare_arrays
from global_vars import ROI_SHAPE


class PytorchSUT(BaseSUT):
    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("pytorch device: {}".format(next(self.model.parameters()).device), flush=True)
        super().__init__(preprocessed_data_dir, performance_count)

    def infer(self, sample_id):
        image = np.expand_dims(self.qsl.get_features(sample_id), axis=0)
        result, norm_map, norm_patch = prepare_arrays(image)
        with torch.no_grad():
            for i, j, k in get_slice_for_sliding_window(image):
                patch = image[:, :, i:i + ROI_SHAPE[0], j:j + ROI_SHAPE[1], k:k + ROI_SHAPE[2]]
                output = self.model(torch.from_numpy(patch).to(self.device)).detach().cpu().numpy()
                result[:, :, i:i + ROI_SHAPE[0], j:j + ROI_SHAPE[1], k:k + ROI_SHAPE[2]] += output * norm_patch
                norm_map[:, :, i:i + ROI_SHAPE[0], j:j + ROI_SHAPE[1], k:k + ROI_SHAPE[2]] += norm_patch
        return finalize(result, norm_map)


def get_sut(model_path, preprocessed_data_dir, performance_count=None):
    return PytorchSUT(model_path, preprocessed_data_dir, performance_count)
''')
accuracy = d / "accuracy_kits.py"
text = accuracy.read_text()
text = text.replace(
    "df = pd.DataFrame()",
    'df = pd.DataFrame(columns=["case", "kidney", "tumor", "composite"])',
)
text = text.replace(
    '''        df = df.append(
            {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite},
            ignore_index=True,
        )''',
    '''        df.loc[len(df)] = {
            "case": case, "kidney": kidney, "tumor": tumor, "composite": composite
        }''',
)
text = text.replace('float(final["composite"])', 'float(final["composite"].iloc[0])')
text = text.replace('float(final["kidney"])', 'float(final["kidney"].iloc[0])')
text = text.replace('float(final["tumor"])', 'float(final["tumor"].iloc[0])')
accuracy.write_text(text)
PY
}

stage_env() {
  local log="$LOG_DIR/$RUN_LABEL.build.log"
  {
    [[ -d "$SOURCE_REPO" ]] || { echo "missing repository: $SOURCE_REPO"; exit 1; }
    [[ -f "$MODEL_FILE" ]] || { echo "missing model: $MODEL_FILE"; exit 1; }
    cp -a "$SOURCE_REPO" "$WORK_REPO"
    write_backend_repair
    uv venv --python /usr/bin/python3.12 "$VENV"
    uv pip install --python "$VENV/bin/python" --offline \
      'torch==2.15.0' 'numpy==1.26.4' 'scipy==1.17.1' 'nibabel==5.4.2' \
      'pandas==3.0.6' 'pybind11==2.11.1' 'setuptools==84.2.1' 'wheel==0.48.0'
    python_include="$(find "$UV_CACHE_DIR/archive-v0" -path '*/python_3_12_x86_64-unknown-linux-gnu/include/python3.12/Python.h' -print -quit)"
    [[ -n "$python_include" ]] || { echo "cached CPython 3.12 headers not found"; exit 1; }
    CPLUS_INCLUDE_PATH="$(dirname "$python_include")${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" \
      uv pip install --python "$VENV/bin/python" --offline --no-build-isolation "$WORK_REPO/loadgen"
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
  } >"$log" 2>&1
  touch "$MARKER_DIR/env.ok"
}

stage_preprocess() {
  local log="$LOG_DIR/$RUN_LABEL.preprocess.log" case_name
  mkdir -p "$RAW_VIEW" "$PREPROCESSED"
  if [[ -n "${REPRO_PREFLIGHT_CASES:-}" ]]; then
    "$VENV/bin/python" - "$BENCH_DIR" "$REPRO_PREFLIGHT_CASES" <<'PY'
from pathlib import Path
import json, sys
d = Path(sys.argv[1]) / "meta"
cases = [x for x in sys.argv[2].split(",") if x]
all_sums = json.loads((d / "checksum_inference.json").read_text())
(d / "inference_cases.json").write_text(json.dumps(cases, indent=4) + "\n")
(d / "checksum_inference.json").write_text(json.dumps({f"{c}.pkl": all_sums[f"{c}.pkl"] for c in cases}, indent=4) + "\n")
PY
  fi
  "$VENV/bin/python" - "$BENCH_DIR/meta/inference_cases.json" <<'PY' | while read -r case_name; do
import json, sys
for case in json.load(open(sys.argv[1])):
    print(case)
PY
    if [[ "$case_name" == "case_00400" ]]; then
      cp -a "$KITS19_RAW_DIR/case_00185" "$RAW_VIEW/case_00400"
    else
      ln -s "$KITS19_RAW_DIR/$case_name" "$RAW_VIEW/$case_name"
    fi
  done
  (
    cd "$BENCH_DIR"
    "$VENV/bin/python" preprocess.py --raw_data_dir "$RAW_VIEW" \
      --results_dir "$PREPROCESSED" --mode preprocess --num_proc "${PREPROCESS_PROCS:-4}"
  ) >"$log" 2>&1
  touch "$MARKER_DIR/preprocess.ok"
}

stage_infer() {
  local log="$LOG_DIR/$RUN_LABEL.run.log"
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  echo "GPU free memory before run: ${free_mib} MiB" >"$LOG_DIR/$RUN_LABEL.gpu-memory.log"
  if (( free_mib < 12000 )); then
    echo "insufficient free GPU memory: ${free_mib} MiB" >>"$LOG_DIR/$RUN_LABEL.gpu-memory.log"
    return 1
  fi
  mkdir -p "$STATE_DIR/loadgen-logs" "$POSTPROCESSED"
  cp "$WORK_REPO/mlperf.conf" "$STATE_DIR/mlperf.conf"
  if [[ -n "${REPRO_PREFLIGHT_CASES:-}" ]]; then
    local count
    count="$(awk -F, '{print NF}' <<<"$REPRO_PREFLIGHT_CASES")"
    sed -i "s/3d-unet.Offline.min_query_count = 43/3d-unet.Offline.min_query_count = $count/" "$STATE_DIR/mlperf.conf"
  fi
  (
    cd "$BENCH_DIR"
    LOG_PATH="$STATE_DIR/loadgen-logs" SKIP_VERIFY_ACCURACY=1 \
      "$VENV/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_FILE" --preprocessed_data_dir="$PREPROCESSED" \
      --mlperf_conf="$STATE_DIR/mlperf.conf" --user_conf=user.conf
  ) >"$log" 2>&1
  touch "$MARKER_DIR/infer.ok"
}

stage_evaluate() {
  local log="$LOG_DIR/$RUN_LABEL.evaluator.log"
  mkdir -p "$OUTPUT_DIR"
  (
    cd "$BENCH_DIR"
    "$VENV/bin/python" accuracy_kits.py \
      --log_file="$STATE_DIR/loadgen-logs/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="$PREPROCESSED" \
      --postprocessed_data_dir="$POSTPROCESSED" \
      --num_proc "${EVALUATOR_PROCS:-4}"
  ) >"$OUTPUT_DIR/accuracy.txt" 2>"$log"
  cat "$OUTPUT_DIR/accuracy.txt" >>"$log"
  cp "$STATE_DIR/loadgen-logs/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
  cp "$STATE_DIR/loadgen-logs/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"
  "$VENV/bin/python" - "$OUTPUT_DIR/mlperf_log_accuracy.json" "$BENCH_DIR/meta/inference_cases.json" <<'PY'
import json, sys
preds = json.load(open(sys.argv[1]))
cases = json.load(open(sys.argv[2]))
indices = [x["qsl_idx"] for x in preds]
assert len(preds) == len(cases), (len(preds), len(cases))
assert sorted(indices) == list(range(len(cases))), indices
PY
  grep -q '^pytorch device: ' "$LOG_DIR/$RUN_LABEL.run.log"
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  touch "$MARKER_DIR/evaluate.ok"
}

run_stage() {
  local stage="$1"
  if [[ -f "$MARKER_DIR/$stage.ok" ]]; then
    echo "stage $stage already complete"
    return
  fi
  case "$stage" in
    env) stage_env ;;
    preprocess) stage_preprocess ;;
    infer) stage_infer ;;
    evaluate) stage_evaluate ;;
    *) echo "invalid stage: $stage" >&2; exit 2 ;;
  esac
}

if [[ "$ACTION" == "from" ]]; then
  clear_from "$START_STAGE"
fi

if [[ "$ACTION" == "stage" ]]; then
  run_stage "$START_STAGE"
else
  begin=0
  [[ "$ACTION" == "from" ]] && begin="$(stage_index "$START_STAGE")"
  for ((i=begin; i<${#STAGES[@]}; i++)); do
    run_stage "${STAGES[$i]}"
  done
fi
