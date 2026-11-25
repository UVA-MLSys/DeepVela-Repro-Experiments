#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=""
START_STAGE="environment"
ONLY_STAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET=${2:-}; shift 2 ;;
    --from) START_STAGE=${2:-}; shift 2 ;;
    --stage) ONLY_STAGE=${2:-}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$TARGET" == "3d-unet" ]] || { echo "supported target: 3d-unet" >&2; exit 2; }

CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
KITS19_RAW_DIR=${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}
MODEL_DIR=${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}
UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
UV_EXCLUDE_NEWER=2026-11-10
export UV_CACHE_DIR UV_EXCLUDE_NEWER

WORK_DIR="$ROOT_DIR/$TARGET"
OUTPUT_DIR=${OUTPUT_DIR:-$WORK_DIR/out}
APP_DIR="$WORK_DIR/app"
PYTHON_DIR="$WORK_DIR/python"
RAW_DIR="$WORK_DIR/raw_data"
PREPROCESSED_DIR="$WORK_DIR/preprocessed_data"
POSTPROCESSED_DIR="$WORK_DIR/postprocessed_data"
MARKER_DIR="$WORK_DIR/markers"
mkdir -p "$OUTPUT_DIR" "$MARKER_DIR"

stages=(environment preprocess run)
stage_index() {
  case "$1" in environment) echo 0;; preprocess) echo 1;; run) echo 2;; *) return 1;; esac
}
stage_index "$START_STAGE" >/dev/null || { echo "bad --from stage: $START_STAGE" >&2; exit 2; }
if [[ -n "$ONLY_STAGE" ]]; then
  stage_index "$ONLY_STAGE" >/dev/null || { echo "bad --stage: $ONLY_STAGE" >&2; exit 2; }
fi

clear_from() {
  local idx
  idx=$(stage_index "$1")
  if (( idx <= 0 )); then
    rm -rf "$APP_DIR" "$PYTHON_DIR" "$RAW_DIR" "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR"
    rm -f "$MARKER_DIR/environment.ok" "$MARKER_DIR/preprocess.ok" "$MARKER_DIR/run.ok"
  elif (( idx <= 1 )); then
    rm -rf "$RAW_DIR" "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR"
    rm -f "$MARKER_DIR/preprocess.ok" "$MARKER_DIR/run.ok"
  else
    rm -rf "$POSTPROCESSED_DIR"
    rm -f "$MARKER_DIR/run.ok"
  fi
  if (( idx <= 2 )); then
    rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt" \
      "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log"
  fi
}

environment_stage() {
  {
    echo "stage: environment"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    test -d "$ROOT_DIR/repo"
    test -s "$MODEL_DIR/3dunet_kits19_pytorch.ptc"
    uv venv --python 3.11 "$PYTHON_DIR"
    uv pip install --python "$PYTHON_DIR/bin/python" \
      'numpy==1.26.4' 'scipy==1.11.4' 'nibabel==5.2.1' 'pandas==1.5.3' \
      'torch==2.4.1' 'pybind11==2.13.6' 'setuptools==75.3.0' 'wheel==0.45.1'
    cp -a "$ROOT_DIR/repo/vision/medical_imaging/3d-unet-kits19" "$APP_DIR"
    cp "$ROOT_DIR/repo/loadgen/mlperf.conf" "$APP_DIR/mlperf.conf"
    cp -a "$ROOT_DIR/repo/loadgen" "$WORK_DIR/loadgen-build"
    uv pip install --python "$PYTHON_DIR/bin/python" --no-build-isolation "$WORK_DIR/loadgen-build"
    rm -rf "$WORK_DIR/loadgen-build"
    "$PYTHON_DIR/bin/python" -c 'import torch, mlperf_loadgen, numpy, scipy, nibabel, pandas; print("torch", torch.__version__, "cuda", torch.version.cuda)'
  } 2>&1 | tee "$OUTPUT_DIR/build.log"

  cat > "$APP_DIR/base_SUT.py" <<'PY'
import mlperf_loadgen as lg


class BaseSUT:
    def __init__(self, qsl):
        self.qsl = qsl
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        raise NotImplementedError

    def flush_queries(self):
        return None
PY

  cat > "$APP_DIR/pytorch_SUT.py" <<'PY'
import numpy as np
import torch
import mlperf_loadgen as lg

from base_SUT import BaseSUT
from inference_utils import prepare_arrays, get_slice_for_sliding_window, finalize
from global_vars import ROI_SHAPE
from kits_QSL import get_kits_QSL


class PytorchSUT(BaseSUT):
    def __init__(self, model_path, preprocessed_data_dir, perf_count):
        self.device = torch.device("cuda:0")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("pytorch device: {}".format(next(self.model.parameters()).device))
        super().__init__(get_kits_QSL(preprocessed_data_dir, perf_count))

    def infer(self, image):
        image = np.expand_dims(image, axis=0)
        result, norm_map, norm_patch = prepare_arrays(image)
        d, h, w = ROI_SHAPE
        with torch.inference_mode():
            for i, j, k in get_slice_for_sliding_window(image):
                patch = np.ascontiguousarray(image[:, :, i:i+d, j:j+h, k:k+w])
                tensor = torch.from_numpy(patch).to(self.device, non_blocking=True)
                output = self.model(tensor)
                output = output.detach().cpu().numpy()
                result[:, :, i:i+d, j:j+h, k:k+w] += output * norm_patch
                norm_map[:, :, i:i+d, j:j+h, k:k+w] += norm_patch
        return np.ascontiguousarray(finalize(result, norm_map), dtype=np.uint8)

    def issue_queries(self, query_samples):
        for query in query_samples:
            prediction = self.infer(self.qsl.get_features(query.index))
            response = lg.QuerySampleResponse(query.id, prediction.ctypes.data, prediction.nbytes)
            lg.QuerySamplesComplete([response])


def get_sut(model_path, preprocessed_data_dir, perf_count=None):
    return PytorchSUT(model_path, preprocessed_data_dir, perf_count)
PY

  if [[ -n "${REPRO_CASE_LIMIT:-}" ]]; then
    "$PYTHON_DIR/bin/python" - "$APP_DIR" "$REPRO_CASE_LIMIT" <<'PY'
import json, pathlib, sys
app, limit = pathlib.Path(sys.argv[1]), int(sys.argv[2])
cases = json.loads((app/'meta/inference_cases.json').read_text())[:limit]
(app/'meta/inference_cases.json').write_text(json.dumps(cases, indent=2) + '\n')
checks = json.loads((app/'meta/checksum_inference.json').read_text())
keep = {f'{case}.pkl' for case in cases}
(app/'meta/checksum_inference.json').write_text(json.dumps({k:v for k,v in checks.items() if k in keep}, indent=2) + '\n')
print('bounded case list:', cases)
PY
  fi
  touch "$MARKER_DIR/environment.ok"
}

preprocess_stage() {
  test -f "$MARKER_DIR/environment.ok"
  {
    echo "stage: preprocess"
    mkdir -p "$RAW_DIR" "$PREPROCESSED_DIR"
    "$PYTHON_DIR/bin/python" - "$APP_DIR/meta/inference_cases.json" "$KITS19_RAW_DIR" "$RAW_DIR" <<'PY'
import json, os, pathlib, sys
manifest, source, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
cases = json.loads(manifest.read_text())
for case in cases:
    actual = 'case_00185' if case == 'case_00400' else case
    origin = source / actual
    if not (origin/'imaging.nii.gz').is_file() or not (origin/'segmentation.nii.gz').is_file():
        raise FileNotFoundError(origin)
    target = dest / case
    if not target.exists():
        target.symlink_to(origin, target_is_directory=True)
print('input cases:', len(cases))
PY
    cd "$APP_DIR"
    "$PYTHON_DIR/bin/python" preprocess.py --raw_data_dir "$RAW_DIR" \
      --results_dir "$PREPROCESSED_DIR" --mode preprocess --num_proc 4
  } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  touch "$MARKER_DIR/preprocess.ok"
}

run_stage() {
  test -f "$MARKER_DIR/preprocess.ok"
  {
    echo "stage: run"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader,nounits
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
    [[ -n "$free_mib" && "$free_mib" -ge 12000 ]] || { echo "insufficient free GPU memory: ${free_mib:-unknown} MiB" >&2; exit 1; }
    mkdir -p "$POSTPROCESSED_DIR"
    cd "$APP_DIR"
    LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 \
      "$PYTHON_DIR/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
      --preprocessed_data_dir="$PREPROCESSED_DIR" \
      --mlperf_conf="$APP_DIR/mlperf.conf" --user_conf="$APP_DIR/user.conf"
  } 2>&1 | tee "$OUTPUT_DIR/run.log"

  {
    cd "$APP_DIR"
    "$PYTHON_DIR/bin/python" accuracy_kits.py \
      --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="$PREPROCESSED_DIR" \
      --postprocessed_data_dir="$POSTPROCESSED_DIR" --num_proc 4
  } 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  touch "$MARKER_DIR/run.ok"
}

run_one() {
  case "$1" in
    environment) environment_stage ;;
    preprocess) preprocess_stage ;;
    run) run_stage ;;
  esac
}

if [[ -n "$ONLY_STAGE" ]]; then
  if [[ ! -f "$MARKER_DIR/$ONLY_STAGE.ok" ]]; then run_one "$ONLY_STAGE"; fi
else
  clear_from "$START_STAGE"
  start=$(stage_index "$START_STAGE")
  for stage in "${stages[@]}"; do
    idx=$(stage_index "$stage")
    if (( idx >= start )); then run_one "$stage"; fi
  done
fi
