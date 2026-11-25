#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET_ID=""
ONLY_STAGE=""
FROM_STAGE=""

while (($#)); do
  case "$1" in
    --target)
      TARGET_ID=${2:-}
      shift 2
      ;;
    --stage)
      ONLY_STAGE=${2:-}
      shift 2
      ;;
    --from)
      FROM_STAGE=${2:-}
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$TARGET_ID" != "3d-unet" ]]; then
  echo "Usage: $0 --target 3d-unet [--stage build|preprocess|run|evaluate] [--from build|preprocess|run|evaluate]" >&2
  exit 2
fi
if [[ -n "$ONLY_STAGE" && -n "$FROM_STAGE" ]]; then
  echo "--stage and --from are mutually exclusive" >&2
  exit 2
fi

for requested in "$ONLY_STAGE" "$FROM_STAGE"; do
  if [[ -n "$requested" && ! "$requested" =~ ^(build|preprocess|run|evaluate)$ ]]; then
    echo "Unknown stage: $requested" >&2
    exit 2
  fi
done

: "${CACHE_ROOT:=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
: "${KITS19_RAW_DIR:=$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}"
: "${MODEL_DIR:=$CACHE_ROOT/3d-unet-kits19/model}"
: "${UV_CACHE_DIR:=$CACHE_ROOT/uv}"
: "${OUTPUT_DIR:=$ROOT/$TARGET_ID/out}"
export UV_CACHE_DIR
export UV_EXCLUDE_NEWER=2026-11-10

REPO=$ROOT/repo
TARGET=$ROOT/$TARGET_ID
WORK=$TARGET/work
BENCH=$WORK/benchmark
VENV=$WORK/venv
MARKERS=$WORK/markers
mkdir -p "$OUTPUT_DIR" "$TARGET"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

for required in "$REPO/vision/medical_imaging/3d-unet-kits19/run.py" "$REPO/loadgen/mlperf.conf" "$KITS19_RAW_DIR" "$MODEL_DIR/3dunet_kits19_pytorch.ptc"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input is missing: $required" >&2
    exit 1
  fi
done

stage_index() {
  case "$1" in
    build) echo 0 ;;
    preprocess) echo 1 ;;
    run) echo 2 ;;
    evaluate) echo 3 ;;
  esac
}

if [[ -n "$FROM_STAGE" ]]; then
  from_index=$(stage_index "$FROM_STAGE")
  if ((from_index == 0)); then
    rm -rf "$WORK"
    rm -f "$TARGET/build.log" "$TARGET/preprocess.log" "$TARGET/run.log" "$TARGET/evaluator.log" "$TARGET/gpu-before-run.log"
    rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/accuracy.txt"
  else
    mkdir -p "$MARKERS"
    if ((from_index <= 1)); then
      rm -rf "$BENCH/build/preprocessed_data" "$BENCH/build/postprocessed_data" "$BENCH/build/logs"
      rm -f "$MARKERS/preprocess.done" "$MARKERS/run.done" "$MARKERS/evaluate.done"
      rm -f "$TARGET/preprocess.log" "$TARGET/run.log" "$TARGET/evaluator.log" "$TARGET/gpu-before-run.log"
      rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/accuracy.txt"
    elif ((from_index <= 2)); then
      rm -rf "$BENCH/build/postprocessed_data" "$BENCH/build/logs"
      rm -f "$MARKERS/run.done" "$MARKERS/evaluate.done"
      rm -f "$TARGET/run.log" "$TARGET/evaluator.log" "$TARGET/gpu-before-run.log"
      rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/accuracy.txt"
    else
      rm -rf "$BENCH/build/postprocessed_data"
      rm -f "$MARKERS/evaluate.done" "$TARGET/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
    fi
  fi
fi

should_run() {
  local stage=$1
  if [[ -n "$ONLY_STAGE" ]]; then
    [[ "$stage" == "$ONLY_STAGE" ]]
  elif [[ -n "$FROM_STAGE" ]]; then
    (( $(stage_index "$stage") >= $(stage_index "$FROM_STAGE") ))
  else
    return 0
  fi
}

run_build() {
  mkdir -p "$WORK" "$MARKERS"
  {
    date -Is
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv --clear --python /usr/bin/python3.11 "$VENV"
    uv pip install --offline --python "$VENV/bin/python" \
      'torch==2.15.0' 'numpy==1.26.4' 'scipy==1.12.0' \
      'nibabel==5.4.2' 'pandas==1.5.3' 'mlcommons-loadgen==6.0.17'
    rm -rf "$BENCH"
    mkdir -p "$BENCH"
    cp -a "$REPO/vision/medical_imaging/3d-unet-kits19/." "$BENCH/"
    mkdir -p "$BENCH/build/model"
    cp "$REPO/loadgen/mlperf.conf" "$BENCH/build/mlperf.conf"
    ln -s "$MODEL_DIR/3dunet_kits19_pytorch.ptc" "$BENCH/build/model/3dunet_kits19_pytorch.ptc"
    sed -i 's/settings.FromConfig(args.mlperf_conf, "3d-unet", args.scenario)/settings.FromConfig(args.mlperf_conf, "3d-unet", args.scenario, 0)/' "$BENCH/run.py"
    grep -q 'settings.FromConfig(args.mlperf_conf, "3d-unet", args.scenario, 0)' "$BENCH/run.py"
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "nibabel", nibabel.__version__, "pandas", pandas.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
  } 2>&1 | tee "$TARGET/build.log"

  cat > "$BENCH/pytorch_SUT.py" <<'PY'
import array

import mlperf_loadgen as lg
import numpy as np
import torch

from inference_utils import finalize, get_slice_for_sliding_window, prepare_arrays
from kits_QSL import get_kits_QSL


class PytorchSUT:
    def __init__(self, model_path, preprocessed_data_dir, performance_count=None):
        self.device = torch.device("cuda:0")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("pytorch device: {}".format(next(self.model.parameters()).device), flush=True)
        self.qsl = get_kits_QSL(preprocessed_data_dir, performance_count)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    @torch.inference_mode()
    def infer(self, image):
        image = image[None]
        result, norm_map, norm_patch = prepare_arrays(image)
        norm_patch = norm_patch[None, None]
        for i, j, k in get_slice_for_sliding_window(image):
            patch = image[..., i:i + 128, j:j + 128, k:k + 128]
            tensor = torch.from_numpy(patch).to(self.device)
            output = self.model(tensor).detach().cpu().numpy()
            result[..., i:i + 128, j:j + 128, k:k + 128] += output * norm_patch
            norm_map[..., i:i + 128, j:j + 128, k:k + 128] += norm_patch
        return finalize(result, norm_map)

    def issue_queries(self, query_samples):
        for sample in query_samples:
            prediction = np.ascontiguousarray(self.infer(self.qsl.get_features(sample.index)))
            payload = array.array("B", prediction.tobytes())
            address, length = payload.buffer_info()
            response = lg.QuerySampleResponse(sample.id, address, length * payload.itemsize)
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        return None


def get_sut(model_path, preprocessed_data_dir, performance_count=None):
    return PytorchSUT(model_path, preprocessed_data_dir, performance_count)
PY
  touch "$MARKERS/build.done"
}

run_preprocess() {
  [[ -f "$MARKERS/build.done" ]] || { echo "build stage has not succeeded" >&2; exit 1; }
  local raw_links=$WORK/raw_data
  rm -rf "$raw_links"
  mkdir -p "$raw_links"
  find "$KITS19_RAW_DIR" -mindepth 1 -maxdepth 1 -type d -name 'case_*' -print0 |
    while IFS= read -r -d '' case_dir; do
      ln -s "$case_dir" "$raw_links/$(basename "$case_dir")"
    done
  if [[ ! -e "$raw_links/case_00400" ]]; then
    ln -s "$KITS19_RAW_DIR/case_00185" "$raw_links/case_00400"
  fi
  mkdir -p "$BENCH/build/preprocessed_data"
  (
    cd "$BENCH"
    PATH="$VENV/bin:$PATH" python3 preprocess.py \
      --raw_data_dir "$raw_links" \
      --results_dir "$BENCH/build/preprocessed_data" \
      --mode preprocess --num_proc 8
  ) 2>&1 | tee "$TARGET/preprocess.log"
  [[ $(find "$BENCH/build/preprocessed_data" -maxdepth 1 -name 'case_*.pkl' | wc -l) -eq 43 ]]
  touch "$MARKERS/preprocess.done"
}

run_inference() {
  [[ -f "$MARKERS/preprocess.done" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  local free_mib
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader | tee "$TARGET/gpu-before-run.log"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if ((free_mib < 12000)); then
    echo "Insufficient free GPU memory: ${free_mib} MiB; need at least 12000 MiB" >&2
    exit 1
  fi
  mkdir -p "$BENCH/build/logs"
  (
    cd "$BENCH"
    export PATH="$VENV/bin:$PATH"
    export LOG_PATH="$BENCH/build/logs"
    export SKIP_VERIFY_ACCURACY=1
    python3 run.py --backend=pytorch --scenario=Offline --accuracy
  ) 2>&1 | tee "$TARGET/run.log"
  grep -q '^pytorch device: ' "$TARGET/run.log"
  cp "$BENCH/build/logs/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
  cp "$BENCH/build/logs/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  touch "$MARKERS/run.done"
}

run_evaluate() {
  [[ -f "$MARKERS/run.done" ]] || { echo "run stage has not succeeded" >&2; exit 1; }
  rm -rf "$BENCH/build/postprocessed_data"
  mkdir -p "$BENCH/build/postprocessed_data"
  (
    cd "$BENCH"
    PATH="$VENV/bin:$PATH" python3 accuracy_kits.py \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --preprocessed_data_dir "$BENCH/build/preprocessed_data" \
      --postprocessed_data_dir "$BENCH/build/postprocessed_data" \
      --num_proc 8
  ) 2>&1 | tee "$TARGET/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  grep -Eq '^Accuracy: mean = [0-9.]+, kidney = [0-9.]+, tumor = [0-9.]+$' "$OUTPUT_DIR/accuracy.txt"
  touch "$MARKERS/evaluate.done"
}

if should_run build && [[ ! -f "$MARKERS/build.done" ]]; then run_build; fi
if should_run preprocess && [[ ! -f "$MARKERS/preprocess.done" ]]; then run_preprocess; fi
if should_run run && [[ ! -f "$MARKERS/run.done" ]]; then run_inference; fi
if should_run evaluate && [[ ! -f "$MARKERS/evaluate.done" ]]; then run_evaluate; fi

if [[ -z "$ONLY_STAGE" && -f "$MARKERS/evaluate.done" ]]; then
  echo "Target 3d-unet complete; outputs are in $OUTPUT_DIR"
fi
