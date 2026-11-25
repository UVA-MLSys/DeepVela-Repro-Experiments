#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet" >&2
  exit 2
}

[[ $# -eq 2 && "$1" == "--target" ]] || usage
TARGET=$2
[[ "$TARGET" == "3d-unet" ]] || usage
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

ROOT=$(cd "$(dirname "$0")" && pwd)
REPO="$ROOT/repo"
TARGET_DIR="$ROOT/$TARGET"
OUT=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)
WORK="$TARGET_DIR/work"
VENV="$TARGET_DIR/venv"
APP="$WORK/app"
LOADGEN="$WORK/loadgen"

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

rm -rf "$WORK" "$VENV"
mkdir -p "$WORK" "$APP" "$OUT"
cp -a "$REPO/vision/medical_imaging/3d-unet-kits19/." "$APP/"
cp -a "$REPO/loadgen/." "$LOADGEN/"

(
  set -x
  uv venv --python /home/mi3se/.local/bin/python3.11 "$VENV"
  uv pip install --python "$VENV/bin/python" \
    'torch==2.1.2' 'numpy==1.26.4' 'scipy==1.10.1' \
    'pandas==1.5.3' 'nibabel==5.2.1' 'pybind11==2.13.6' \
    'setuptools<81' wheel
  uv pip install --python "$VENV/bin/python" "$LOADGEN"
) >"$OUT/build.log" 2>&1

cat >"$APP/pytorch_SUT.py" <<'PY'
import array
import numpy as np
import torch
import mlperf_loadgen as lg

from inference_utils import prepare_arrays, get_slice_for_sliding_window, finalize
from global_vars import ROI_SHAPE
from kits_QSL import get_kits_QSL


class PytorchSUT:
    def __init__(self, model_path, preprocessed_data_dir, perf_count):
        self.device = torch.device("cuda:0")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("pytorch device: {}".format(next(self.model.parameters()).device), flush=True)
        self.qsl = get_kits_QSL(preprocessed_data_dir, perf_count)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def infer(self, image):
        image = np.expand_dims(image, axis=0)
        result, norm_map, norm_patch = prepare_arrays(image)
        rd, rh, rw = ROI_SHAPE
        with torch.no_grad():
            for i, j, k in get_slice_for_sliding_window(image):
                patch = image[..., i:i + rd, j:j + rh, k:k + rw]
                patch = torch.from_numpy(np.ascontiguousarray(patch)).to(self.device)
                output = self.model(patch)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output = output.detach().cpu().numpy()
                result[..., i:i + rd, j:j + rh, k:k + rw] += output * norm_patch
                norm_map[..., i:i + rd, j:j + rh, k:k + rw] += norm_patch
        return np.ascontiguousarray(finalize(result, norm_map), dtype=np.uint8)

    def issue_queries(self, query_samples):
        for sample in query_samples:
            prediction = self.infer(self.qsl.get_features(sample.index))
            payload = array.array("B", prediction.tobytes())
            address, length = payload.buffer_info()
            response = lg.QuerySampleResponse(sample.id, address, length)
            lg.QuerySamplesComplete([response])

    @staticmethod
    def flush_queries():
        return


def get_sut(model_path, preprocessed_data_dir, perf_count=None):
    return PytorchSUT(model_path, preprocessed_data_dir, perf_count)
PY

mkdir -p "$APP/build/raw_data" "$APP/build/model" "$APP/build/preprocessed_data" "$APP/build/postprocessed_data"
while IFS= read -r case_name; do
  [[ "$case_name" == "case_00400" ]] && source_name=case_00185 || source_name=$case_name
  ln -s "$RAW_INPUT/$source_name" "$APP/build/raw_data/$case_name"
done < <("$VENV/bin/python" -c 'import json; print(*json.load(open("'"$APP"'/meta/inference_cases.json")), sep="\n")')
ln -s "$MODEL_INPUT" "$APP/build/model/3dunet_kits19_pytorch.ptc"
cp "$LOADGEN/mlperf.conf" "$APP/build/mlperf.conf"

(
  set -x
  cd "$APP"
  "$VENV/bin/python" preprocess.py \
    --raw_data_dir="$APP/build/raw_data" \
    --results_dir="$APP/build/preprocessed_data" \
    --mode=preprocess --num_proc=4
) >"$OUT/preprocess.log" 2>&1

{
  date -Is
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader
} >"$OUT/gpu-state.log" 2>&1

(
  set -x
  cd "$APP"
  export LOG_PATH="$OUT"
  export SKIP_VERIFY_ACCURACY=1
  "$VENV/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
    --model="$MODEL_INPUT" \
    --preprocessed_data_dir="$APP/build/preprocessed_data" \
    --mlperf_conf="$APP/build/mlperf.conf" \
    --user_conf="$APP/user.conf"
) 2>&1 | tee "$OUT/run.log"

(
  cd "$APP"
  "$VENV/bin/python" accuracy_kits.py \
    --log_file="$OUT/mlperf_log_accuracy.json" \
    --preprocessed_data_dir="$APP/build/preprocessed_data" \
    --postprocessed_data_dir="$APP/build/postprocessed_data" \
    --num_proc=4
) >"$OUT/accuracy.txt" 2>"$OUT/evaluator-stderr.log"
cp "$OUT/accuracy.txt" "$OUT/evaluator.log"
cat "$OUT/accuracy.txt"

test -s "$OUT/mlperf_log_accuracy.json"
test -s "$OUT/mlperf_log_detail.txt"
test -s "$OUT/accuracy.txt"
