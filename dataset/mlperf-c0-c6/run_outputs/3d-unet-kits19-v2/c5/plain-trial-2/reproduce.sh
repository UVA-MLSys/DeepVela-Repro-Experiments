#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
TARGET=""
if [[ ${1:-} == "--target" && -n ${2:-} ]]; then
  TARGET=$2
fi
if [[ "$TARGET" != "3d-unet" ]]; then
  echo "usage: $0 --target 3d-unet" >&2
  exit 2
fi

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_DATA=$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data
MODEL=$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc
export UV_CACHE_DIR=$CACHE_ROOT/uv
export UV_EXCLUDE_NEWER=2026-11-10

WORK=$ROOT/$TARGET
OUT=${OUTPUT_DIR:-$WORK/out}
BENCH=$ROOT/repo/vision/medical_imaging/3d-unet-kits19
VENV=$WORK/.venv
RUNTIME=$WORK/runtime
PREPROCESSED=$WORK/preprocessed_data
POSTPROCESSED=$WORK/postprocessed_data
LOGDIR=$WORK/loadgen_logs
mkdir -p "$WORK" "$OUT" "$RUNTIME" "$POSTPROCESSED" "$LOGDIR"
OUT=$(cd "$OUT" && pwd)

if [[ ! -x "$VENV/bin/python" ]]; then
  {
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'numpy<2' 'scipy<1.13' 'pandas<2' nibabel torch==2.6.0 \
      pybind11==2.11.1 setuptools wheel
    uv pip install --python "$VENV/bin/python" "$ROOT/repo/loadgen"
  } >"$WORK/build.log" 2>&1
else
  echo "Reusing existing Python environment $VENV" >"$WORK/build.log"
fi
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$RUNTIME${PYTHONPATH:+:$PYTHONPATH}"

cat >"$RUNTIME/base_SUT.py" <<'PY'
import array
import numpy as np
import mlperf_loadgen as lg
from kits_QSL import get_kits_QSL


class BaseSUT:
    def __init__(self, preprocessed_data_dir, performance_count=None):
        self.qsl = get_kits_QSL(preprocessed_data_dir, performance_count)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        for query_sample in query_samples:
            image = self.qsl.get_features(query_sample.index)
            prediction = np.ascontiguousarray(self.infer_single_query(image), dtype=np.uint8)
            response_data = array.array("B", prediction.tobytes())
            address, length = response_data.buffer_info()
            response = lg.QuerySampleResponse(query_sample.id, address, length)
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass
PY

cat >"$RUNTIME/pytorch_SUT.py" <<'PY'
import numpy as np
import torch

from base_SUT import BaseSUT
from global_vars import ROI_SHAPE, SLIDE_OVERLAP_FACTOR
from inference_utils import finalize, get_slice_for_sliding_window, prepare_arrays


class PytorchSUT(BaseSUT):
    def __init__(self, model_path, preprocessed_data_dir, performance_count=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the PyTorch target")
        self.device = torch.device("cuda:0")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("pytorch device: {}".format(next(self.model.parameters()).device), flush=True)
        super().__init__(preprocessed_data_dir, performance_count)

    def infer_single_query(self, image):
        image = np.expand_dims(image, axis=0)
        result, norm_map, norm_patch = prepare_arrays(image)
        rd, rh, rw = ROI_SHAPE
        with torch.inference_mode():
            for d, h, w in get_slice_for_sliding_window(
                    image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
                patch = np.ascontiguousarray(image[..., d:d+rd, h:h+rh, w:w+rw])
                output = self.model(torch.from_numpy(patch).to(self.device))
                output = output.detach().cpu().numpy()
                weighted = output * norm_patch[None, None, ...]
                result[..., d:d+rd, h:h+rh, w:w+rw] += weighted
                norm_map[..., d:d+rd, h:h+rh, w:w+rw] += norm_patch
        return finalize(result, norm_map)


def get_sut(model_path, preprocessed_data_dir, performance_count=None):
    return PytorchSUT(model_path, preprocessed_data_dir, performance_count)
PY

RAW_VIEW=$WORK/raw_data
mkdir -p "$RAW_VIEW"
for case_path in "$RAW_DATA"/case_*; do
  case_name=${case_path##*/}
  if [[ "$case_name" != "case_00400" ]]; then
    ln -sfn "$case_path" "$RAW_VIEW/$case_name"
  fi
done
ln -sfn "$RAW_DATA/case_00185" "$RAW_VIEW/case_00400"

if [[ ! -s "$PREPROCESSED/preprocessed_files.pkl" ]]; then
  (
    cd "$BENCH"
    python3 preprocess.py --raw_data_dir "$RAW_VIEW" \
      --results_dir "$PREPROCESSED" --mode preprocess --num_proc 2
  ) >"$WORK/preprocess.log" 2>&1
else
  echo "Reusing completed preprocessing in $PREPROCESSED" >"$WORK/preprocess.log"
fi

cp "$ROOT/repo/loadgen/mlperf.conf" "$WORK/mlperf.conf"
: >"$WORK/audit.conf"

{
  echo "GPU state immediately before inference:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} >"$WORK/gpu-before.log"
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if (( FREE_MIB < 20000 )); then
  echo "Insufficient free GPU memory: ${FREE_MIB} MiB (need at least 20000 MiB)" \
    | tee -a "$WORK/gpu-before.log" >&2
  exit 1
fi

rm -f "$LOGDIR"/mlperf_log_*
(
  cd "$BENCH"
  export LOG_PATH="$LOGDIR"
  export SKIP_VERIFY_ACCURACY=1
  python3 run.py --backend=pytorch --scenario=Offline --accuracy \
    --model="$MODEL" --preprocessed_data_dir="$PREPROCESSED" \
    --mlperf_conf="$WORK/mlperf.conf" --user_conf="$BENCH/user.conf" \
    --audit_conf="$WORK/audit.conf"
) >"$WORK/run.log" 2>&1

cp "$LOGDIR/mlperf_log_accuracy.json" "$OUT/mlperf_log_accuracy.json"
cp "$LOGDIR/mlperf_log_detail.txt" "$OUT/mlperf_log_detail.txt"

(
  cd "$BENCH"
  python3 accuracy_kits.py --log_file="$OUT/mlperf_log_accuracy.json" \
    --preprocessed_data_dir="$PREPROCESSED" \
    --postprocessed_data_dir="$POSTPROCESSED" --num_proc 8
) >"$OUT/accuracy.txt" 2>"$WORK/evaluator.stderr.log"
cp "$OUT/accuracy.txt" "$WORK/evaluator.log"

grep -q '^pytorch device: ' "$WORK/run.log"
grep -q '^Accuracy: mean = ' "$OUT/accuracy.txt"
test -s "$OUT/mlperf_log_accuracy.json"
test -s "$OUT/mlperf_log_detail.txt"
