#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet" >&2
  exit 2
}

if [[ $# -ne 2 || "$1" != "--target" ]]; then
  usage
fi
target="$2"
if [[ "$target" != "3d-unet" ]]; then
  echo "unknown target: $target" >&2
  exit 2
fi

root_dir="$(pwd)"
repo_dir="$root_dir/repo"
work_dir="$root_dir/$target"
output_dir="${OUTPUT_DIR:-$work_dir/out}"
cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
raw_source="$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data"
model_source="$cache_root/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
uv_cache="$cache_root/uv"

mkdir -p "$work_dir" "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
exec > >(tee "$work_dir/reproduce.log") 2>&1

if [[ ! -f "$repo_dir/vision/medical_imaging/3d-unet-kits19/run.py" ]]; then
  echo "fresh repository not found at $repo_dir" >&2
  exit 1
fi
if [[ ! -d "$raw_source" || ! -f "$model_source" ]]; then
  echo "required cached inputs are missing" >&2
  exit 1
fi

export UV_CACHE_DIR="$uv_cache"
export UV_EXCLUDE_NEWER="2026-11-10"
venv="$work_dir/.venv"
bench="$work_dir/benchmark"

if [[ ! -x "$venv/bin/python" ]]; then
  {
    uv venv --python 3.11 --seed "$venv"
    uv pip install --python "$venv/bin/python" \
      'torch==2.5.1' 'numpy==1.26.4' 'scipy==1.12.0' \
      'nibabel==5.2.1' 'pandas==1.5.3' 'mlcommons-loadgen==6.0.17'
  } 2>&1 | tee "$work_dir/build.log"
else
  echo "Reusing $venv" | tee "$work_dir/build.log"
fi

if [[ ! -d "$bench" ]]; then
  cp -a "$repo_dir/vision/medical_imaging/3d-unet-kits19" "$bench"
fi
mkdir -p "$bench/build" "$bench/build/model" "$bench/build/raw_data" \
  "$bench/build/logs" "$bench/build/postprocessed_data"
cp "$repo_dir/loadgen/mlperf.conf" "$bench/build/mlperf.conf"
ln -sfn "$model_source" "$bench/build/model/3dunet_kits19_pytorch.ptc"

# Expose the cached input cases without modifying the read-only cache.  KiTS19
# case_00400 is the benchmark's documented duplicate of case_00185.
while IFS= read -r case_name; do
  if [[ "$case_name" == "case_00400" ]]; then
    ln -sfn "$raw_source/case_00185" "$bench/build/raw_data/$case_name"
  else
    ln -sfn "$raw_source/$case_name" "$bench/build/raw_data/$case_name"
  fi
done < <("$venv/bin/python" - <<'PY'
import json
for case in json.load(open('repo/vision/medical_imaging/3d-unet-kits19/meta/inference_cases.json')):
    print(case)
PY
)

# The source snapshot lacks the two PyTorch SUT adapter modules imported by
# run.py.  These implement the benchmark's sequential sliding-window adapter.
cat > "$bench/base_SUT.py" <<'PY'
import array
import mlperf_loadgen as lg

from kits_QSL import get_kits_QSL


class BaseSUT:
    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        self.qsl = get_kits_QSL(preprocessed_data_dir, performance_count)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        for query_sample in query_samples:
            prediction = self.process_sample(self.qsl.get_features(query_sample.index))
            response_array = array.array("B", prediction.tobytes())
            address, size = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_sample.id, address, size)
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass
PY

cat > "$bench/pytorch_SUT.py" <<'PY'
import numpy as np
import torch

from base_SUT import BaseSUT
from global_vars import ROI_SHAPE, SLIDE_OVERLAP_FACTOR
from inference_utils import finalize, get_slice_for_sliding_window, prepare_arrays


class PytorchSUT(BaseSUT):
    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this target")
        self.device = torch.device("cuda:0")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("pytorch device: {}".format(next(self.model.parameters()).device))
        super().__init__(model_path, preprocessed_data_dir, performance_count)

    def process_sample(self, image):
        # Preprocessing stores C,D,H,W; the model consumes N,C,D,H,W.
        image = np.expand_dims(image, axis=0)
        result, norm_map, norm_patch = prepare_arrays(image, ROI_SHAPE)
        with torch.inference_mode():
            for i, j, k in get_slice_for_sliding_window(
                    image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
                patch = image[:, :, i:i+ROI_SHAPE[0], j:j+ROI_SHAPE[1], k:k+ROI_SHAPE[2]]
                output = self.model(torch.from_numpy(patch).to(self.device))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output = output.detach().cpu().numpy()
                result[:, :, i:i+ROI_SHAPE[0], j:j+ROI_SHAPE[1], k:k+ROI_SHAPE[2]] += output * norm_patch
                norm_map[:, :, i:i+ROI_SHAPE[0], j:j+ROI_SHAPE[1], k:k+ROI_SHAPE[2]] += norm_patch
        return finalize(result, norm_map)


def get_sut(model_path, preprocessed_data_dir, performance_count=None):
    return PytorchSUT(model_path, preprocessed_data_dir, performance_count)
PY

cd "$bench"
if [[ ! -f build/preprocessed_data/preprocessed_files.pkl ]]; then
  "$venv/bin/python" preprocess.py \
    --raw_data_dir build/raw_data \
    --results_dir build/preprocessed_data \
    --mode preprocess --num_proc 4 2>&1 | tee "$work_dir/preprocess.log"
else
  echo "Reusing preprocessed data" | tee "$work_dir/preprocess.log"
fi

echo "GPU memory immediately before run:" | tee "$work_dir/gpu_memory.log"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits | tee -a "$work_dir/gpu_memory.log"

export LOG_PATH="$bench/build/logs"
export SKIP_VERIFY_ACCURACY=1
"$venv/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
  --model="$bench/build/model/3dunet_kits19_pytorch.ptc" \
  --preprocessed_data_dir="$bench/build/preprocessed_data" \
  --mlperf_conf="$bench/build/mlperf.conf" \
  --user_conf="$bench/user.conf" 2>&1 | tee "$work_dir/run.log"

cp "$bench/build/logs/mlperf_log_accuracy.json" "$output_dir/mlperf_log_accuracy.json"
cp "$bench/build/logs/mlperf_log_detail.txt" "$output_dir/mlperf_log_detail.txt"

"$venv/bin/python" accuracy_kits.py \
  --log_file="$output_dir/mlperf_log_accuracy.json" \
  --preprocessed_data_dir="$bench/build/preprocessed_data" \
  --postprocessed_data_dir="$bench/build/postprocessed_data" \
  --num_proc=4 \
  > >(tee "$work_dir/evaluator.log" "$output_dir/accuracy.txt") \
  2> >(tee "$work_dir/evaluator_stderr.log" >&2)

echo "Target $target completed; outputs are in $output_dir"
