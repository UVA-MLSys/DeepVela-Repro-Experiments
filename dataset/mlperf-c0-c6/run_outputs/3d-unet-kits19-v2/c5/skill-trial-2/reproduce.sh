#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --from) FROM="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "3d-unet" ]] || { echo "--target must be 3d-unet" >&2; exit 2; }
[[ -z "$STAGE" || -z "$FROM" ]] || { echo "use only one of --stage and --from" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="$ROOT/repo"
TARGET_DIR="$ROOT/$TARGET"
OUT="${OUTPUT_DIR:-$TARGET_DIR/out}"
WORK="$TARGET_DIR/work"
COPY="$WORK/repo"
BENCH="$COPY/vision/medical_imaging/3d-unet-kits19"
VENV="$WORK/venv"
STAMPS="$WORK/stamps"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$OUT" "$WORK" "$STAMPS"
exec > >(tee -a "$OUT/driver.log") 2>&1

stages=(environment preprocess run evaluate)
valid_stage() {
  local wanted=$1 item
  for item in "${stages[@]}"; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}
if [[ -n "$STAGE" ]]; then valid_stage "$STAGE" || { echo "invalid stage: $STAGE"; exit 2; }; fi
if [[ -n "$FROM" ]]; then valid_stage "$FROM" || { echo "invalid stage: $FROM"; exit 2; }; fi

write_sut_bridge() {
  cat > "$BENCH/base_SUT.py" <<'PY'
import array
import mlperf_loadgen as lg

from kits_QSL import get_kits_QSL


class BaseSUT:
    def __init__(self, preprocessed_data_dir, performance_count):
        self.qsl = get_kits_QSL(preprocessed_data_dir, performance_count)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        for query_sample in query_samples:
            prediction, _ = self.infer_single_query(
                self.qsl.get_features(query_sample.index)
            )
            payload = array.array("B", prediction.tobytes())
            address, size = payload.buffer_info()
            response = lg.QuerySampleResponse(query_sample.id, address, size)
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        return
PY
  cat > "$BENCH/pytorch_SUT.py" <<'PY'
import numpy as np
import torch

from base_SUT import BaseSUT
from global_vars import ROI_SHAPE, SLIDE_OVERLAP_FACTOR
from inference_utils import (
    finalize,
    get_slice_for_sliding_window,
    prepare_arrays,
    runtime_measure,
)


class PytorchSUT(BaseSUT):
    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("pytorch device: {}".format(self.device), flush=True)
        super().__init__(preprocessed_data_dir, performance_count)

    @runtime_measure
    def infer_single_query(self, image):
        image = np.expand_dims(image, axis=0)
        result, norm_map, norm_patch = prepare_arrays(image)
        depth, height, width = ROI_SHAPE
        with torch.no_grad():
            tensor = torch.from_numpy(image).to(self.device)
            for i, j, k in get_slice_for_sliding_window(
                tensor, ROI_SHAPE, SLIDE_OVERLAP_FACTOR
            ):
                patch = tensor[:, :, i:i+depth, j:j+height, k:k+width]
                output = self.model(patch).detach().cpu().numpy()
                weighted = output * norm_patch
                result[:, :, i:i+depth, j:j+height, k:k+width] += weighted
                norm_map[:, :, i:i+depth, j:j+height, k:k+width] += norm_patch
        prediction = finalize(result, norm_map)
        return prediction, "PyTorch inference on volume {}".format(tuple(image.shape))


def get_sut(model_path, preprocessed_data_dir, performance_count=None):
    return PytorchSUT(model_path, preprocessed_data_dir, performance_count)
PY
}

reset_from() {
  local from=$1 seen=0 s
  for s in "${stages[@]}"; do
    [[ "$s" == "$from" ]] && seen=1
    if [[ $seen -eq 1 ]]; then rm -f "$STAMPS/$s.ok"; fi
  done
  case "$from" in
    environment)
      rm -rf "$COPY" "$VENV" "$WORK/raw_data" "$WORK/preprocessed" "$WORK/postprocessed"
      rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt"
      ;;
    preprocess)
      rm -rf "$WORK/preprocessed" "$WORK/postprocessed"
      rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt"
      ;;
    run)
      rm -rf "$WORK/postprocessed"
      rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" "$OUT/accuracy.txt"
      ;;
    evaluate)
      rm -rf "$WORK/postprocessed"
      rm -f "$OUT/accuracy.txt"
      ;;
  esac
}

stage_environment() {
  [[ -f "$REPO/vision/medical_imaging/3d-unet-kits19/run.py" ]] || { echo "fresh repository missing"; exit 1; }
  [[ -f "$MODEL_INPUT" ]] || { echo "model input missing: $MODEL_INPUT"; exit 1; }
  [[ -d "$RAW_INPUT" ]] || { echo "raw input missing: $RAW_INPUT"; exit 1; }
  mkdir -p "$COPY"
  cp -a "$REPO/." "$COPY/"
  write_sut_bridge
  uv venv --python /home/mi3se/.local/bin/python3.11 "$VENV"
  uv pip install --offline --python "$VENV/bin/python" \
    'numpy==1.26.4' 'scipy==1.10.1' 'pandas==1.5.3' 'nibabel==5.3.2' \
    'torch==2.4.1' "$COPY/loadgen"
  "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "nibabel", nibabel.__version__)
PY
  mkdir -p "$BENCH/build/model"
  ln -sfn "$MODEL_INPUT" "$BENCH/build/model/3dunet_kits19_pytorch.ptc"
  cp "$COPY/loadgen/mlperf.conf" "$BENCH/build/mlperf.conf"
  touch "$STAMPS/environment.ok"
}

stage_preprocess() {
  [[ -f "$STAMPS/environment.ok" ]] || { echo "environment stage has not succeeded"; exit 1; }
  mkdir -p "$WORK/raw_data"
  while IFS= read -r case_name; do
    if [[ "$case_name" == case_00400 ]]; then
      ln -sfn "$RAW_INPUT/case_00185" "$WORK/raw_data/case_00400"
    else
      ln -sfn "$RAW_INPUT/$case_name" "$WORK/raw_data/$case_name"
    fi
  done < <("$VENV/bin/python" -c 'import json; print(*json.load(open("'"$BENCH"'/meta/inference_cases.json")), sep="\n")')
  if [[ -n "${PREFLIGHT_CASE:-}" ]]; then
    "$VENV/bin/python" - "$BENCH" "$PREFLIGHT_CASE" <<'PY'
import json, pathlib, sys
bench, case = pathlib.Path(sys.argv[1]), sys.argv[2]
(bench / "meta/inference_cases.json").write_text(json.dumps([case], indent=2) + "\n")
checks = json.loads((bench / "meta/checksum_inference.json").read_text())
(bench / "meta/checksum_inference.json").write_text(json.dumps({case + ".pkl": checks[case + ".pkl"]}, indent=2) + "\n")
PY
  fi
  (cd "$BENCH" && "$VENV/bin/python" preprocess.py \
    --raw_data_dir "$WORK/raw_data" --results_dir "$WORK/preprocessed" \
    --mode preprocess --num_proc 4)
  touch "$STAMPS/preprocess.ok"
}

stage_run() {
  [[ -f "$STAMPS/preprocess.ok" ]] || { echo "preprocess stage has not succeeded"; exit 1; }
  echo "GPU memory immediately before run:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ && "$free_mib" -ge 12000 ]] || { echo "insufficient free GPU memory: ${free_mib:-unknown} MiB"; exit 1; }
  (cd "$BENCH" && LOG_PATH="$OUT" SKIP_VERIFY_ACCURACY=1 \
    "$VENV/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
    --model "$MODEL_INPUT" --preprocessed_data_dir "$WORK/preprocessed" \
    --mlperf_conf "$BENCH/build/mlperf.conf" --user_conf "$BENCH/user.conf")
  [[ -s "$OUT/mlperf_log_accuracy.json" && -s "$OUT/mlperf_log_detail.txt" ]]
  touch "$STAMPS/run.ok"
}

stage_evaluate() {
  [[ -f "$STAMPS/run.ok" ]] || { echo "run stage has not succeeded"; exit 1; }
  mkdir -p "$WORK/postprocessed"
  (cd "$BENCH" && "$VENV/bin/python" accuracy_kits.py \
    --log_file "$OUT/mlperf_log_accuracy.json" \
    --preprocessed_data_dir "$WORK/preprocessed" \
    --postprocessed_data_dir "$WORK/postprocessed" --num_proc 4) | tee "$OUT/accuracy.txt"
  rg -q '^Accuracy: mean = [0-9.]+, kidney = [0-9.]+, tumor = [0-9.]+' "$OUT/accuracy.txt"
  touch "$STAMPS/evaluate.ok"
}

if [[ -n "$FROM" ]]; then reset_from "$FROM"; fi
start=0
[[ -z "$STAGE" && -z "$FROM" ]] && start=1
for s in "${stages[@]}"; do
  [[ -n "$FROM" && "$s" == "$FROM" ]] && start=1
  if [[ -n "$STAGE" && "$s" != "$STAGE" ]]; then continue; fi
  [[ $start -eq 1 || "$s" == "$STAGE" ]] || continue
  if [[ -f "$STAMPS/$s.ok" ]]; then echo "stage $s already complete"; continue; fi
  echo "starting stage $s at $(date --iso-8601=seconds)"
  "stage_$s" 2>&1 | tee "$OUT/$s.log"
  echo "finished stage $s at $(date --iso-8601=seconds)"
done
