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
[[ "$TARGET" == "3d-unet" ]] || { echo "target must be 3d-unet" >&2; exit 2; }
[[ -z "$STAGE" || -z "$FROM" ]] || { echo "use only one of --stage or --from" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/repo"
OUT="${OUTPUT_DIR:-$ROOT/$TARGET/out}"
WORK="$ROOT/$TARGET/work"
CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
KITS19_RAW_DIR="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR="$CACHE_ROOT/3d-unet-kits19/model"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER="2026-11-10"
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"

[[ -f "$REPO/vision/medical_imaging/3d-unet-kits19/run.py" ]] || { echo "fresh repository not found at $REPO" >&2; exit 1; }
[[ -d "$KITS19_RAW_DIR" && -f "$MODEL_DIR/3dunet_kits19_pytorch.ptc" ]] || { echo "fixed benchmark inputs are missing" >&2; exit 1; }
mkdir -p "$OUT" "$ROOT/$TARGET"

stage_index() {
  case "$1" in build) echo 0;; preprocess) echo 1;; run) echo 2;; evaluate) echo 3;; *) return 1;; esac
}

clean_from() {
  local start="$1" idx
  idx="$(stage_index "$start")" || { echo "invalid stage: $start" >&2; exit 2; }
  [[ "$WORK" == "$ROOT/$TARGET/work" ]] || { echo "refusing unsafe work path" >&2; exit 1; }
  if (( idx == 0 )); then
    rm -rf -- "$WORK"
    rm -f -- "$OUT"/*.log "$OUT/accuracy.txt" "$OUT"/mlperf_log_*.json "$OUT"/mlperf_log_*.txt
  elif (( idx == 1 )); then
    rm -rf -- "$WORK/benchmark/build/preprocessed_data" "$WORK/benchmark/build/logs" "$WORK/benchmark/build/postprocessed_data"
    rm -f -- "$WORK"/.preprocess.done "$WORK"/.run.done "$WORK"/.evaluate.done "$OUT"/preprocess.log "$OUT"/gpu.log "$OUT"/evaluator.log "$OUT"/accuracy.txt "$OUT"/mlperf_log_*.json "$OUT"/mlperf_log_*.txt
  elif (( idx == 2 )); then
    rm -rf -- "$WORK/benchmark/build/logs" "$WORK/benchmark/build/postprocessed_data"
    rm -f -- "$WORK"/.run.done "$WORK"/.evaluate.done "$OUT"/gpu.log "$OUT"/evaluator.log "$OUT"/accuracy.txt "$OUT"/mlperf_log_*.json "$OUT"/mlperf_log_*.txt
  else
    rm -rf -- "$WORK/benchmark/build/postprocessed_data"
    rm -f -- "$WORK"/.evaluate.done "$OUT"/evaluator.log "$OUT"/accuracy.txt
  fi
}

run_build() {
  [[ -f "$WORK/.build.done" ]] && return
  mkdir -p "$WORK"
  {
    echo "stage: build"
    echo "source revision: 3fbc329939999c13d0a7b5e67fb2092287e06047"
    cp -a "$REPO" "$WORK/source"
    cp -a "$WORK/source/vision/medical_imaging/3d-unet-kits19" "$WORK/benchmark"
    mkdir -p "$WORK/benchmark/build/model"
    ln -s "$MODEL_DIR/3dunet_kits19_pytorch.ptc" "$WORK/benchmark/build/model/3dunet_kits19_pytorch.ptc"
    cp "$WORK/source/mlperf.conf" "$WORK/benchmark/build/mlperf.conf"

    cat > "$WORK/benchmark/base_SUT.py" <<'PY'
import array
import mlperf_loadgen as lg
from kits_QSL import get_kits_QSL

class BaseSUT:
    def __init__(self, preprocessed_data_dir, performance_count=None):
        self.qsl = get_kits_QSL(preprocessed_data_dir, performance_count)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        for query_sample in query_samples:
            result = self.do_infer(self.qsl.get_features(query_sample.index))
            payload = array.array("B", result.tobytes())
            addr, size = payload.buffer_info()
            response = lg.QuerySampleResponse(query_sample.id, addr, size * payload.itemsize)
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass
PY
    cat > "$WORK/benchmark/pytorch_SUT.py" <<'PY'
import numpy as np
import torch
from base_SUT import BaseSUT
from global_vars import ROI_SHAPE, SLIDE_OVERLAP_FACTOR
from inference_utils import prepare_arrays, get_slice_for_sliding_window, finalize

class PytorchSUT(BaseSUT):
    def __init__(self, model_path, preprocessed_data_dir, performance_count=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("pytorch device: {}".format(self.device), flush=True)
        super().__init__(preprocessed_data_dir, performance_count)

    def do_infer(self, image):
        image = np.expand_dims(image, axis=0)
        result, norm_map, norm_patch = prepare_arrays(image)
        norm_patch = norm_patch[np.newaxis, np.newaxis, ...]
        tensor = torch.from_numpy(image).to(self.device)
        with torch.inference_mode():
            for i, j, k in get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
                patch = tensor[..., i:i+ROI_SHAPE[0], j:j+ROI_SHAPE[1], k:k+ROI_SHAPE[2]]
                output = self.model(patch)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output = output.detach().cpu().numpy()
                result[..., i:i+ROI_SHAPE[0], j:j+ROI_SHAPE[1], k:k+ROI_SHAPE[2]] += output * norm_patch
                norm_map[..., i:i+ROI_SHAPE[0], j:j+ROI_SHAPE[1], k:k+ROI_SHAPE[2]] += norm_patch
        return finalize(result, norm_map)

def get_sut(model_path, preprocessed_data_dir, performance_count=None):
    return PytorchSUT(model_path, preprocessed_data_dir, performance_count)
PY
    uv python install 3.8
    uv venv --python 3.8 "$WORK/venv"
    uv pip install --python "$WORK/venv/bin/python" 'numpy==1.23.5' 'scipy==1.10.1' 'nibabel==3.2.1' 'pandas==1.5.3' 'torch==2.4.1'
    uv pip install --python "$WORK/venv/bin/python" "$WORK/source/loadgen"
    "$WORK/venv/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "nibabel", nibabel.__version__, "pandas", pandas.__version__)
PY
  } 2>&1 | tee "$OUT/build.log"
  touch "$WORK/.build.done"
}

run_preprocess() {
  [[ -f "$WORK/.preprocess.done" ]] && return
  local b="$WORK/benchmark"
  mkdir -p "$b/build/raw_data" "$b/build/preprocessed_data"
  for case_dir in "$KITS19_RAW_DIR"/case_*; do ln -s "$case_dir" "$b/build/raw_data/$(basename "$case_dir")"; done
  [[ -e "$b/build/raw_data/case_00400" ]] || ln -s "$KITS19_RAW_DIR/case_00185" "$b/build/raw_data/case_00400"
  if [[ -n "${REPRO_PREFLIGHT_CASE:-}" ]]; then
    "$WORK/venv/bin/python" - "$b" "$REPRO_PREFLIGHT_CASE" <<'PY'
import json, pathlib, sys
b, case = pathlib.Path(sys.argv[1]), sys.argv[2]
meta = b / "meta"
cases = json.loads((meta / "inference_cases.json").read_text())
if case not in cases: raise SystemExit("preflight case is not in inference manifest")
checks = json.loads((meta / "checksum_inference.json").read_text())
(meta / "inference_cases.json").write_text(json.dumps([case]))
(meta / "checksum_inference.json").write_text(json.dumps({case + ".pkl": checks[case + ".pkl"]}))
PY
  fi
  {
    echo "stage: preprocess"
    cd "$b"
    "$WORK/venv/bin/python" preprocess.py --raw_data_dir "$b/build/raw_data" --results_dir "$b/build/preprocessed_data" --mode preprocess --num_proc "${PREPROCESS_PROCS:-8}"
  } 2>&1 | tee "$OUT/preprocess.log"
  touch "$WORK/.preprocess.done"
}

run_gpu() {
  [[ -f "$WORK/.run.done" ]] && return
  local b="$WORK/benchmark" free_mib
  {
    echo "stage: run"
    echo "GPU memory immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    (( free_mib >= 12000 )) || { echo "insufficient free GPU memory: ${free_mib} MiB" >&2; exit 1; }
    cd "$b"
    export LOG_PATH="$b/build/logs"
    export SKIP_VERIFY_ACCURACY=1
    "$WORK/venv/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy --model="$b/build/model/3dunet_kits19_pytorch.ptc" --preprocessed_data_dir="$b/build/preprocessed_data" --mlperf_conf="$b/build/mlperf.conf" --user_conf="$b/user.conf"
  } 2>&1 | tee "$OUT/gpu.log"
  cp "$b/build/logs/mlperf_log_accuracy.json" "$OUT/mlperf_log_accuracy.json"
  cp "$b/build/logs/mlperf_log_detail.txt" "$OUT/mlperf_log_detail.txt"
  touch "$WORK/.run.done"
}

run_evaluate() {
  [[ -f "$WORK/.evaluate.done" ]] && return
  local b="$WORK/benchmark"
  mkdir -p "$b/build/postprocessed_data"
  {
    echo "stage: evaluate"
    cd "$b"
    "$WORK/venv/bin/python" accuracy_kits.py --log_file="$OUT/mlperf_log_accuracy.json" --preprocessed_data_dir="$b/build/preprocessed_data" --postprocessed_data_dir="$b/build/postprocessed_data" --num_proc "${EVAL_PROCS:-8}"
  } 2>&1 | tee "$OUT/evaluator.log" "$OUT/accuracy.txt"
  touch "$WORK/.evaluate.done"
}

[[ -z "$FROM" ]] || clean_from "$FROM"
start=0
[[ -z "$FROM" ]] || start="$(stage_index "$FROM")"
if [[ -n "$STAGE" ]]; then
  case "$STAGE" in build) run_build;; preprocess) run_preprocess;; run) run_gpu;; evaluate) run_evaluate;; *) echo "invalid stage: $STAGE" >&2; exit 2;; esac
else
  (( start > 0 )) || run_build
  (( start > 1 )) || run_preprocess
  (( start > 2 )) || run_gpu
  run_evaluate
fi
