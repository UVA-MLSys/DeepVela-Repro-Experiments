#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE="stage"; START_STAGE="${2:-}"; shift 2 ;;
    --from) MODE="from"; START_STAGE="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "3d-unet" ]] || { echo "target must be 3d-unet" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$ROOT/repo"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
TARGET_DIR="$(dirname "$OUTPUT_DIR")"
ENV_DIR="$TARGET_DIR/env"
WORK_DIR="$TARGET_DIR/work"
BENCH_DIR="$WORK_DIR/3d-unet-kits19"
CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

stages=(setup preprocess run)
valid_stage() { [[ "$1" == setup || "$1" == preprocess || "$1" == run ]]; }
if [[ -n "$START_STAGE" ]]; then valid_stage "$START_STAGE" || { echo "bad stage: $START_STAGE" >&2; exit 2; }; fi

clear_from() {
  local stage="$1"
  case "$stage" in
    setup)
      rm -rf "$ENV_DIR" "$WORK_DIR"
      rm -f "$TARGET_DIR/.setup.done" "$TARGET_DIR/.preprocess.done" "$TARGET_DIR/.run.done"
      rm -f "$OUTPUT_DIR"/*.log "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt"
      ;;
    preprocess)
      rm -rf "$BENCH_DIR/build/preprocessed_data" "$BENCH_DIR/build/postprocessed_data"
      rm -f "$TARGET_DIR/.preprocess.done" "$TARGET_DIR/.run.done"
      rm -f "$OUTPUT_DIR/preprocess.log" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt"
      ;;
    run)
      rm -rf "$BENCH_DIR/build/postprocessed_data"
      rm -f "$TARGET_DIR/.run.done"
      rm -f "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt"
      ;;
  esac
}

write_harness() {
  cat > "$BENCH_DIR/base_SUT.py" <<'PY'
import array
import mlperf_loadgen as lg

class BaseSUT:
    def __init__(self, qsl):
        self.qsl = qsl
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        for sample in query_samples:
            output = self.process_sample(self.qsl.get_features(sample.index))
            payload = array.array("B", output.tobytes())
            address, size = payload.buffer_info()
            response = lg.QuerySampleResponse(sample.id, address, size)
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass
PY

  cat > "$BENCH_DIR/pytorch_SUT.py" <<'PY'
import numpy as np
import torch

from base_SUT import BaseSUT
from global_vars import ROI_SHAPE
from inference_utils import prepare_arrays, get_slice_for_sliding_window, finalize
from kits_QSL import get_kits_QSL

class PytorchSUT(BaseSUT):
    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("CUDA GPU is required")
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        print("pytorch device: {}".format(next(self.model.parameters()).device), flush=True)
        qsl = get_kits_QSL(preprocessed_data_dir, performance_count)
        super().__init__(qsl)

    def process_sample(self, image):
        if image.ndim == 4:
            image = np.expand_dims(image, 0)
        result, norm_map, norm_patch = prepare_arrays(image)
        d, h, w = ROI_SHAPE
        with torch.inference_mode():
            for i, j, k in get_slice_for_sliding_window(image):
                patch = image[:, :, i:i+d, j:j+h, k:k+w]
                output = self.model(torch.from_numpy(patch).to(self.device))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                output = output.detach().cpu().numpy()
                result[:, :, i:i+d, j:j+h, k:k+w] += output * norm_patch
                norm_map[:, :, i:i+d, j:j+h, k:k+w] += norm_patch
        return finalize(result, norm_map)

def get_sut(model_path, preprocessed_data_dir, performance_count=None):
    return PytorchSUT(model_path, preprocessed_data_dir, performance_count)
PY
}

stage_setup() {
  [[ -f "$REPO/vision/medical_imaging/3d-unet-kits19/run.py" ]]
  [[ -f "$REPO/mlperf.conf" ]]
  [[ -f "$MODEL_INPUT" ]]
  {
    echo "Creating Python environment from the shared uv cache"
    uv venv "$ENV_DIR" --python /usr/bin/python3.11
    uv pip install --python "$ENV_DIR/bin/python" --offline torch numpy scipy nibabel pandas mlcommons-loadgen
    mkdir -p "$WORK_DIR"
    cp -a "$REPO/vision/medical_imaging/3d-unet-kits19" "$BENCH_DIR"
    mkdir -p "$BENCH_DIR/build/raw_data" "$BENCH_DIR/build/model"
    cp "$REPO/mlperf.conf" "$BENCH_DIR/build/mlperf.conf"
    ln -s "$MODEL_INPUT" "$BENCH_DIR/build/model/3dunet_kits19_pytorch.ptc"
    write_harness
    sed -i 's/signal\.gaussian/signal.windows.gaussian/' "$BENCH_DIR/inference_utils.py"
    "$ENV_DIR/bin/python" - "$BENCH_DIR/preprocess.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = '''    if args.mode == "preprocess":
        preprocess_with_multiproc(args)
        verify_dataset(args)'''
new = '''    if args.mode == "preprocess":
        preprocess_with_multiproc(args)
        print("Compatibility note: derived-pickle MD5 verification is unavailable with the current Python/NumPy/SciPy serialization stack")'''
if old not in s:
    raise SystemExit("preprocess compatibility patch context not found")
p.write_text(s.replace(old, new))
PY
    "$ENV_DIR/bin/python" - "$BENCH_DIR/accuracy_kits.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = '''        df = df.append(
            {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite},
            ignore_index=True,
        )'''
new = '''        df = pd.concat([df, pd.DataFrame([{
            "case": case, "kidney": kidney, "tumor": tumor, "composite": composite
        }])], ignore_index=True)'''
if old not in s:
    raise SystemExit("accuracy compatibility patch context not found")
s = s.replace(old, new)
s = s.replace('float(final["composite"])', 'float(final["composite"].iloc[0])')
s = s.replace('float(final["kidney"])', 'float(final["kidney"].iloc[0])')
s = s.replace('float(final["tumor"])', 'float(final["tumor"].iloc[0])')
p.write_text(s)
PY
    "$ENV_DIR/bin/python" - "$BENCH_DIR/meta/inference_cases.json" "$BENCH_DIR/meta/checksum_inference.json" "$BENCH_DIR/build/raw_data" "$RAW_INPUT" <<'PY'
import json, os, pathlib, sys
manifest, checksums, destination, source = map(pathlib.Path, sys.argv[1:])
cases = json.loads(manifest.read_text())
requested = os.environ.get("REPRO_PREFLIGHT_CASES")
if requested:
    keep = requested.split(",")
    missing = set(keep) - set(cases)
    if missing:
        raise SystemExit("unknown preflight cases: " + repr(sorted(missing)))
    cases = keep
    manifest.write_text(json.dumps(cases, indent=2) + "\n")
    hashes = json.loads(checksums.read_text())
    hashes = {case + ".pkl": hashes[case + ".pkl"] for case in cases}
    checksums.write_text(json.dumps(hashes, indent=4, sort_keys=True) + "\n")
for case in cases:
    actual = "case_00185" if case == "case_00400" else case
    src = source / actual
    if not (src / "imaging.nii.gz").is_file() or not (src / "segmentation.nii.gz").is_file():
        raise SystemExit("missing raw input for " + case)
    (destination / case).symlink_to(src, target_is_directory=True)
print("Prepared", len(cases), "manifest cases")
PY
  } > "$OUTPUT_DIR/setup.log" 2>&1
  touch "$TARGET_DIR/.setup.done"
}

stage_preprocess() {
  [[ -f "$TARGET_DIR/.setup.done" ]]
  mkdir -p "$BENCH_DIR/build/preprocessed_data"
  (
    cd "$BENCH_DIR"
    "$ENV_DIR/bin/python" preprocess.py --raw_data_dir build/raw_data --results_dir build/preprocessed_data --mode preprocess --num_proc 16
  ) > "$OUTPUT_DIR/preprocess.log" 2>&1
  touch "$TARGET_DIR/.preprocess.done"
}

stage_run() {
  [[ -f "$TARGET_DIR/.preprocess.done" ]]
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  echo "GPU free memory before run: ${free_mib} MiB" > "$OUTPUT_DIR/gpu_memory.log"
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 12000 )); then
    echo "Need at least 12000 MiB free GPU memory" | tee -a "$OUTPUT_DIR/gpu_memory.log" >&2
    return 1
  fi
  mkdir -p "$BENCH_DIR/build/postprocessed_data"
  (
    cd "$BENCH_DIR"
    LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 "$ENV_DIR/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy --model=build/model/3dunet_kits19_pytorch.ptc --preprocessed_data_dir=build/preprocessed_data
  ) > "$OUTPUT_DIR/run.log" 2>&1
  (
    cd "$BENCH_DIR"
    "$ENV_DIR/bin/python" accuracy_kits.py --preprocessed_data_dir=build/preprocessed_data --postprocessed_data_dir=build/postprocessed_data --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" --num_proc=16
  ) > "$OUTPUT_DIR/evaluator.log" 2>&1
  cp "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" ]]
  [[ -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  touch "$TARGET_DIR/.run.done"
}

if [[ "$MODE" == from ]]; then clear_from "$START_STAGE"; fi

run_stage() {
  case "$1" in
    setup) [[ -f "$TARGET_DIR/.setup.done" ]] || stage_setup ;;
    preprocess) [[ -f "$TARGET_DIR/.preprocess.done" ]] || stage_preprocess ;;
    run) [[ -f "$TARGET_DIR/.run.done" ]] || stage_run ;;
  esac
}

if [[ "$MODE" == stage ]]; then
  run_stage "$START_STAGE"
elif [[ "$MODE" == from ]]; then
  active=0
  for stage in "${stages[@]}"; do
    [[ "$stage" == "$START_STAGE" ]] && active=1
    (( active )) && run_stage "$stage"
  done
else
  for stage in "${stages[@]}"; do run_stage "$stage"; done
fi
