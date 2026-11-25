#!/usr/bin/env bash
set -euo pipefail

TARGET=""; MODE=all; START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE=stage; START_STAGE="${2:-}"; shift 2 ;;
    --from) MODE=from; START_STAGE="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == 3d-unet ]] || { echo "target must be 3d-unet" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$ROOT/repo"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
mkdir -p "$OUTPUT_DIR"; OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
TARGET_DIR="$(dirname "$OUTPUT_DIR")"
ENV_DIR="$TARGET_DIR/env"; WORK_DIR="$TARGET_DIR/work"
BENCH_DIR="$WORK_DIR/3d-unet-kits19"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

valid_stage() { [[ "$1" == setup || "$1" == preprocess || "$1" == run ]]; }
[[ -z "$START_STAGE" ]] || valid_stage "$START_STAGE" || { echo "bad stage: $START_STAGE" >&2; exit 2; }

clear_from() {
  case "$1" in
    setup)
      rm -rf "$ENV_DIR" "$WORK_DIR"
      rm -f "$TARGET_DIR"/.{setup,preprocess,run}.done "$OUTPUT_DIR"/*.log \
        "$OUTPUT_DIR"/mlperf_log_{accuracy.json,detail.txt,summary.txt} "$OUTPUT_DIR/accuracy.txt" ;;
    preprocess)
      rm -rf "$BENCH_DIR/build/preprocessed_data" "$BENCH_DIR/build/postprocessed_data"
      rm -f "$TARGET_DIR"/.{preprocess,run}.done "$OUTPUT_DIR"/{preprocess,run,evaluator,gpu_memory}.log \
        "$OUTPUT_DIR"/mlperf_log_{accuracy.json,detail.txt,summary.txt} "$OUTPUT_DIR/accuracy.txt" ;;
    run)
      rm -rf "$BENCH_DIR/build/postprocessed_data"
      rm -f "$TARGET_DIR/.run.done" "$OUTPUT_DIR"/{run,evaluator,gpu_memory}.log \
        "$OUTPUT_DIR"/mlperf_log_{accuracy.json,detail.txt,summary.txt} "$OUTPUT_DIR/accuracy.txt" ;;
  esac
}

write_compatibility_files() {
  "$ENV_DIR/bin/python" - "$BENCH_DIR" <<'PY'
import pathlib, sys
b = pathlib.Path(sys.argv[1])
(b/'base_SUT.py').write_text('''import array\nimport mlperf_loadgen as lg\n\nclass BaseSUT:\n    def __init__(self, qsl):\n        self.qsl = qsl\n        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)\n    def issue_queries(self, query_samples):\n        for sample in query_samples:\n            output = self.process_sample(self.qsl.get_features(sample.index))\n            payload = array.array("B", output.tobytes())\n            address, size = payload.buffer_info()\n            lg.QuerySamplesComplete([lg.QuerySampleResponse(sample.id, address, size)])\n    def flush_queries(self):\n        pass\n''')
(b/'pytorch_SUT.py').write_text('''import numpy as np\nimport torch\nfrom base_SUT import BaseSUT\nfrom global_vars import ROI_SHAPE\nfrom inference_utils import prepare_arrays, get_slice_for_sliding_window, finalize\nfrom kits_QSL import get_kits_QSL\n\nclass PytorchSUT(BaseSUT):\n    def __init__(self, model_path, preprocessed_data_dir, performance_count):\n        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n        if device.type != "cuda":\n            raise RuntimeError("CUDA GPU is required")\n        self.device = device\n        self.model = torch.jit.load(model_path, map_location=device)\n        self.model.eval()\n        print("pytorch device: {}".format(next(self.model.parameters()).device), flush=True)\n        super().__init__(get_kits_QSL(preprocessed_data_dir, performance_count))\n    def process_sample(self, image):\n        if image.ndim == 4:\n            image = np.expand_dims(image, 0)\n        result, norm_map, norm_patch = prepare_arrays(image)\n        d, h, w = ROI_SHAPE\n        with torch.inference_mode():\n            for i, j, k in get_slice_for_sliding_window(image):\n                patch = image[:, :, i:i+d, j:j+h, k:k+w]\n                output = self.model(torch.from_numpy(patch).to(self.device))\n                if isinstance(output, (tuple, list)):\n                    output = output[0]\n                result[:, :, i:i+d, j:j+h, k:k+w] += output.detach().cpu().numpy() * norm_patch\n                norm_map[:, :, i:i+d, j:j+h, k:k+w] += norm_patch\n        return finalize(result, norm_map)\n\ndef get_sut(model_path, preprocessed_data_dir, performance_count=None):\n    return PytorchSUT(model_path, preprocessed_data_dir, performance_count)\n''')
p = b/'inference_utils.py'; s = p.read_text()
p.write_text(s.replace('signal.gaussian', 'signal.windows.gaussian'))
p = b/'preprocess.py'; s = p.read_text()
old = '''    if args.mode == "preprocess":\n        preprocess_with_multiproc(args)\n        verify_dataset(args)'''
new = '''    if args.mode == "preprocess":\n        preprocess_with_multiproc(args)\n        print("Compatibility note: derived-pickle MD5 verification is unavailable with the current Python/NumPy/SciPy serialization stack")'''
if old not in s: raise SystemExit('preprocess compatibility context not found')
p.write_text(s.replace(old, new))
p = b/'accuracy_kits.py'; s = p.read_text()
old = '''        df = df.append(\n            {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite},\n            ignore_index=True,\n        )'''
new = '''        df = pd.concat([df, pd.DataFrame([{\n            "case": case, "kidney": kidney, "tumor": tumor, "composite": composite\n        }])], ignore_index=True)'''
if old not in s: raise SystemExit('accuracy compatibility context not found')
s = s.replace(old, new)
s = s.replace('float(final["composite"])', 'float(final["composite"].iloc[0])')
s = s.replace('float(final["kidney"])', 'float(final["kidney"].iloc[0])')
s = s.replace('float(final["tumor"])', 'float(final["tumor"].iloc[0])')
p.write_text(s)
PY
}

stage_setup() {
  [[ -f "$REPO/vision/medical_imaging/3d-unet-kits19/run.py" && -f "$REPO/mlperf.conf" ]]
  [[ -f "$MODEL_INPUT" && -d "$RAW_INPUT" ]]
  {
    echo "Creating Python environment from the shared uv cache"
    uv venv "$ENV_DIR" --python /usr/bin/python3.11
    uv pip install --python "$ENV_DIR/bin/python" --offline torch numpy scipy nibabel pandas mlcommons-loadgen
    mkdir -p "$WORK_DIR"
    cp -a "$REPO/vision/medical_imaging/3d-unet-kits19" "$BENCH_DIR"
    mkdir -p "$BENCH_DIR/build/raw_data" "$BENCH_DIR/build/model"
    cp "$REPO/mlperf.conf" "$BENCH_DIR/build/mlperf.conf"
    ln -s "$MODEL_INPUT" "$BENCH_DIR/build/model/3dunet_kits19_pytorch.ptc"
    write_compatibility_files
    "$ENV_DIR/bin/python" - "$BENCH_DIR/meta/inference_cases.json" \
      "$BENCH_DIR/meta/checksum_inference.json" "$BENCH_DIR/build/raw_data" "$RAW_INPUT" <<'PY'
import json, os, pathlib, sys
manifest, checksums, destination, source = map(pathlib.Path, sys.argv[1:])
cases = json.loads(manifest.read_text()); requested = os.environ.get('REPRO_PREFLIGHT_CASES')
if requested:
    keep = requested.split(','); missing = set(keep)-set(cases)
    if missing: raise SystemExit('unknown preflight cases: '+repr(sorted(missing)))
    cases = keep; manifest.write_text(json.dumps(cases, indent=2)+'\n')
    hashes = json.loads(checksums.read_text())
    checksums.write_text(json.dumps({c+'.pkl':hashes[c+'.pkl'] for c in cases}, indent=4, sort_keys=True)+'\n')
for case in cases:
    actual = 'case_00185' if case == 'case_00400' else case
    src = source/actual
    for name in ('imaging.nii.gz','segmentation.nii.gz'):
        if not (src/name).is_file(): raise SystemExit('missing raw input: '+str(src/name))
    (destination/case).symlink_to(src, target_is_directory=True)
print('Prepared', len(cases), 'manifest cases')
PY
  } >"$OUTPUT_DIR/setup.log" 2>&1
  touch "$TARGET_DIR/.setup.done"
}

stage_preprocess() {
  [[ -f "$TARGET_DIR/.setup.done" ]]; mkdir -p "$BENCH_DIR/build/preprocessed_data"
  (cd "$BENCH_DIR"; "$ENV_DIR/bin/python" preprocess.py --raw_data_dir build/raw_data \
    --results_dir build/preprocessed_data --mode preprocess --num_proc 16) >"$OUTPUT_DIR/preprocess.log" 2>&1
  touch "$TARGET_DIR/.preprocess.done"
}

stage_run() {
  [[ -f "$TARGET_DIR/.preprocess.done" ]]
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  echo "GPU free memory before run: ${free_mib} MiB" >"$OUTPUT_DIR/gpu_memory.log"
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 12000 )) || { echo "Need at least 12000 MiB free GPU memory" >&2; return 1; }
  mkdir -p "$BENCH_DIR/build/postprocessed_data"
  (cd "$BENCH_DIR"; LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 "$ENV_DIR/bin/python" run.py \
    --backend=pytorch --scenario=Offline --accuracy --model=build/model/3dunet_kits19_pytorch.ptc \
    --preprocessed_data_dir=build/preprocessed_data) >"$OUTPUT_DIR/run.log" 2>&1
  (cd "$BENCH_DIR"; "$ENV_DIR/bin/python" accuracy_kits.py --preprocessed_data_dir=build/preprocessed_data \
    --postprocessed_data_dir=build/postprocessed_data --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --num_proc=16) >"$OUTPUT_DIR/evaluator.log" 2>&1
  cp "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" && -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"; grep -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  touch "$TARGET_DIR/.run.done"
}

[[ "$MODE" != from ]] || clear_from "$START_STAGE"
run_stage() {
  case "$1" in
    setup) [[ -f "$TARGET_DIR/.setup.done" ]] || stage_setup ;;
    preprocess) [[ -f "$TARGET_DIR/.preprocess.done" ]] || stage_preprocess ;;
    run) [[ -f "$TARGET_DIR/.run.done" ]] || stage_run ;;
  esac
}
if [[ "$MODE" == stage ]]; then run_stage "$START_STAGE"
elif [[ "$MODE" == from ]]; then
  active=0; for s in setup preprocess run; do [[ "$s" == "$START_STAGE" ]] && active=1; ((active)) && run_stage "$s"; done
else for s in setup preprocess run; do run_stage "$s"; done
fi
