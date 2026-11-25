#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM_STAGE=""
while (($#)); do
    case "$1" in
        --target) TARGET="${2:?missing target id}"; shift 2 ;;
        --stage) STAGE="${2:?missing stage name}"; shift 2 ;;
        --from) FROM_STAGE="${2:?missing stage name}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
    echo "unsupported target: $TARGET" >&2
    exit 2
fi
if [[ -n "$STAGE" && -n "$FROM_STAGE" ]]; then
    echo "--stage and --from are mutually exclusive" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/repo"
CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER="2026-11-10"

OUT="${OUTPUT_DIR:-$ROOT/$TARGET/out}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
TARGET_DIR="$ROOT/$TARGET"
WORK="$TARGET_DIR/work"
VENV="$TARGET_DIR/venv"
SRC="$WORK/src"
LOADGEN_SRC="$WORK/loadgen"
PREPROCESSED="$WORK/preprocessed_data"
POSTPROCESSED="$WORK/postprocessed_data"
RAW_VIEW="$WORK/raw_data"
mkdir -p "$WORK"

if [[ ! -d "$REPO/vision/medical_imaging/3d-unet-kits19" || ! -d "$REPO/loadgen" ]]; then
    echo "fresh repository copy is missing required benchmark sources" >&2
    exit 1
fi
if [[ ! -f "$MODEL_INPUT" || ! -d "$RAW_INPUT" ]]; then
    echo "required cached model or KiTS19 raw data is missing" >&2
    exit 1
fi

clear_from() {
    case "$1" in
        setup)
            python3.11 - "$VENV" "$WORK" <<'PY'
import shutil, sys
for path in sys.argv[1:]:
    shutil.rmtree(path, ignore_errors=True)
PY
            mkdir -p "$WORK"
            rm -f "$TARGET_DIR/.setup.ok" "$TARGET_DIR/.preprocess.ok" "$TARGET_DIR/.run.ok"
            rm -f "$OUT/build.log" "$OUT/preprocess.log" \
                  "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
                  "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json" \
                  "$OUT/accuracy.txt" "$OUT/evaluator.log" "$OUT/run.log" "$OUT/gpu_memory.log"
            ;;
        preprocess)
            python3.11 - "$PREPROCESSED" "$POSTPROCESSED" "$RAW_VIEW" <<'PY'
import shutil, sys
for path in sys.argv[1:]:
    shutil.rmtree(path, ignore_errors=True)
PY
            rm -f "$TARGET_DIR/.preprocess.ok" "$TARGET_DIR/.run.ok"
            rm -f "$OUT/preprocess.log" "$OUT/mlperf_log_accuracy.json" \
                  "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" \
                  "$OUT/mlperf_log_trace.json" "$OUT/accuracy.txt" \
                  "$OUT/evaluator.log" "$OUT/run.log" "$OUT/gpu_memory.log"
            ;;
        run)
            python3.11 - "$POSTPROCESSED" <<'PY'
import shutil, sys
shutil.rmtree(sys.argv[1], ignore_errors=True)
PY
            rm -f "$TARGET_DIR/.run.ok"
            rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
                  "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json" \
                  "$OUT/accuracy.txt" "$OUT/evaluator.log" "$OUT/run.log" "$OUT/gpu_memory.log"
            ;;
        *) echo "unknown stage: $1" >&2; exit 2 ;;
    esac
}

stage_setup() {
    if [[ -f "$TARGET_DIR/.setup.ok" ]]; then return; fi
    {
        echo "setup: Python 3.11 environment and repository LoadGen"
        uv venv --python python3.11 "$VENV"
        uv pip install --python "$VENV/bin/python" \
            "numpy==1.26.4" "scipy==1.11.4" "nibabel==5.2.1" \
            "pandas==1.5.3" "torch==2.1.2" "setuptools<81" wheel "pybind11==2.11.1"
        mkdir -p "$WORK"
        cp -a "$REPO/loadgen" "$LOADGEN_SRC"
        cp -a "$REPO/vision/medical_imaging/3d-unet-kits19" "$SRC"
        uv pip install --python "$VENV/bin/python" --no-build-isolation "$LOADGEN_SRC"
        "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("nibabel", nibabel.__version__, "pandas", pandas.__version__)
print("loadgen import ok")
PY
    } 2>&1 | tee "$OUT/build.log"
    touch "$TARGET_DIR/.setup.ok"
}

stage_preprocess() {
    if [[ -f "$TARGET_DIR/.preprocess.ok" ]]; then return; fi
    mkdir -p "$RAW_VIEW" "$PREPROCESSED"
    cp "$REPO/vision/medical_imaging/3d-unet-kits19/meta/inference_cases.json" "$SRC/meta/inference_cases.json"
    cp "$REPO/vision/medical_imaging/3d-unet-kits19/meta/checksum_inference.json" "$SRC/meta/checksum_inference.json"
    (
        cd "$SRC"
        "$VENV/bin/python" - "$RAW_INPUT" "$RAW_VIEW" "${REPRO_CASES:-}" <<'PY'
import json, pathlib, sys
raw, view, requested = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
meta = pathlib.Path("meta")
with (meta / "inference_cases.json").open() as f:
    official = json.load(f)
cases = [x for x in requested.split(",") if x] if requested else official
unknown = sorted(set(cases) - set(official))
if unknown:
    raise SystemExit(f"unknown REPRO_CASES: {unknown}")
with (meta / "checksum_inference.json").open() as f:
    checksums = json.load(f)
with (meta / "inference_cases.json").open("w") as f:
    json.dump(cases, f, indent=4)
    f.write("\n")
with (meta / "checksum_inference.json").open("w") as f:
    json.dump({f"{c}.pkl": checksums[f"{c}.pkl"] for c in cases}, f, indent=4, sort_keys=True)
    f.write("\n")
for case in cases:
    source_case = "case_00185" if case == "case_00400" else case
    source = raw / source_case
    if not all((source / name).is_file() for name in ("imaging.nii.gz", "segmentation.nii.gz")):
        raise SystemExit(f"raw input missing for {case}: {source}")
    destination = view / case
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source, target_is_directory=True)
print(f"prepared raw view for {len(cases)} cases")
PY
        "$VENV/bin/python" preprocess.py \
            --raw_data_dir "$RAW_VIEW" --results_dir "$PREPROCESSED" \
            --mode preprocess --num_proc 8
    ) 2>&1 | tee "$OUT/preprocess.log"
    touch "$TARGET_DIR/.preprocess.ok"
}

stage_run() {
    if [[ -f "$TARGET_DIR/.run.ok" ]]; then return; fi
    mkdir -p "$POSTPROCESSED"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader | tee "$OUT/gpu_memory.log"
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if [[ ! "$free_mib" =~ ^[0-9]+$ || "$free_mib" -lt 16000 ]]; then
        echo "insufficient free GPU memory before run: ${free_mib:-unknown} MiB" >&2
        exit 1
    fi
    (
        cd "$SRC"
        "$VENV/bin/python" - "$MODEL_INPUT" <<'PY'
import sys, torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise SystemExit("CUDA is unavailable")
model = torch.jit.load(sys.argv[1], map_location=device)
try:
    actual = next(model.parameters()).device
except StopIteration:
    actual = device
print(f"pytorch device: {actual}")
del model
torch.cuda.empty_cache()
PY
        LOG_PATH="$OUT" SKIP_VERIFY_ACCURACY=1 "$VENV/bin/python" run.py \
            --backend=pytorch --scenario=Offline --accuracy \
            --model="$MODEL_INPUT" --preprocessed_data_dir="$PREPROCESSED" \
            --mlperf_conf="$REPO/mlperf.conf" --user_conf=user.conf
    ) 2>&1 | tee "$OUT/run.log"
    (
        cd "$SRC"
        "$VENV/bin/python" accuracy_kits.py \
            --log_file="$OUT/mlperf_log_accuracy.json" \
            --preprocessed_data_dir="$PREPROCESSED" \
            --postprocessed_data_dir="$POSTPROCESSED" --num_proc=8
    ) 2>&1 | tee "$OUT/accuracy.txt" "$OUT/evaluator.log"
    touch "$TARGET_DIR/.run.ok"
}

stages=(setup preprocess run)
if [[ -n "$FROM_STAGE" ]]; then
    clear_from "$FROM_STAGE"
fi
for candidate in "${stages[@]}"; do
    if [[ -n "$STAGE" && "$candidate" != "$STAGE" ]]; then continue; fi
    if [[ -n "$FROM_STAGE" ]]; then
        case "$FROM_STAGE:$candidate" in
            setup:*) ;;
            preprocess:setup) continue ;;
            run:setup|run:preprocess) continue ;;
        esac
    fi
    "stage_$candidate"
done
