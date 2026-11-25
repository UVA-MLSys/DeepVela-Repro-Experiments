#!/usr/bin/env bash
set -euo pipefail

TARGET=""
SELECTED_STAGE=""
FROM_STAGE=""
PREFLIGHT=0
while (($#)); do
    case "$1" in
        --target) TARGET="${2:?missing target id}"; shift 2 ;;
        --stage) SELECTED_STAGE="${2:?missing stage name}"; shift 2 ;;
        --from) FROM_STAGE="${2:?missing stage name}"; shift 2 ;;
        --preflight) PREFLIGHT=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
    echo "unsupported target: $TARGET" >&2
    exit 2
fi
if [[ -n "$SELECTED_STAGE" && -n "$FROM_STAGE" ]]; then
    echo "--stage and --from are mutually exclusive" >&2
    exit 2
fi
for stage_name in "$SELECTED_STAGE" "$FROM_STAGE"; do
    if [[ -n "$stage_name" && "$stage_name" != "environment" && "$stage_name" != "data" && "$stage_name" != "run" ]]; then
        echo "unknown stage: $stage_name (expected environment, data, or run)" >&2
        exit 2
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="$SCRIPT_DIR/repo"
CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
KITS19_RAW_DIR="${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_EXCLUDE_NEWER=2026-11-10

PYTHON_BIN=""
for python_candidate in /home/mi3se/.local/bin/python3.11 "$(command -v python3.11 2>/dev/null || true)" /usr/bin/python3.11; do
    if [[ -x "$python_candidate" ]]; then
        python_include="$($python_candidate -c 'import sysconfig; print(sysconfig.get_path("include"))')"
        if [[ -f "$python_include/Python.h" ]]; then
            PYTHON_BIN="$python_candidate"
            break
        fi
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    echo "no Python 3.11 interpreter with development headers was found" >&2
    exit 1
fi

if [[ ! -d "$SOURCE_REPO/vision/medical_imaging/3d-unet-kits19" ]]; then
    echo "fresh source copy not found at $SOURCE_REPO" >&2
    exit 1
fi
if [[ ! -d "$KITS19_RAW_DIR" || ! -s "$MODEL_DIR/3dunet_kits19_pytorch.ptc" ]]; then
    echo "required read-only dataset or model input is missing" >&2
    exit 1
fi

if ((PREFLIGHT)); then
    TARGET_DIR="$SCRIPT_DIR/3d-unet/preflight"
else
    TARGET_DIR="$SCRIPT_DIR/3d-unet"
fi
WORK_DIR="$TARGET_DIR/work"
SRC_DIR="$WORK_DIR/repo"
APP_DIR="$SRC_DIR/vision/medical_imaging/3d-unet-kits19"
ENV_DIR="$WORK_DIR/env"
BUILD_DIR="$APP_DIR/build"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
    echo "OUTPUT_DIR must be set" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR" "$TARGET_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

ENV_MARKER="$WORK_DIR/.environment.success"
DATA_MARKER="$WORK_DIR/.data.success"
RUN_MARKER="$WORK_DIR/.run.success"

clear_from() {
    case "$1" in
        environment)
            rm -rf -- "$WORK_DIR"
            rm -f -- "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt"
            ;;
        data)
            rm -rf -- "$APP_DIR/build"
            rm -f -- "$DATA_MARKER" "$RUN_MARKER"
            rm -f -- "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt"
            ;;
        run)
            rm -rf -- "$BUILD_DIR/postprocessed_data"
            rm -f -- "$RUN_MARKER"
            rm -f -- "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt"
            ;;
    esac
}

if [[ -n "$FROM_STAGE" ]]; then
    clear_from "$FROM_STAGE"
fi

stage_environment() {
    [[ -f "$ENV_MARKER" ]] && return 0
    mkdir -p "$WORK_DIR"
    rm -rf -- "$SRC_DIR" "$ENV_DIR"
    cp -a "$SOURCE_REPO" "$SRC_DIR"

    "$PYTHON_BIN" - "$SRC_DIR/loadgen/setup.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
stale = '    "trace_generator.h",\n'
if stale in s:
    p.write_text(s.replace(stale, "", 1))
PY

    {
        date -Ins
        echo "source_identity=3fbc329939999c13d0a7b5e67fb2092287e06047"
        echo "python=$PYTHON_BIN"
        uv --version
        /usr/bin/time -v uv venv --python "$PYTHON_BIN" "$ENV_DIR"
        /usr/bin/time -v uv pip install --python "$ENV_DIR/bin/python" --offline \
            'numpy==1.26.4' 'scipy==1.13.1' 'nibabel==5.4.2' 'pandas==1.5.3' \
            'torch==2.1.2' 'pybind11==2.13.6' 'setuptools==84.2.1' 'wheel==0.48.0'
        /usr/bin/time -v uv pip install --python "$ENV_DIR/bin/python" --offline --no-build-isolation "$SRC_DIR/loadgen"
        "$ENV_DIR/bin/python" -c 'import torch, numpy, scipy, nibabel, pandas, mlperf_loadgen; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("numpy", numpy.__version__, "scipy", scipy.__version__, "nibabel", nibabel.__version__, "pandas", pandas.__version__)'
    } >"$TARGET_DIR/build.log" 2>&1

    "$ENV_DIR/bin/python" - "$APP_DIR/pytorch_SUT.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
needle = "        self.model.eval()\n"
replacement = needle + '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n'
if replacement not in s:
    if needle not in s:
        raise SystemExit("device instrumentation insertion point not found")
    p.write_text(s.replace(needle, replacement, 1))
PY
    "$ENV_DIR/bin/python" - "$APP_DIR/inference_utils.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
legacy = "signal.gaussian(n, std)"
current = "signal.windows.gaussian(n, std)"
if legacy in s:
    p.write_text(s.replace(legacy, current, 1))
PY
    touch "$ENV_MARKER"
}

stage_data() {
    [[ -f "$DATA_MARKER" ]] && return 0
    [[ -f "$ENV_MARKER" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
    mkdir -p "$BUILD_DIR/raw_data" "$BUILD_DIR/preprocessed_data" "$BUILD_DIR/model"
    cp "$SRC_DIR/loadgen/mlperf.conf" "$BUILD_DIR/mlperf.conf"
    ln -sfn "$MODEL_DIR/3dunet_kits19_pytorch.ptc" "$BUILD_DIR/model/3dunet_kits19_pytorch.ptc"

    if ((PREFLIGHT)); then
        "$ENV_DIR/bin/python" - "$APP_DIR/meta/inference_cases.json" "$APP_DIR/meta/checksum_inference.json" <<'PY'
import json
from pathlib import Path
import sys
case = "case_00000"
cases_path, sums_path = map(Path, sys.argv[1:])
cases_path.write_text(json.dumps([case], indent=4) + "\n")
sums = json.loads(sums_path.read_text())
key = case + ".pkl"
sums_path.write_text(json.dumps({key: sums[key]}, indent=4, sort_keys=True) + "\n")
PY
    fi

    "$ENV_DIR/bin/python" - "$APP_DIR/meta/inference_cases.json" "$KITS19_RAW_DIR" "$BUILD_DIR/raw_data" <<'PY'
import json
import os
from pathlib import Path
import sys
manifest, raw_root, local_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
for case in json.loads(manifest.read_text()):
    source_case = "case_00185" if case == "case_00400" else case
    target = local_root / case
    target.mkdir(parents=True, exist_ok=True)
    for name in ("imaging.nii.gz", "segmentation.nii.gz"):
        source = raw_root / source_case / name
        if not source.is_file():
            raise SystemExit(f"missing input {source}")
        link = target / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(source)
PY

    {
        date -Ins
        echo "raw_input=$KITS19_RAW_DIR"
        echo "model_input=$MODEL_DIR/3dunet_kits19_pytorch.ptc"
        sha256sum "$MODEL_DIR/3dunet_kits19_pytorch.ptc"
        cd "$APP_DIR"
        /usr/bin/time -v "$ENV_DIR/bin/python" preprocess.py \
            --raw_data_dir "$BUILD_DIR/raw_data" \
            --results_dir "$BUILD_DIR/preprocessed_data" \
            --mode preprocess --num_proc 4
    } >"$TARGET_DIR/preprocess.log" 2>&1
    touch "$DATA_MARKER"
}

wait_for_gpu_memory() {
    local required_mib=20000
    local free_mib
    for _attempt in $(seq 1 40); do
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
        free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && ((free_mib >= required_mib)); then
            return 0
        fi
        echo "waiting for at least ${required_mib} MiB free GPU memory; observed ${free_mib:-unknown} MiB"
        sleep 15
    done
    echo "insufficient GPU memory after 10 minutes" >&2
    return 1
}

stage_run() {
    [[ -f "$RUN_MARKER" ]] && return 0
    [[ -f "$DATA_MARKER" ]] || { echo "data stage has not succeeded" >&2; exit 1; }
    mkdir -p "$BUILD_DIR/postprocessed_data" "$OUTPUT_DIR"
    {
        date -Ins
        wait_for_gpu_memory
        cd "$APP_DIR"
        LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 \
            /usr/bin/time -v "$ENV_DIR/bin/python" run.py \
            --backend=pytorch --scenario=Offline --accuracy \
            --model="$BUILD_DIR/model/3dunet_kits19_pytorch.ptc" \
            --preprocessed_data_dir="$BUILD_DIR/preprocessed_data" \
            --mlperf_conf="$BUILD_DIR/mlperf.conf" \
            --user_conf="$APP_DIR/user.conf"
    } >"$TARGET_DIR/run.log" 2>&1

    date -Ins >"$TARGET_DIR/evaluator.log"
    cd "$APP_DIR"
    /usr/bin/time -v "$ENV_DIR/bin/python" accuracy_kits.py \
        --preprocessed_data_dir="$BUILD_DIR/preprocessed_data" \
        --postprocessed_data_dir="$BUILD_DIR/postprocessed_data" \
        --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" --num_proc 4 \
        2>>"$TARGET_DIR/evaluator.log" \
        | tee "$OUTPUT_DIR/accuracy.txt" >>"$TARGET_DIR/evaluator.log"

    test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
    test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
    test -s "$OUTPUT_DIR/accuracy.txt"
    grep -q '^pytorch device: ' "$TARGET_DIR/run.log"
    grep -Eq '^Accuracy: mean = [^,]+, kidney = [^,]+, tumor = .+$' "$OUTPUT_DIR/accuracy.txt"
    touch "$RUN_MARKER"
}

run_selected() {
    case "$1" in
        environment) stage_environment ;;
        data) stage_data ;;
        run) stage_run ;;
    esac
}

if [[ -n "$SELECTED_STAGE" ]]; then
    run_selected "$SELECTED_STAGE"
elif [[ -n "$FROM_STAGE" ]]; then
    case "$FROM_STAGE" in
        environment) stage_environment; stage_data; stage_run ;;
        data) stage_data; stage_run ;;
        run) stage_run ;;
    esac
else
    stage_environment
    stage_data
    stage_run
fi

