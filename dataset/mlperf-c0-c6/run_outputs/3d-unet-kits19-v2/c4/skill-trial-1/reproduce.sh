#!/usr/bin/env bash
set -euo pipefail

TARGET=""
ONLY_STAGE=""
FROM_STAGE=""
while (($#)); do
    case "$1" in
        --target) TARGET="${2:?missing target id}"; shift 2 ;;
        --stage) ONLY_STAGE="${2:?missing stage name}"; shift 2 ;;
        --from) FROM_STAGE="${2:?missing stage name}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "$TARGET" == "3d-unet" ]] || { echo "supported target: 3d-unet" >&2; exit 2; }
[[ -z "$ONLY_STAGE" || -z "$FROM_STAGE" ]] || { echo "--stage and --from are mutually exclusive" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR:-$SCRIPT_DIR/$TARGET/out}")"
TARGET_DIR="$(dirname "$OUTPUT_DIR")"
WORK_DIR="$TARGET_DIR/work"
BENCH_DIR="$WORK_DIR/benchmark"
VENV_DIR="$WORK_DIR/venv"
MARKER_DIR="$WORK_DIR/markers"

CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
KITS19_RAW_DIR="${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

[[ "$OUTPUT_DIR" != "/" && "$WORK_DIR" != "/" ]] || { echo "refusing broad output/work directory" >&2; exit 2; }
[[ -d "$REPO_DIR" ]] || { echo "missing repository: $REPO_DIR" >&2; exit 2; }
[[ -d "$KITS19_RAW_DIR" ]] || { echo "missing raw data: $KITS19_RAW_DIR" >&2; exit 2; }
[[ -f "$MODEL_DIR/3dunet_kits19_pytorch.ptc" ]] || { echo "missing model in $MODEL_DIR" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR" "$WORK_DIR" "$MARKER_DIR"

STAGES=(build preprocess infer evaluate)
valid_stage() {
    local wanted="$1" stage
    for stage in "${STAGES[@]}"; do [[ "$stage" == "$wanted" ]] && return 0; done
    return 1
}
[[ -z "$ONLY_STAGE" ]] || valid_stage "$ONLY_STAGE" || { echo "invalid stage: $ONLY_STAGE" >&2; exit 2; }
[[ -z "$FROM_STAGE" ]] || valid_stage "$FROM_STAGE" || { echo "invalid stage: $FROM_STAGE" >&2; exit 2; }

invalidate_from() {
    local wanted="$1" active=0 stage
    for stage in "${STAGES[@]}"; do
        [[ "$stage" == "$wanted" ]] && active=1
        ((active)) && rm -f "$MARKER_DIR/$stage.done"
    done
    case "$wanted" in
        build)
            rm -rf "$BENCH_DIR" "$VENV_DIR" "$WORK_DIR/raw_data"
            rm -f "$OUTPUT_DIR/build.log"
            ;&
        preprocess)
            rm -rf "$BENCH_DIR/build/preprocessed_data"
            rm -f "$OUTPUT_DIR/preprocess.log"
            ;&
        infer)
            rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
                "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" \
                "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/gpu-memory.log"
            ;&
        evaluate)
            rm -rf "$BENCH_DIR/build/postprocessed_data"
            rm -f "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
            ;;
    esac
}
[[ -z "$FROM_STAGE" ]] || invalidate_from "$FROM_STAGE"

stage_build() {
    [[ ! -e "$MARKER_DIR/build.done" ]] || return 0
    : >"$OUTPUT_DIR/build.log"
    cp -a "$REPO_DIR/vision/medical_imaging/3d-unet-kits19" "$BENCH_DIR"
    cp "$REPO_DIR/mlperf.conf" "$BENCH_DIR/mlperf.conf"
    sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$BENCH_DIR/pytorch_SUT.py"
    sed -i '/df.set_index("case", inplace=True)/a\    df.loc["mean"] = df.mean(numeric_only=True)' "$BENCH_DIR/accuracy_kits.py"
    sed -i 's/^    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)$/    if not args.accuracy:\n        settings.FromConfig(args.user_conf, "3d-unet", args.scenario)/' "$BENCH_DIR/run.py"

    uv venv --python /usr/bin/python3.11 "$VENV_DIR" >>"$OUTPUT_DIR/build.log" 2>&1
    uv pip install --python "$VENV_DIR/bin/python" --offline \
        'numpy==1.26.4' 'scipy==1.11.4' 'pandas==1.5.3' 'nibabel==5.2.1' \
        'torch==2.15.0' 'mlcommons-loadgen==6.0.17' >>"$OUTPUT_DIR/build.log" 2>&1

    mkdir -p "$BENCH_DIR/build/model"
    ln -s "$MODEL_DIR/3dunet_kits19_pytorch.ptc" "$BENCH_DIR/build/model/3dunet_kits19_pytorch.ptc"
    cp "$REPO_DIR/mlperf.conf" "$BENCH_DIR/build/mlperf.conf"

    if [[ -n "${PREFLIGHT_CASE:-}" ]]; then
        (
            cd "$BENCH_DIR"
            "$VENV_DIR/bin/python" - "$PREFLIGHT_CASE" <<'PY'
import json
import pathlib
import sys

case = sys.argv[1]
meta = pathlib.Path("meta")
checksums = json.loads((meta / "checksum_inference.json").read_text())
key = case + ".pkl"
if key not in checksums:
    raise SystemExit(f"no published checksum for {case}")
(meta / "inference_cases.json").write_text(json.dumps([case], indent=4) + "\n")
(meta / "checksum_inference.json").write_text(json.dumps({key: checksums[key]}, indent=4) + "\n")
PY
        ) >>"$OUTPUT_DIR/build.log" 2>&1
        sed -i 's/^3d-unet\.Offline\.min_query_count = 43$/3d-unet.Offline.min_query_count = 1/' "$BENCH_DIR/build/mlperf.conf"
    fi
    touch "$MARKER_DIR/build.done"
}

stage_preprocess() {
    [[ ! -e "$MARKER_DIR/preprocess.done" ]] || return 0
    [[ -e "$MARKER_DIR/build.done" ]] || { echo "build stage has not succeeded" >&2; exit 2; }
    rm -rf "$WORK_DIR/raw_data" "$BENCH_DIR/build/preprocessed_data"
    mkdir -p "$WORK_DIR/raw_data"
    ln -s "$KITS19_RAW_DIR"/case_* "$WORK_DIR/raw_data/"
    if [[ ! -e "$WORK_DIR/raw_data/case_00400" ]]; then
        ln -s "$KITS19_RAW_DIR/case_00185" "$WORK_DIR/raw_data/case_00400"
    fi
    (
        cd "$BENCH_DIR"
        /usr/bin/time -v "$VENV_DIR/bin/python" preprocess.py \
            --raw_data_dir "$WORK_DIR/raw_data" --results_dir build/preprocessed_data \
            --mode preprocess --num_proc "${PREPROCESS_PROCS:-8}"
        /usr/bin/time -v "$VENV_DIR/bin/python" preprocess.py \
            --raw_data_dir "$WORK_DIR/raw_data" --results_dir build/preprocessed_data \
            --mode verify --num_proc "${PREPROCESS_PROCS:-8}"
    ) >"$OUTPUT_DIR/preprocess.log" 2>&1
    touch "$MARKER_DIR/preprocess.done"
}

stage_infer() {
    [[ ! -e "$MARKER_DIR/infer.done" ]] || return 0
    [[ -e "$MARKER_DIR/preprocess.done" ]] || { echo "preprocess stage has not succeeded" >&2; exit 2; }
    rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader \
        >"$OUTPUT_DIR/gpu-memory.log"
    (
        cd "$BENCH_DIR"
        export PATH="$VENV_DIR/bin:$PATH"
        export LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1
        /usr/bin/time -v python3 run.py --backend=pytorch --scenario=Offline --accuracy
    ) >"$OUTPUT_DIR/run.log" 2>&1
    [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" && -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
    touch "$MARKER_DIR/infer.done"
}

stage_evaluate() {
    [[ ! -e "$MARKER_DIR/evaluate.done" ]] || return 0
    [[ -e "$MARKER_DIR/infer.done" ]] || { echo "infer stage has not succeeded" >&2; exit 2; }
    rm -rf "$BENCH_DIR/build/postprocessed_data"
    mkdir -p "$BENCH_DIR/build/postprocessed_data"
    (
        cd "$BENCH_DIR"
        /usr/bin/time -v "$VENV_DIR/bin/python" accuracy_kits.py \
            --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
            --preprocessed_data_dir build/preprocessed_data \
            --postprocessed_data_dir build/postprocessed_data \
            --num_proc "${EVALUATOR_PROCS:-4}"
    ) >"$OUTPUT_DIR/accuracy.txt" 2>"$OUTPUT_DIR/evaluator.log"
    rg -q '^Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
    touch "$MARKER_DIR/evaluate.done"
}

run_stage() {
    case "$1" in
        build) stage_build ;;
        preprocess) stage_preprocess ;;
        infer) stage_infer ;;
        evaluate) stage_evaluate ;;
    esac
}

if [[ -n "$ONLY_STAGE" ]]; then
    run_stage "$ONLY_STAGE"
elif [[ -n "$FROM_STAGE" ]]; then
    active=0
    for stage in "${STAGES[@]}"; do
        [[ "$stage" == "$FROM_STAGE" ]] && active=1
        ((active)) && run_stage "$stage"
    done
else
    for stage in "${STAGES[@]}"; do run_stage "$stage"; done
fi
