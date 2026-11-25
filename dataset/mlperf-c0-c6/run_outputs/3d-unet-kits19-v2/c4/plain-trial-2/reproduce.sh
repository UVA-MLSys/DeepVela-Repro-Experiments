#!/usr/bin/env bash
set -euo pipefail

target=""
while (( $# )); do
    case "$1" in
        --target)
            target="${2:-}"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "$target" != "3d-unet" ]]; then
    echo "unsupported target: $target" >&2
    exit 2
fi
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="$ROOT/repo"
TARGET_DIR="$ROOT/$target"
WORK_DIR="$TARGET_DIR/work"
SRC_DIR="$WORK_DIR/src"
RAW_DIR="$WORK_DIR/raw_data"
PREPROCESSED_DIR="$WORK_DIR/preprocessed_data"
POSTPROCESSED_DIR="$WORK_DIR/postprocessed_data"
LOADGEN_LOG_DIR="$WORK_DIR/loadgen_logs"
VENV="$TARGET_DIR/.venv"
PYTHON="$VENV/bin/python"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
KITS19_RAW_DIR="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR="$CACHE_ROOT/3d-unet-kits19/model"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
mkdir -p "$TARGET_DIR" "$WORK_DIR" "$RAW_DIR" "$PREPROCESSED_DIR" \
    "$POSTPROCESSED_DIR" "$LOADGEN_LOG_DIR"

{
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$PYTHON" \
        'torch==2.1.2' \
        'numpy==1.26.4' \
        'scipy==1.10.1' \
        'pandas==1.5.3' \
        'nibabel==5.3.2' \
        "$REPO/loadgen"
} 2>&1 | tee "$TARGET_DIR/build.log"

cp -a "$REPO/vision/medical_imaging/3d-unet-kits19" "$SRC_DIR"

patch -d "$SRC_DIR" -p0 <<'PATCH'
--- pytorch_SUT.py
+++ pytorch_SUT.py
@@ -76,6 +76,7 @@
             "cuda:0" if torch.cuda.is_available() else "cpu")
         self.model = torch.jit.load(model_path, map_location=self.device)
         self.model.eval()
+        print("pytorch device: {}".format(next(self.model.parameters()).device))
 
     def do_infer(self, input_tensor):
         """
--- accuracy_kits.py
+++ accuracy_kits.py
@@ -214,6 +214,7 @@
         )
 
     df.set_index("case", inplace=True)
+    df.loc["mean"] = df.mean(numeric_only=True)
 
     df.to_csv(sum_path)
 
PATCH

while IFS= read -r case_name; do
    source_name="$case_name"
    if [[ "$case_name" == case_00400 ]]; then
        source_name=case_00185
    fi
    ln -s "$KITS19_RAW_DIR/$source_name" "$RAW_DIR/$case_name"
done < <(sed -n 's/.*"\(case_[0-9]*\)".*/\1/p' "$SRC_DIR/meta/inference_cases.json")

(
    cd "$SRC_DIR"
    "$PYTHON" preprocess.py \
        --raw_data_dir "$RAW_DIR" \
        --results_dir "$PREPROCESSED_DIR" \
        --mode preprocess \
        --num_proc 8
) 2>&1 | tee "$TARGET_DIR/preprocess.log"

free_mib=0
for attempt in $(seq 1 120); do
    observation=$(nvidia-smi \
        --query-gpu=index,name,memory.total,memory.free,memory.used \
        --format=csv,noheader,nounits | head -n 1)
    printf '%s gpu memory MiB: %s\n' "$(date -Iseconds)" "$observation" \
        | tee -a "$TARGET_DIR/gpu.log"
    free_mib=$(printf '%s\n' "$observation" | awk -F, '{gsub(/ /,"",$4); print $4}')
    if (( free_mib >= 24576 )); then
        break
    fi
    sleep 5
done
if (( free_mib < 24576 )); then
    echo "insufficient free GPU memory after waiting: ${free_mib} MiB" \
        | tee -a "$TARGET_DIR/gpu.log" >&2
    exit 1
fi

(
    cd "$SRC_DIR"
    export CUDA_VISIBLE_DEVICES=0
    export LOG_PATH="$LOADGEN_LOG_DIR"
    export SKIP_VERIFY_ACCURACY=1
    "$PYTHON" run.py \
        --backend=pytorch \
        --scenario=Offline \
        --accuracy \
        --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
        --preprocessed_data_dir="$PREPROCESSED_DIR" \
        --mlperf_conf="$REPO/loadgen/mlperf.conf" \
        --user_conf="$SRC_DIR/user.conf"
) 2>&1 | tee "$TARGET_DIR/run.log"

cp "$LOADGEN_LOG_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
cp "$LOADGEN_LOG_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"

(
    cd "$SRC_DIR"
    "$PYTHON" accuracy_kits.py \
        --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
        --preprocessed_data_dir="$PREPROCESSED_DIR" \
        --postprocessed_data_dir="$POSTPROCESSED_DIR" \
        --num_proc 8 \
        2>"$TARGET_DIR/evaluator.stderr.log"
) | tee "$OUTPUT_DIR/accuracy.txt" "$TARGET_DIR/evaluator.log"
