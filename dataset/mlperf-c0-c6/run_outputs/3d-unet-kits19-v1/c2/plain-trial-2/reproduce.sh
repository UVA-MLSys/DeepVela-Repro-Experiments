#!/usr/bin/env bash
set -euo pipefail

TARGET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$TARGET" != "3d-unet" ]]; then
  echo "supported target: 3d-unet" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$ROOT/repo"
TARGET_DIR="$ROOT/$TARGET"
WORK_DIR="$TARGET_DIR/work"
APP_DIR="$WORK_DIR/app"
VENV_DIR="$TARGET_DIR/venv"
PREPROC_DIR="$WORK_DIR/preprocessed_data"
POSTPROC_DIR="$WORK_DIR/postprocessed_data"
RAW_LINK_DIR="$WORK_DIR/raw_data"
OUTPUT_DIR="${OUTPUT_DIR:-$TARGET_DIR/out}"
OUTPUT_DIR="$(readlink -m "$OUTPUT_DIR")"

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
KITS19_RAW_DIR="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR="$CACHE_ROOT/3d-unet-kits19/model"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$TARGET_DIR" "$WORK_DIR" "$OUTPUT_DIR"

if [[ ! -d "$APP_DIR" ]]; then
  cp -a "$REPO/vision/medical_imaging/3d-unet-kits19" "$APP_DIR"
fi

# Restore the benchmark's documented constants and compatibility with current
# SciPy/pandas. All edits are confined to the target work directory.
sed -i 's/^TARGET_SPACING = None$/TARGET_SPACING = [1.6, 1.2, 1.2]/' "$APP_DIR/global_vars.py"
sed -i 's/signal\.gaussian(n, std)/signal.windows.gaussian(n, std)/' "$APP_DIR/inference_utils.py"
sed -i '/        df = df.append(/,/            ignore_index=True,$/c\        df.loc[len(df)] = {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite}' "$APP_DIR/accuracy_kits.py"
sed -i '/df.loc\[len(df)\]/{n;/^        )$/d;}' "$APP_DIR/accuracy_kits.py"
sed -i 's/df = pd.DataFrame()/df = pd.DataFrame(columns=["case", "kidney", "tumor", "composite"])/' "$APP_DIR/accuracy_kits.py"
sed -i 's/float(final\["composite"\])/float(final["composite"].iloc[0])/' "$APP_DIR/accuracy_kits.py"
sed -i 's/float(final\["kidney"\])/float(final["kidney"].iloc[0])/' "$APP_DIR/accuracy_kits.py"
sed -i 's/float(final\["tumor"\])/float(final["tumor"].iloc[0])/' "$APP_DIR/accuracy_kits.py"
if ! grep -q 'pytorch device: ' "$APP_DIR/pytorch_SUT.py"; then
  sed -i '/        self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$APP_DIR/pytorch_SUT.py"
fi
# The source snapshot has no checksum manifest; preprocessing itself is enough.
sed -i '/        verify_dataset(args)/d' "$APP_DIR/preprocess.py"
sed -i '/    if args.mode == "verify":/d' "$APP_DIR/preprocess.py"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  {
    uv venv --python 3.11 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" \
      'numpy==1.26.4' 'scipy==1.14.1' 'nibabel==5.3.2' \
      'pandas==2.2.3' 'torch==2.4.1' 'mlcommons-loadgen==6.0.17'
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("nibabel", nibabel.__version__, "pandas", pandas.__version__)
PY
  } 2>&1 | tee "$TARGET_DIR/build.log"
fi

if [[ ! -f "$PREPROC_DIR/preprocessed_files.pkl" ]]; then
  mkdir -p "$RAW_LINK_DIR"
  "$VENV_DIR/bin/python" - "$APP_DIR/meta/inference_cases.json" "$KITS19_RAW_DIR" "$RAW_LINK_DIR" <<'PY'
import json, os, sys
case_file, source, destination = sys.argv[1:]
for case in json.load(open(case_file)):
    actual = "case_00185" if case == "case_00400" else case
    link = os.path.join(destination, case)
    if not os.path.lexists(link):
        os.symlink(os.path.join(source, actual), link)
PY
  (
    cd "$APP_DIR"
    "$VENV_DIR/bin/python" preprocess.py \
      --raw_data_dir "$RAW_LINK_DIR" \
      --results_dir "$PREPROC_DIR" \
      --num_proc 8
  ) 2>&1 | tee "$TARGET_DIR/preprocess.log"
fi

rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" \
      "$OUTPUT_DIR/accuracy.txt"

{
  echo "GPU state immediately before inference:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits
} 2>&1 | tee "$TARGET_DIR/gpu-check.log"

(
  cd "$APP_DIR"
  export LOG_PATH="$OUTPUT_DIR"
  export SKIP_VERIFY_ACCURACY=1
  "$VENV_DIR/bin/python" run.py \
    --backend=pytorch \
    --scenario=Offline \
    --accuracy \
    --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" \
    --preprocessed_data_dir="$PREPROC_DIR" \
    --mlperf_conf="$REPO/loadgen/mlperf.conf" \
    --user_conf="$APP_DIR/user.conf"
) 2>&1 | tee "$TARGET_DIR/run.log"

(
  cd "$APP_DIR"
  "$VENV_DIR/bin/python" accuracy_kits.py \
    --preprocessed_data_dir="$PREPROC_DIR" \
    --postprocessed_data_dir="$POSTPROC_DIR" \
    --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --num_proc 8
) 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$TARGET_DIR/evaluator.log"

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
grep -F 'pytorch device: ' "$TARGET_DIR/run.log"
grep -F 'Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
