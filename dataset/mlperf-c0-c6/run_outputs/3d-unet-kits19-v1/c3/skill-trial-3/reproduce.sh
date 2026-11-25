#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet [--stage build|preprocess|run] [--from build|preprocess|run]" >&2
  exit 2
}

TARGET=""
STAGE=""
FROM=""
while (($#)); do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; TARGET="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; STAGE="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; FROM="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$TARGET" == "3d-unet" ]] || usage
[[ -z "$STAGE" || -z "$FROM" ]] || usage
case "${STAGE:-${FROM:-build}}" in build|preprocess|run) ;; *) usage ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
BENCH_SRC="$REPO_DIR/vision/medical_imaging/3d-unet-kits19"
CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
RAW_INPUT="$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_INPUT="$CACHE_ROOT/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
export UV_OFFLINE=1

PYTHON_BIN=""
for candidate in "${PYTHON311:-}" /home/mi3se/.local/bin/python3.11 /usr/bin/python3.11; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" -c 'import os, sysconfig; raise SystemExit(not os.path.isfile(os.path.join(sysconfig.get_path("include"), "Python.h")))'; then
    PYTHON_BIN="$candidate"
    break
  fi
done
[[ -n "$PYTHON_BIN" ]] || { echo "no Python 3.11 interpreter with development headers found" >&2; exit 1; }

: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
TARGET_DIR="$(dirname "$OUTPUT_DIR")"
WORK_DIR="$TARGET_DIR/work"
BENCH_DIR="$WORK_DIR/3d-unet-kits19"
VENV_DIR="$TARGET_DIR/venv"
MARKER_DIR="$TARGET_DIR/markers"
PREFLIGHT="${REPRO_PREFLIGHT:-0}"

[[ -d "$REPO_DIR/loadgen" && -d "$BENCH_SRC" ]] || { echo "fresh repository not found beside reproduce.sh" >&2; exit 1; }
[[ -d "$RAW_INPUT" && -s "$MODEL_INPUT" ]] || { echo "required cached inputs are missing" >&2; exit 1; }

clear_from() {
  local first="$1"
  mkdir -p "$MARKER_DIR"
  case "$first" in
    build)
      rm -rf -- "$WORK_DIR" "$VENV_DIR"
      rm -f -- "$MARKER_DIR/build.ok" "$MARKER_DIR/preprocess.ok" "$MARKER_DIR/run.ok"
      find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
      ;;
    preprocess)
      rm -rf -- "$BENCH_DIR/build/preprocessed_data" "$BENCH_DIR/build/postprocessed_data"
      rm -f -- "$MARKER_DIR/preprocess.ok" "$MARKER_DIR/run.ok"
      find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
      ;;
    run)
      rm -rf -- "$BENCH_DIR/build/postprocessed_data"
      rm -f -- "$MARKER_DIR/run.ok"
      find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
      ;;
  esac
}

stage_build() {
  [[ ! -e "$MARKER_DIR/build.ok" ]] || return 0
  mkdir -p "$WORK_DIR" "$MARKER_DIR" "$OUTPUT_DIR"
  cp -a "$BENCH_SRC" "$BENCH_DIR"
  cp -a "$REPO_DIR/loadgen" "$WORK_DIR/loadgen"
  mkdir -p "$BENCH_DIR/build/model" "$BENCH_DIR/build/raw_data" "$BENCH_DIR/build/logs"
  cp "$REPO_DIR/loadgen/mlperf.conf" "$BENCH_DIR/build/mlperf.conf"
  ln -s "$MODEL_INPUT" "$BENCH_DIR/build/model/3dunet_kits19_pytorch.ptc"

  while IFS= read -r case_name; do
    source_name="$case_name"
    [[ "$case_name" != "case_00400" ]] || source_name="case_00185"
    ln -s "$RAW_INPUT/$source_name" "$BENCH_DIR/build/raw_data/$case_name"
  done < <("$PYTHON_BIN" -c 'import json; print(*json.load(open("'"$BENCH_DIR"'/meta/inference_cases.json")), sep="\n")')

  if [[ "$PREFLIGHT" == "1" ]]; then
    "$PYTHON_BIN" - "$BENCH_DIR" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "meta"
case = "case_00000"
(p / "inference_cases.json").write_text(json.dumps([case], indent=4) + "\n")
checks = json.loads((p / "checksum_inference.json").read_text())
(p / "checksum_inference.json").write_text(json.dumps({case + ".pkl": checks[case + ".pkl"]}, indent=4) + "\n")
PY
  fi

  "$PYTHON_BIN" - "$BENCH_DIR/pytorch_SUT.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
needle = "        self.model.eval()\n"
assert needle in s
s = s.replace(needle, needle + '        print(f"pytorch device: {next(self.model.parameters()).device}")\n', 1)
p.write_text(s)
PY
  "$PYTHON_BIN" - "$BENCH_DIR/inference_utils.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
needle = "signal.gaussian(n, std)"
assert needle in s
p.write_text(s.replace(needle, "signal.windows.gaussian(n, std)"))
PY
  "$PYTHON_BIN" - "$BENCH_DIR/accuracy_kits.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = '''        df = df.append(
            {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite},
            ignore_index=True,
        )'''
new = '''        df = pd.concat(
            [df, pd.DataFrame([{"case": case, "kidney": kidney, "tumor": tumor, "composite": composite}])],
            ignore_index=True,
        )'''
assert old in s
s = s.replace(old, new)
s = s.replace('float(final["composite"])', 'float(final["composite"].iloc[0])')
s = s.replace('float(final["kidney"])', 'float(final["kidney"].iloc[0])')
s = s.replace('float(final["tumor"])', 'float(final["tumor"].iloc[0])')
p.write_text(s)
PY

  {
    echo "Building cached Python environment"
    /usr/bin/time -v uv venv --python "$PYTHON_BIN" "$VENV_DIR"
    /usr/bin/time -v uv pip install --python "$VENV_DIR/bin/python" \
      'torch==2.7.1' 'numpy<2' scipy nibabel pandas 'pybind11==2.11.1' setuptools wheel
    /usr/bin/time -v uv pip install --python "$VENV_DIR/bin/python" --no-build-isolation "$WORK_DIR/loadgen"
    "$VENV_DIR/bin/python" -c 'import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch; print("torch", torch.__version__, "cuda", torch.version.cuda); print("packages import: ok")'
  } 2>&1 | tee "$OUTPUT_DIR/build.log"
  test "${PIPESTATUS[0]}" -eq 0
  touch "$MARKER_DIR/build.ok"
}

stage_preprocess() {
  [[ -e "$MARKER_DIR/build.ok" ]] || { echo "build stage has not succeeded" >&2; exit 1; }
  [[ ! -e "$MARKER_DIR/preprocess.ok" ]] || return 0
  mkdir -p "$BENCH_DIR/build/preprocessed_data"
  (
    cd "$BENCH_DIR"
    /usr/bin/time -v "$VENV_DIR/bin/python" preprocess.py \
      --raw_data_dir build/raw_data --results_dir build/preprocessed_data \
      --mode preprocess --num_proc "${PREPROCESS_PROCS:-4}"
  ) 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  test "${PIPESTATUS[0]}" -eq 0
  touch "$MARKER_DIR/preprocess.ok"
}

stage_run() {
  [[ -e "$MARKER_DIR/preprocess.ok" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  [[ ! -e "$MARKER_DIR/run.ok" ]] || return 0
  mkdir -p "$OUTPUT_DIR" "$BENCH_DIR/build/postprocessed_data"
  local free_mb
  free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader | tee "$OUTPUT_DIR/gpu-before.log"
  [[ "$free_mb" =~ ^[0-9]+$ && "$free_mb" -ge 16000 ]] || { echo "less than 16 GiB GPU memory is free; refusing GPU run" | tee -a "$OUTPUT_DIR/gpu-before.log"; exit 1; }
  (
    cd "$BENCH_DIR"
    export LOG_PATH="$OUTPUT_DIR"
    export SKIP_VERIFY_ACCURACY=1
    /usr/bin/time -v "$VENV_DIR/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model=build/model/3dunet_kits19_pytorch.ptc \
      --preprocessed_data_dir=build/preprocessed_data \
      --mlperf_conf=build/mlperf.conf --user_conf=user.conf
  ) 2>&1 | tee "$OUTPUT_DIR/run.log"
  test "${PIPESTATUS[0]}" -eq 0
  (
    cd "$BENCH_DIR"
    /usr/bin/time -v "$VENV_DIR/bin/python" accuracy_kits.py \
      --preprocessed_data_dir=build/preprocessed_data \
      --postprocessed_data_dir=build/postprocessed_data \
      --log_file="$OUTPUT_DIR/mlperf_log_accuracy.json"
  ) 2>&1 | tee "$OUTPUT_DIR/accuracy.txt"
  test "${PIPESTATUS[0]}" -eq 0
  cp "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  grep -F 'pytorch device: cuda:0' "$OUTPUT_DIR/run.log"
  grep -F 'Accuracy: mean = ' "$OUTPUT_DIR/accuracy.txt"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  touch "$MARKER_DIR/run.ok"
}

if [[ -n "$FROM" ]]; then
  clear_from "$FROM"
fi

if [[ -n "$STAGE" ]]; then
  "stage_$STAGE"
else
  start="${FROM:-build}"
  case "$start" in
    build) stage_build; stage_preprocess; stage_run ;;
    preprocess) stage_preprocess; stage_run ;;
    run) stage_run ;;
  esac
fi
