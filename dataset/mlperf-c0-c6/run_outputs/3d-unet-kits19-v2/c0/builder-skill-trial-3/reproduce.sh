#!/usr/bin/env bash
set -euo pipefail

TARGET=""; MODE=all; STAGE=""; START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE=stage; STAGE="${2:-}"; shift 2 ;;
    --from) MODE=from; START_STAGE="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == 3d-unet ]] || { echo "supported target: 3d-unet" >&2; exit 2; }
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "OUTPUT_DIR must be set" >&2; exit 2; }

ROOT_DIR="$(pwd -P)"; TARGET_DIR="$ROOT_DIR/$TARGET"; VENV_DIR="$TARGET_DIR/.venv"
WORK_DIR="$TARGET_DIR/work"; WORK_REPO="$WORK_DIR/repo"
BENCH_DIR="$WORK_REPO/vision/medical_imaging/3d-unet-kits19"
PREPROCESSED_DIR="$WORK_DIR/preprocessed_data"; POSTPROCESSED_DIR="$WORK_DIR/postprocessed_data"
RAW_VIEW="$WORK_DIR/raw_data"
CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
KITS19_RAW_DIR="${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"; UV_EXCLUDE_NEWER=2026-11-10
export UV_CACHE_DIR UV_EXCLUDE_NEWER
mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"; OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

if [[ -d "$ROOT_DIR/repo/loadgen" && -d "$ROOT_DIR/repo/vision" ]]; then SOURCE_REPO="$ROOT_DIR/repo"
elif [[ -d "$ROOT_DIR/loadgen" && -d "$ROOT_DIR/vision" ]]; then SOURCE_REPO="$ROOT_DIR"
else echo "fresh repository not found at ./repo or current directory" >&2; exit 2; fi
for f in "$KITS19_RAW_DIR/case_00185/imaging.nii.gz" "$KITS19_RAW_DIR/case_00185/segmentation.nii.gz" "$MODEL_DIR/3dunet_kits19_pytorch.ptc"; do
  [[ -r "$f" ]] || { echo "missing input: $f" >&2; exit 2; }
done

stage_index() { case "$1" in environment) echo 0;; data) echo 1;; run) echo 2;; *) echo "invalid stage: $1" >&2; exit 2;; esac; }
clear_from() {
  local n; n="$(stage_index "$1")"
  if (( n == 0 )); then
    rm -rf "$VENV_DIR" "$WORK_DIR"
    rm -f "$TARGET_DIR"/{.environment.success,.data.success,.run.success,build.log,preprocess.log,run.log,evaluator.log,evaluator.stderr.log,gpu_memory.log,validation.log}
  elif (( n == 1 )); then
    rm -rf "$PREPROCESSED_DIR" "$POSTPROCESSED_DIR" "$RAW_VIEW"
    rm -f "$TARGET_DIR"/{.data.success,.run.success,preprocess.log,run.log,evaluator.log,evaluator.stderr.log,gpu_memory.log,validation.log}
  else
    rm -rf "$POSTPROCESSED_DIR"
    rm -f "$TARGET_DIR"/{.run.success,run.log,evaluator.log,evaluator.stderr.log,gpu_memory.log,validation.log}
  fi
  rm -f "$OUTPUT_DIR"/{mlperf_log_accuracy.json,mlperf_log_detail.txt,accuracy.txt}
}

environment_stage() {
  [[ ! -f "$TARGET_DIR/.environment.success" ]] || return 0
  mkdir -p "$WORK_DIR"; cp -a "$SOURCE_REPO/." "$WORK_REPO/"
  {
    echo "source tree: $SOURCE_REPO"; echo "UV_CACHE_DIR=$UV_CACHE_DIR"; echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv --python 3.11 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" "setuptools<76" wheel "pybind11==2.12.0" "numpy==1.26.4" "scipy==1.10.1" "nibabel==5.3.2" "pandas==1.5.3" "torch==2.4.1"
    uv pip install --python "$VENV_DIR/bin/python" --no-build-isolation "$WORK_REPO/loadgen"
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__, "nibabel", nibabel.__version__, "pandas", pandas.__version__)
print("loadgen import ok")
PY
  } >"$TARGET_DIR/build.log" 2>&1
  "$VENV_DIR/bin/python" - "$BENCH_DIR/pytorch_SUT.py" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text(); needle="        self.model.eval()\n"
if "pytorch device: " not in s:
    assert needle in s
    p.write_text(s.replace(needle, needle+'        print("pytorch device: {}".format(next(self.model.parameters()).device))\n', 1))
PY
  touch "$TARGET_DIR/.environment.success"
}

data_stage() {
  [[ ! -f "$TARGET_DIR/.data.success" ]] || return 0
  [[ -f "$TARGET_DIR/.environment.success" ]] || { echo "environment stage incomplete" >&2; exit 2; }
  mkdir -p "$RAW_VIEW" "$PREPROCESSED_DIR"; local manifest="$BENCH_DIR/meta/inference_cases.json"
  if [[ -n "${PREFLIGHT_CASE_COUNT:-}" ]]; then
    "$VENV_DIR/bin/python" - "$manifest" "$BENCH_DIR/meta/checksum_inference.json" "$PREFLIGHT_CASE_COUNT" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); cases=json.loads(p.read_text()); n=int(sys.argv[3]); assert 1 <= n <= len(cases)
cases=cases[:n]; p.write_text(json.dumps(cases, indent=2)+"\n")
q=pathlib.Path(sys.argv[2]); checks=json.loads(q.read_text()); q.write_text(json.dumps({c+".pkl":checks[c+".pkl"] for c in cases}, indent=4, sort_keys=True)+"\n")
PY
  fi
  while IFS= read -r c; do
    src="$c"; [[ "$c" != case_00400 ]] || src=case_00185
    [[ -d "$KITS19_RAW_DIR/$src" ]] || { echo "missing case: $src" >&2; exit 2; }
    ln -s "$KITS19_RAW_DIR/$src" "$RAW_VIEW/$c"
  done < <("$VENV_DIR/bin/python" - "$manifest" <<'PY'
import json,sys
print(*json.load(open(sys.argv[1])),sep="\n")
PY
  )
  (cd "$BENCH_DIR"; "$VENV_DIR/bin/python" preprocess.py --raw_data_dir "$RAW_VIEW" --results_dir "$PREPROCESSED_DIR" --mode preprocess --num_proc 4) >"$TARGET_DIR/preprocess.log" 2>&1
  "$VENV_DIR/bin/python" - "$PREPROCESSED_DIR/preprocessed_files.pkl" "$manifest" >>"$TARGET_DIR/preprocess.log" <<'PY'
import json,pickle,sys
with open(sys.argv[1],"rb") as f: got=pickle.load(f)["file_list"]
want=json.load(open(sys.argv[2])); assert got == want; print("validated preprocessed cases:",len(got))
PY
  touch "$TARGET_DIR/.data.success"
}

run_stage() {
  [[ ! -f "$TARGET_DIR/.run.success" ]] || return 0
  [[ -f "$TARGET_DIR/.data.success" ]] || { echo "data stage incomplete" >&2; exit 2; }
  mkdir -p "$POSTPROCESSED_DIR" "$OUTPUT_DIR"
  { date --iso-8601=seconds; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits; } >"$TARGET_DIR/gpu_memory.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -dc '0-9')"
  [[ -n "$free_mib" && "$free_mib" -ge 16384 ]] || { echo "insufficient free GPU memory: ${free_mib:-unknown} MiB; need 16384" | tee -a "$TARGET_DIR/gpu_memory.log" >&2; exit 1; }
  (cd "$BENCH_DIR"; LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 "$VENV_DIR/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy --model="$MODEL_DIR/3dunet_kits19_pytorch.ptc" --preprocessed_data_dir="$PREPROCESSED_DIR" --mlperf_conf="$WORK_REPO/loadgen/mlperf.conf" --user_conf="$BENCH_DIR/user.conf") >"$TARGET_DIR/run.log" 2>&1
  (cd "$BENCH_DIR"; "$VENV_DIR/bin/python" accuracy_kits.py --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" --preprocessed_data_dir "$PREPROCESSED_DIR" --postprocessed_data_dir "$POSTPROCESSED_DIR" --num_proc 4) >"$OUTPUT_DIR/accuracy.txt" 2>"$TARGET_DIR/evaluator.stderr.log"
  { cat "$OUTPUT_DIR/accuracy.txt"; cat "$TARGET_DIR/evaluator.stderr.log"; } >"$TARGET_DIR/evaluator.log"
  "$VENV_DIR/bin/python" - "$OUTPUT_DIR" "$TARGET_DIR/run.log" "$BENCH_DIR/meta/inference_cases.json" >"$TARGET_DIR/validation.log" <<'PY'
import json,pathlib,re,sys
out,run,manifest=map(pathlib.Path,sys.argv[1:])
for name in ("mlperf_log_accuracy.json","mlperf_log_detail.txt","accuracy.txt"):
 p=out/name; assert p.is_file() and p.stat().st_size > 0,p
records=json.loads((out/"mlperf_log_accuracy.json").read_text()); cases=json.loads(manifest.read_text()); assert len(records)==len(cases),(len(records),len(cases))
assert re.search(r"(?m)^pytorch device: cuda(?::0)?$",run.read_text()),"CUDA device log missing"
m=re.search(r"(?m)^Accuracy: mean = (\d+\.\d{5}), kidney = (\d+\.\d{4}), tumor = (\d+\.\d{4})$",(out/"accuracy.txt").read_text()); assert m,"result line missing"
print("validated samples:",len(records)); print("validated result:",m.group(0))
PY
  touch "$TARGET_DIR/.run.success"
}

run_one() { case "$1" in environment) environment_stage;; data) data_stage;; run) run_stage;; *) echo "invalid stage: $1" >&2; exit 2;; esac; }
if [[ "$MODE" == stage ]]; then run_one "$STAGE"
else [[ "$MODE" != from ]] || clear_from "$START_STAGE"; environment_stage; data_stage; run_stage; fi
