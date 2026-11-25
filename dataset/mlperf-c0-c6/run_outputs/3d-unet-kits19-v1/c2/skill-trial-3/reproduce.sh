#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
KITS19_RAW_DIR=${KITS19_RAW_DIR:-$CACHE_ROOT/3d-unet-kits19/kits19_raw_data/kits19/data}
MODEL_DIR=${MODEL_DIR:-$CACHE_ROOT/3d-unet-kits19/model}
UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
UV_EXCLUDE_NEWER=2026-11-10
export UV_CACHE_DIR UV_EXCLUDE_NEWER

target=""
stage=""
from=""
preflight=0
while (($#)); do
  case "$1" in
    --target) target=${2:?missing target}; shift 2 ;;
    --stage) stage=${2:?missing stage}; shift 2 ;;
    --from) from=${2:?missing stage}; shift 2 ;;
    --preflight) preflight=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$target" == "3d-unet" ]] || { echo "--target 3d-unet is required" >&2; exit 2; }

OUT=${OUTPUT_DIR:-$SCRIPT_DIR/$target/out}
mkdir -p "$OUT"
if ((preflight)); then
  ROOT="$SCRIPT_DIR/$target/preflight"
else
  ROOT="$SCRIPT_DIR/$target/work"
fi
VENV="$ROOT/venv"
BENCH="$ROOT/benchmark"
PREP="$ROOT/preprocessed_data"
POST="$ROOT/postprocessed_data"
LOGS="$ROOT/loadgen_logs"
RAW="$ROOT/raw_data"
MODEL="$MODEL_DIR/3dunet_kits19_pytorch.ptc"
REPO="$SCRIPT_DIR/repo"

stages=(setup preprocess run)
stage_index() {
  case "$1" in setup) echo 0;; preprocess) echo 1;; run) echo 2;; *) return 1;; esac
}

if [[ -n "$from" ]]; then
  from_i=$(stage_index "$from") || { echo "invalid stage: $from" >&2; exit 2; }
  for ((i=from_i; i<${#stages[@]}; i++)); do
    rm -f "$ROOT/.${stages[$i]}.success"
  done
  if ((from_i <= 1)); then rm -rf "$PREP" "$POST" "$LOGS"; fi
  if ((from_i <= 2)); then
    rm -rf "$POST" "$LOGS"
    rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/accuracy.txt" "$OUT/run.log" "$OUT/evaluator.log"
  fi
fi

run_setup() {
  mkdir -p "$ROOT"
  exec > >(tee "$OUT/build.log") 2>&1
  echo "setup: creating Python environment"
  if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python "$(command -v python3.11)" "$VENV"
  fi
  uv pip install --python "$VENV/bin/python" --offline \
    'numpy<2' 'scipy<1.14' 'nibabel<6' 'pandas<2' 'torch' 'pybind11' 'setuptools<81'

  rm -rf "$BENCH"
  cp -a "$REPO/vision/medical_imaging/3d-unet-kits19" "$BENCH"
  cp "$REPO/loadgen/mlperf.conf" "$BENCH/mlperf.conf"

  # Restore the benchmark constant omitted from this source snapshot.
  sed -i 's/^TARGET_SPACING = None$/TARGET_SPACING = [1.6, 1.2, 1.2]/' "$BENCH/global_vars.py"
  sed -i 's/signal\.gaussian(/signal.windows.gaussian(/' "$BENCH/inference_utils.py"
  # This snapshot omits the published checksum files; retain all input presence/count checks.
  if [[ ! -f "$BENCH/meta/checksum_inference.json" ]]; then
    "$VENV/bin/python" - "$BENCH/preprocess.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = '    if args.mode == "preprocess":\n        preprocess_with_multiproc(args)\n        verify_dataset(args)\n'
new = '    if args.mode == "preprocess":\n        preprocess_with_multiproc(args)\n'
assert old in s
p.write_text(s.replace(old, new, 1))
PY
  fi
  # pandas <2 is selected, but concat keeps the evaluator usable if resolution changes.
  "$VENV/bin/python" - "$BENCH/accuracy_kits.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = s.replace(
'''        df = df.append(
            {"case": case, "kidney": kidney, "tumor": tumor, "composite": composite},
            ignore_index=True,
        )''',
'''        df = pd.concat([df, pd.DataFrame([{
            "case": case, "kidney": kidney, "tumor": tumor, "composite": composite
        }])], ignore_index=True)''')
p.write_text(s)
PY
  "$VENV/bin/python" - "$BENCH/pytorch_SUT.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = "        self.model.eval()\n"
new = old + '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n'
assert old in s
p.write_text(s.replace(old, new, 1))
PY

  uv pip install --python "$VENV/bin/python" --offline "$REPO/loadgen"
  "$VENV/bin/python" - <<'PY'
import torch, numpy, scipy, nibabel, pandas, mlperf_loadgen
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "nibabel", nibabel.__version__, "pandas", pandas.__version__)
PY
  [[ -s "$MODEL" ]]
  touch "$ROOT/.setup.success"
}

run_preprocess() {
  exec > >(tee "$OUT/preprocess.log") 2>&1
  echo "preprocess: validating and linking fixed inputs"
  rm -rf "$RAW" "$PREP"
  mkdir -p "$RAW" "$PREP"
  mapfile -t cases < <("$VENV/bin/python" - "$BENCH/meta/inference_cases.json" <<'PY'
import json, sys
print(*json.load(open(sys.argv[1])), sep="\n")
PY
  )
  if ((preflight)); then cases=("${cases[0]}"); fi
  for case_name in "${cases[@]}"; do
    source_name=$case_name
    [[ "$case_name" == case_00400 ]] && source_name=case_00185
    [[ -s "$KITS19_RAW_DIR/$source_name/imaging.nii.gz" && -s "$KITS19_RAW_DIR/$source_name/segmentation.nii.gz" ]]
    ln -s "$KITS19_RAW_DIR/$source_name" "$RAW/$case_name"
  done
  if ((preflight)); then
    "$VENV/bin/python" - "$BENCH/meta/inference_cases.json" "${cases[0]}" <<'PY'
import json, sys
json.dump([sys.argv[2]], open(sys.argv[1], "w"), indent=4)
PY
  fi
  (cd "$BENCH" && "$VENV/bin/python" preprocess.py --raw_data_dir "$RAW" --results_dir "$PREP" --mode preprocess --num_proc 4)
  expected=43; ((preflight)) && expected=1
  actual=$(find "$PREP" -maxdepth 1 -name 'case_*.pkl' -type f | wc -l)
  [[ "$actual" -eq "$expected" ]]
  touch "$ROOT/.preprocess.success"
}

run_inference() {
  exec > >(tee "$OUT/run.log") 2>&1
  echo "run: checking shared GPU immediately before launch"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader,nounits
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [[ "$free_mb" -ge 12000 ]] || { echo "insufficient free GPU memory: ${free_mb} MiB" >&2; exit 1; }
  rm -rf "$POST" "$LOGS"
  mkdir -p "$POST" "$LOGS"
  (cd "$BENCH" && LOG_PATH="$LOGS" SKIP_VERIFY_ACCURACY=1 "$VENV/bin/python" run.py \
    --backend=pytorch --scenario=Offline --accuracy --model="$MODEL" \
    --preprocessed_data_dir="$PREP" --mlperf_conf="$BENCH/mlperf.conf" --user_conf="$BENCH/user.conf")
  cp "$LOGS/mlperf_log_accuracy.json" "$OUT/mlperf_log_accuracy.json"
  cp "$LOGS/mlperf_log_detail.txt" "$OUT/mlperf_log_detail.txt"
  (cd "$BENCH" && "$VENV/bin/python" accuracy_kits.py \
    --preprocessed_data_dir="$PREP" --postprocessed_data_dir="$POST" \
    --log_file="$LOGS/mlperf_log_accuracy.json" --num_proc 4) \
    2>&1 | tee "$OUT/accuracy.txt" "$OUT/evaluator.log"
  grep -q '^pytorch device: ' "$OUT/run.log"
  grep -q '^Accuracy: mean = ' "$OUT/accuracy.txt"
  touch "$ROOT/.run.success"
}

run_one() {
  local s=$1
  [[ -f "$ROOT/.$s.success" ]] && { echo "$s already complete"; return; }
  case "$s" in setup) run_setup;; preprocess) run_preprocess;; run) run_inference;; esac
}

if [[ -n "$stage" ]]; then
  stage_index "$stage" >/dev/null || { echo "invalid stage: $stage" >&2; exit 2; }
  run_one "$stage"
else
  start=0
  if [[ -n "$from" ]]; then start=$(stage_index "$from"); fi
  for ((i=start; i<${#stages[@]}; i++)); do run_one "${stages[$i]}"; done
fi
