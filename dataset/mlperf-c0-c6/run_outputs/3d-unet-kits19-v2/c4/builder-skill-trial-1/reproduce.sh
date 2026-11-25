#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet [--stage environment|preprocess|run|evaluate] [--from environment|preprocess|run|evaluate]" >&2
  exit 2
}

target=""
stage_only=""
from_stage=""
while (($#)); do
  case "$1" in
    --target) (($# >= 2)) || usage; target="$2"; shift 2 ;;
    --stage) (($# >= 2)) || usage; stage_only="$2"; shift 2 ;;
    --from) (($# >= 2)) || usage; from_stage="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$target" == "3d-unet" ]] || usage
[[ -z "$stage_only" || -z "$from_stage" ]] || usage

stages=(environment preprocess run evaluate)
valid_stage() {
  local wanted="$1" item
  for item in "${stages[@]}"; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}
[[ -z "$stage_only" ]] || valid_stage "$stage_only" || usage
[[ -z "$from_stage" ]] || valid_stage "$from_stage" || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$script_dir/repo/vision/medical_imaging/3d-unet-kits19" ]]; then
  repo_root="$script_dir/repo"
elif [[ -d "$script_dir/vision/medical_imaging/3d-unet-kits19" ]]; then
  repo_root="$script_dir"
else
  echo "fresh repository not found beside reproduce.sh" >&2
  exit 1
fi

cache_root="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
raw_dir="${KITS19_RAW_DIR:-$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data}"
model_dir="${MODEL_DIR:-$cache_root/3d-unet-kits19/model}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache_root/uv}"
export UV_EXCLUDE_NEWER=2026-11-10
managed_python="$(uv python find 3.11 --managed-python)"

run_label="${RUN_LABEL:-full}"
[[ "$run_label" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid RUN_LABEL" >&2; exit 2; }
target_dir="$script_dir/$target"
work_dir="$target_dir/work-$run_label"
source_dir="$work_dir/source"
venv_dir="$work_dir/.venv"
preprocessed_dir="$work_dir/preprocessed_data"
raw_stage_dir="$work_dir/raw_data"
postprocessed_dir="$work_dir/postprocessed_data"
log_dir="${OUTPUT_DIR:-$target_dir/out}"
mkdir -p "$target_dir" "$log_dir"
log_dir="$(cd "$log_dir" && pwd)"

for required in "$raw_dir" "$model_dir/3dunet_kits19_pytorch.ptc" "$UV_CACHE_DIR"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

stage_index() {
  case "$1" in environment) echo 0;; preprocess) echo 1;; run) echo 2;; evaluate) echo 3;; *) return 1;; esac
}

if [[ -n "$from_stage" ]]; then
  start_idx="$(stage_index "$from_stage")"
  for ((i=start_idx; i<${#stages[@]}; i++)); do
    rm -f "$work_dir/.${stages[$i]}.done"
  done
else
  start_idx=0
fi

should_run() {
  local current="$1" idx
  idx="$(stage_index "$current")"
  if [[ -n "$stage_only" ]]; then [[ "$current" == "$stage_only" ]]; return; fi
  ((idx >= start_idx))
}

environment_stage() {
  if [[ "$work_dir" != "$target_dir/work-$run_label" ]]; then
    echo "refusing unsafe work directory: $work_dir" >&2
    exit 1
  fi
  rm -rf "$work_dir"
  mkdir -p "$work_dir" "$source_dir"
  {
    echo "stage=environment started=$(date -Is)"
    echo "repo_root=$repo_root"
    echo "python=$($managed_python --version 2>&1) path=$managed_python"
    echo "uv=$(uv --version)"
    cp -a "$repo_root/." "$source_dir/"
    "$managed_python" - "$source_dir" "${PREFLIGHT_CASES:-}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
subset = [x for x in sys.argv[2].split(",") if x]
app = root / "vision/medical_imaging/3d-unet-kits19"

p = app / "pytorch_SUT.py"
s = p.read_text()
old = "        self.model = torch.jit.load(model_path, map_location=self.device)\n        self.model.eval()"
new = "        self.model = torch.jit.load(model_path, map_location=self.device)\n        self.model.eval()\n        print(f\"pytorch device: {self.device}\")"
if old not in s:
    raise SystemExit("device logging patch context not found")
p.write_text(s.replace(old, new, 1))

p = app / "accuracy_kits.py"
s = p.read_text()
old = '    df.set_index("case", inplace=True)\n\n    df.to_csv(sum_path)'
new = '    df.set_index("case", inplace=True)\n    df.loc["mean"] = df.mean(numeric_only=True)\n\n    df.to_csv(sum_path)'
if old not in s:
    raise SystemExit("metric aggregation patch context not found")
p.write_text(s.replace(old, new, 1))

if subset:
    manifest = app / "meta/inference_cases.json"
    canonical = json.loads(manifest.read_text())
    unknown = sorted(set(subset) - set(canonical))
    if unknown:
        raise SystemExit(f"unknown PREFLIGHT_CASES: {unknown}")
    manifest.write_text(json.dumps(subset, indent=4) + "\n")
    checksums = app / "meta/checksum_inference.json"
    all_hashes = json.loads(checksums.read_text())
    reduced = {f"{case}.pkl": all_hashes[f"{case}.pkl"] for case in subset}
    checksums.write_text(json.dumps(reduced, indent=4, sort_keys=True) + "\n")
PY
    uv venv "$venv_dir" --python "$managed_python" --seed
    uv pip install --python "$venv_dir/bin/python" --offline \
      "torch==2.5.1" "numpy==1.26.4" "scipy==1.10.1" \
      "nibabel==5.2.1" "pandas==1.5.3" "pybind11==2.11.1" \
      "setuptools==75.3.0" "wheel==0.45.1"
    uv pip install --python "$venv_dir/bin/python" --offline --no-build-isolation "$source_dir/loadgen"
    "$venv_dir/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "nibabel", nibabel.__version__)
assert torch.cuda.is_available(), "CUDA is required by the benchmark contract"
PY
    echo "stage=environment finished=$(date -Is)"
  } 2>&1 | tee "$log_dir/build.log"
  touch "$work_dir/.environment.done"
}

preprocess_stage() {
  [[ -f "$work_dir/.environment.done" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
  rm -rf "$raw_stage_dir" "$preprocessed_dir"
  mkdir -p "$raw_stage_dir" "$preprocessed_dir"
  {
    echo "stage=preprocess started=$(date -Is)"
    "$venv_dir/bin/python" - "$source_dir" "$raw_dir" "$raw_stage_dir" <<'PY'
import json
import os
import sys
from pathlib import Path

root, raw, stage = map(Path, sys.argv[1:])
manifest = root / "vision/medical_imaging/3d-unet-kits19/meta/inference_cases.json"
cases = json.loads(manifest.read_text())
for case in cases:
    source_case = "case_00185" if case == "case_00400" else case
    src = raw / source_case
    if not (src / "imaging.nii.gz").is_file() or not (src / "segmentation.nii.gz").is_file():
        raise SystemExit(f"incomplete raw case: {src}")
    os.symlink(src, stage / case, target_is_directory=True)
print(f"staged {len(cases)} cases")
PY
    cd "$source_dir/vision/medical_imaging/3d-unet-kits19"
    "$venv_dir/bin/python" preprocess.py \
      --raw_data_dir "$raw_stage_dir" \
      --results_dir "$preprocessed_dir" \
      --mode preprocess --num_proc 8
    echo "stage=preprocess finished=$(date -Is)"
  } 2>&1 | tee "$log_dir/preprocess.log"
  touch "$work_dir/.preprocess.done"
}

run_stage() {
  [[ -f "$work_dir/.preprocess.done" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  rm -rf "$postprocessed_dir"
  mkdir -p "$postprocessed_dir"
  rm -f "$log_dir/mlperf_log_accuracy.json" "$log_dir/mlperf_log_detail.txt" \
    "$log_dir/mlperf_log_summary.txt" "$log_dir/mlperf_log_trace.json"
  {
    echo "stage=run gpu_check=$(date -Is)"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used \
      --format=csv,noheader,nounits
  } 2>&1 | tee "$log_dir/gpu_memory.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  ((free_mib >= 20000)) || { echo "insufficient free GPU memory: ${free_mib} MiB" | tee -a "$log_dir/gpu_memory.log"; exit 1; }
  {
    echo "stage=run started=$(date -Is) free_mib=$free_mib"
    cd "$source_dir/vision/medical_imaging/3d-unet-kits19"
    LOG_PATH="$log_dir" SKIP_VERIFY_ACCURACY=1 "$venv_dir/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$model_dir/3dunet_kits19_pytorch.ptc" \
      --preprocessed_data_dir="$preprocessed_dir" \
      --mlperf_conf="$source_dir/loadgen/mlperf.conf" \
      --user_conf=user.conf
    echo "stage=run finished=$(date -Is)"
  } 2>&1 | tee "$log_dir/run.log"
  grep -q '^pytorch device: cuda' "$log_dir/run.log"
  [[ -s "$log_dir/mlperf_log_accuracy.json" && -s "$log_dir/mlperf_log_detail.txt" ]]
  touch "$work_dir/.run.done"
}

evaluate_stage() {
  [[ -f "$work_dir/.run.done" ]] || { echo "run stage has not succeeded" >&2; exit 1; }
  rm -rf "$postprocessed_dir"
  mkdir -p "$postprocessed_dir"
  : > "$log_dir/evaluator.log"
  echo "stage=evaluate started=$(date -Is)" | tee -a "$log_dir/evaluator.log"
  (
    cd "$source_dir/vision/medical_imaging/3d-unet-kits19"
    "$venv_dir/bin/python" accuracy_kits.py \
      --log_file="$log_dir/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="$preprocessed_dir" \
      --postprocessed_data_dir="$postprocessed_dir" \
      --num_proc=8 2> >(tee -a "$log_dir/evaluator.log" >&2)
  ) | tee "$log_dir/accuracy.txt" | tee -a "$log_dir/evaluator.log"
  echo "stage=evaluate finished=$(date -Is)" | tee -a "$log_dir/evaluator.log"
  "$venv_dir/bin/python" - "$source_dir" "$log_dir" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

source, out = map(Path, sys.argv[1:])
expected = len(json.loads((source / "vision/medical_imaging/3d-unet-kits19/meta/inference_cases.json").read_text()))
records = json.loads((out / "mlperf_log_accuracy.json").read_text())
indices = sorted(x["qsl_idx"] for x in records)
assert len(records) == expected, (len(records), expected)
assert indices == list(range(expected)), indices
text = (out / "accuracy.txt").read_text()
m = re.search(r"^Accuracy: mean = ([0-9.]+), kidney = ([0-9.]+), tumor = ([0-9.]+)$", text, re.M)
assert m, "final accuracy line not found"
assert all(math.isfinite(float(x)) for x in m.groups())
print(f"validated {expected} unique predictions and metrics {m.groups()}")
PY
  touch "$work_dir/.evaluate.done"
}

for stage in "${stages[@]}"; do
  should_run "$stage" || continue
  marker="$work_dir/.${stage}.done"
  if [[ -f "$marker" && -z "$from_stage" ]]; then
    echo "skipping completed stage: $stage"
    continue
  fi
  "${stage}_stage"
done

