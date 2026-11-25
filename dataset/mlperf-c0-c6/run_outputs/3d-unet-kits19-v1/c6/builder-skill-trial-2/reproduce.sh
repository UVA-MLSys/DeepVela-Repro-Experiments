#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet [--stage env|data|run|eval] [--from env|data|run|eval]" >&2
  exit 2
}

target=""
stage=""
from=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; target="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; stage="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; from="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$target" == "3d-unet" ]] || usage
[[ -z "$stage" || -z "$from" ]] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$script_dir/repo"
[[ -d "$repo_dir/vision/medical_imaging/3d-unet-kits19" ]] || {
  echo "fresh repository copy not found at $repo_dir" >&2
  exit 1
}

cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
raw_input="${KITS19_RAW_DIR:-$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data}"
model_input="${MODEL_DIR:-$cache_root/3d-unet-kits19/model}"
export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER="2026-11-10"

target_dir="$script_dir/$target"
output_dir="${OUTPUT_DIR:-$target_dir/out}"
env_dir="$target_dir/env"
profile="full"
case_count="43"
if [[ -n "${PREFLIGHT_CASE_COUNT:-}" ]]; then
  [[ "$PREFLIGHT_CASE_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid PREFLIGHT_CASE_COUNT" >&2; exit 2; }
  (( PREFLIGHT_CASE_COUNT <= 43 )) || { echo "PREFLIGHT_CASE_COUNT exceeds 43" >&2; exit 2; }
  profile="preflight-$PREFLIGHT_CASE_COUNT"
  case_count="$PREFLIGHT_CASE_COUNT"
fi
work_dir="$target_dir/work-$profile"
app_dir="$work_dir/app"
raw_stage="$work_dir/raw_data"
preprocessed="$work_dir/preprocessed_data"
postprocessed="$work_dir/postprocessed_data"
marker_dir="$target_dir/markers-$profile"
mkdir -p "$output_dir" "$marker_dir"

stages=(env data run eval)
valid_stage() {
  local candidate="$1" item
  for item in "${stages[@]}"; do [[ "$candidate" == "$item" ]] && return 0; done
  return 1
}
[[ -z "$stage" ]] || valid_stage "$stage" || usage
[[ -z "$from" ]] || valid_stage "$from" || usage

if [[ -n "$from" ]]; then
  found=0
  for item in "${stages[@]}"; do
    [[ "$item" == "$from" ]] && found=1
    if (( found )); then
      rm -f "$marker_dir/$item.done"
      case "$item" in
        env) rm -rf "$env_dir" "$target_dir/loadgen-build" ;;
        data) rm -rf "$work_dir" ;;
        run) rm -rf "$postprocessed"; rm -f "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" "$output_dir/mlperf_log_summary.txt" ;;
        eval) rm -rf "$postprocessed"; rm -f "$output_dir/accuracy.txt" ;;
      esac
    fi
  done
fi

run_env() {
  local log="$output_dir/build.log"
  exec > >(tee -a "$log") 2>&1
  echo "[env] started $(date --iso-8601=seconds)"
  command -v uv
  rm -rf "$env_dir" "$target_dir/loadgen-build"
  uv venv --python 3.11 --seed "$env_dir"
  uv pip install --python "$env_dir/bin/python" \
    "numpy==1.26.4" "scipy==1.11.4" "nibabel==5.2.1" "pandas==1.5.3" \
    "torch==2.6.0" "pybind11==2.11.1" "setuptools<81" wheel
  cp -a "$repo_dir/loadgen" "$target_dir/loadgen-build"
  uv pip install --python "$env_dir/bin/python" "$target_dir/loadgen-build"
  "$env_dir/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
print("nibabel", nibabel.__version__, "pandas", pandas.__version__)
print("mlperf_loadgen", mlperf_loadgen.__file__)
PY
  echo "[env] finished $(date --iso-8601=seconds)"
  touch "$marker_dir/env.done"
}

run_data() {
  local log="$output_dir/preprocess.log"
  exec > >(tee -a "$log") 2>&1
  [[ -f "$marker_dir/env.done" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
  [[ -f "$raw_input/case_00185/imaging.nii.gz" && -f "$model_input/3dunet_kits19_pytorch.ptc" ]] || {
    echo "required cached data/model missing" >&2; exit 1;
  }
  echo "[data] started $(date --iso-8601=seconds), profile=$profile"
  rm -rf "$work_dir"
  mkdir -p "$work_dir" "$raw_stage"
  cp -a "$repo_dir/vision/medical_imaging/3d-unet-kits19" "$app_dir"
  cp "$repo_dir/loadgen/mlperf.conf" "$app_dir/mlperf.conf"
  "$env_dir/bin/python" - "$app_dir/meta/inference_cases.json" "$case_count" "$raw_input" "$raw_stage" <<'PY'
import json, os, pathlib, sys
manifest = pathlib.Path(sys.argv[1])
count = int(sys.argv[2])
raw = pathlib.Path(sys.argv[3])
stage = pathlib.Path(sys.argv[4])
cases = json.loads(manifest.read_text())[:count]
manifest.write_text(json.dumps(cases, indent=4) + "\n")
checksum_path = manifest.with_name("checksum_inference.json")
checksums = json.loads(checksum_path.read_text())
checksum_path.write_text(json.dumps({f"{c}.pkl": checksums[f"{c}.pkl"] for c in cases}, indent=4, sort_keys=True) + "\n")
for case in cases:
    source_case = "case_00185" if case == "case_00400" else case
    source = raw / source_case
    if not (source / "imaging.nii.gz").is_file() or not (source / "segmentation.nii.gz").is_file():
        raise SystemExit(f"missing raw case files: {source}")
    os.symlink(source, stage / case, target_is_directory=True)
print(f"staged {len(cases)} cases; case_00400 maps to case_00185 when selected")
PY
  if [[ "$profile" != "full" ]]; then
    sed -i "s/^3d-unet\.Offline\.min_query_count = 43$/3d-unet.Offline.min_query_count = $case_count/" "$app_dir/mlperf.conf"
    grep '^3d-unet\.Offline\.min_query_count' "$app_dir/mlperf.conf"
  fi
  cd "$app_dir"
  "$env_dir/bin/python" preprocess.py --raw_data_dir "$raw_stage" --results_dir "$preprocessed" --mode preprocess --num_proc 4
  echo "[data] finished $(date --iso-8601=seconds)"
  touch "$marker_dir/data.done"
}

run_gpu() {
  local log="$output_dir/run.log"
  exec > >(tee -a "$log") 2>&1
  [[ -f "$marker_dir/data.done" ]] || { echo "data stage has not succeeded" >&2; exit 1; }
  echo "[run] started $(date --iso-8601=seconds), profile=$profile"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  (( free_mib >= 24000 )) || { echo "insufficient free GPU memory: ${free_mib} MiB (need 24000 MiB)" >&2; exit 1; }
  cd "$app_dir"
  "$env_dir/bin/python" - "$model_input/3dunet_kits19_pytorch.ptc" <<'PY'
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
  rm -rf "$postprocessed"
  mkdir -p "$postprocessed" "$output_dir"
  LOG_PATH="$output_dir" SKIP_VERIFY_ACCURACY=1 "$env_dir/bin/python" run.py \
    --backend=pytorch --scenario=Offline --accuracy \
    --model="$model_input/3dunet_kits19_pytorch.ptc" \
    --preprocessed_data_dir="$preprocessed" \
    --mlperf_conf="$app_dir/mlperf.conf" --user_conf="$app_dir/user.conf"
  [[ -s "$output_dir/mlperf_log_accuracy.json" && -s "$output_dir/mlperf_log_detail.txt" ]] || {
    echo "LoadGen did not create required logs" >&2; exit 1;
  }
  echo "[run] finished $(date --iso-8601=seconds)"
  touch "$marker_dir/run.done"
}

run_eval() {
  local log="$output_dir/evaluator.log"
  exec > >(tee -a "$log") 2>&1
  [[ -f "$marker_dir/run.done" ]] || { echo "run stage has not succeeded" >&2; exit 1; }
  echo "[eval] started $(date --iso-8601=seconds), profile=$profile"
  cd "$app_dir"
  "$env_dir/bin/python" accuracy_kits.py \
    --preprocessed_data_dir="$preprocessed" \
    --postprocessed_data_dir="$postprocessed" \
    --log_file="$output_dir/mlperf_log_accuracy.json" --num_proc 4 | tee "$output_dir/accuracy.txt"
  grep -q '^Accuracy: mean = ' "$output_dir/accuracy.txt"
  "$env_dir/bin/python" - "$output_dir/mlperf_log_accuracy.json" "$case_count" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
indices = {row["qsl_idx"] for row in rows}
if len(rows) != expected or indices != set(range(expected)):
    raise SystemExit(f"accuracy log coverage failure: rows={len(rows)} unique_indices={len(indices)} expected={expected}")
print(f"validated accuracy log coverage: {expected}/{expected} unique query indices")
PY
  grep -q '^pytorch device: ' "$output_dir/run.log"
  echo "[eval] finished $(date --iso-8601=seconds)"
  touch "$marker_dir/eval.done"
}

run_one() {
  case "$1" in
    env) run_env ;;
    data) run_data ;;
    run) run_gpu ;;
    eval) run_eval ;;
  esac
}

if [[ -n "$stage" ]]; then
  [[ ! -f "$marker_dir/$stage.done" ]] || { echo "[$stage] already complete"; exit 0; }
  run_one "$stage"
  exit 0
fi

start=0
[[ -z "$from" ]] && start=1
for item in "${stages[@]}"; do
  [[ "$item" == "$from" ]] && start=1
  (( start )) || continue
  if [[ -f "$marker_dir/$item.done" ]]; then
    echo "[$item] already complete"
  else
    run_one "$item"
  fi
done
