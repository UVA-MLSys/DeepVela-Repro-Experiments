#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet [--stage environment|data|gpu|evaluate] [--from environment|data|gpu|evaluate] [--preflight]" >&2
  exit 2
}

target=""; only_stage=""; from_stage=""; preflight=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; target="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; only_stage="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; from_stage="$2"; shift 2 ;;
    --preflight) preflight=1; shift ;;
    *) usage ;;
  esac
done
[[ "$target" == "3d-unet" ]] || usage
[[ -z "$only_stage" || -z "$from_stage" ]] || usage

stages=(environment data gpu evaluate)
valid_stage() { local x="$1" s; for s in "${stages[@]}"; do [[ "$x" == "$s" ]] && return 0; done; return 1; }
[[ -z "$only_stage" ]] || valid_stage "$only_stage" || usage
[[ -z "$from_stage" ]] || valid_stage "$from_stage" || usage

root_dir="$(pwd -P)"
repo_dir="$root_dir/repo"
cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
raw_input="$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data"
model_input="$cache_root/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER="2026-11-10"

if [[ $preflight -eq 1 ]]; then
  target_dir="$root_dir/$target/preflight"
  output_dir="$target_dir/out"
else
  target_dir="$root_dir/$target"
  output_dir="${OUTPUT_DIR:-$target_dir/out}"
  [[ "$output_dir" = /* ]] || output_dir="$root_dir/$output_dir"
fi
work_dir="$target_dir/work"; app_dir="$work_dir/app"; venv_dir="$target_dir/venv"; marker_dir="$target_dir/markers"
mkdir -p "$output_dir" "$marker_dir"

require_inputs() {
  [[ -d "$repo_dir/vision/medical_imaging/3d-unet-kits19" ]] || { echo "missing repository at $repo_dir" >&2; exit 1; }
  [[ -d "$raw_input" ]] || { echo "missing raw data at $raw_input" >&2; exit 1; }
  [[ -s "$model_input" ]] || { echo "missing model at $model_input" >&2; exit 1; }
}

clear_from() {
  local start="$1" active=0 s
  for s in "${stages[@]}"; do [[ "$s" == "$start" ]] && active=1; [[ $active -eq 0 ]] || rm -f "$marker_dir/$s.ok"; done
  case "$start" in
    environment)
      rm -rf "$venv_dir" "$work_dir"
      rm -f "$output_dir"/{build,preprocess,gpu_memory,run,evaluator}.log "$output_dir"/{accuracy.txt,mlperf_log_accuracy.json,mlperf_log_detail.txt} ;;
    data)
      rm -rf "$work_dir/raw_data" "$app_dir/build/preprocessed_data" "$app_dir/build/logs" "$app_dir/build/postprocessed_data"
      rm -f "$output_dir"/{preprocess,gpu_memory,run,evaluator}.log "$output_dir"/{accuracy.txt,mlperf_log_accuracy.json,mlperf_log_detail.txt} ;;
    gpu)
      rm -rf "$app_dir/build/logs" "$app_dir/build/postprocessed_data"
      rm -f "$output_dir"/{gpu_memory,run,evaluator}.log "$output_dir"/{accuracy.txt,mlperf_log_accuracy.json,mlperf_log_detail.txt} ;;
    evaluate)
      rm -rf "$app_dir/build/postprocessed_data"
      rm -f "$output_dir"/{evaluator.log,accuracy.txt} ;;
  esac
}

stage_environment() {
  [[ ! -e "$marker_dir/environment.ok" ]] || return 0
  require_inputs
  mkdir -p "$work_dir"; rm -rf "$app_dir" "$work_dir/loadgen"
  cp -a "$repo_dir/vision/medical_imaging/3d-unet-kits19" "$app_dir"
  cp -a "$repo_dir/loadgen" "$work_dir/loadgen"
  sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$app_dir/pytorch_SUT.py"
  if [[ $preflight -eq 1 ]]; then
    printf '[\n    "case_00005"\n]\n' > "$app_dir/meta/inference_cases.json"
    printf '{\n    "case_00005.pkl": "dbb4141573d72278c1724debe16311f5"\n}\n' > "$app_dir/meta/checksum_inference.json"
  fi
  {
    echo "Python environment: CPython 3.11"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv --python 3.11 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" \
      'torch==2.5.1' 'numpy==1.26.4' 'scipy==1.11.4' 'pandas==1.5.3' \
      'nibabel==5.2.1' 'pybind11==2.11.1' 'setuptools==75.6.0' 'wheel==0.45.1'
    MAX_JOBS=8 uv pip install --python "$venv_dir/bin/python" --no-build-isolation "$work_dir/loadgen"
    "$venv_dir/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__, "nibabel", nibabel.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
  } 2>&1 | tee "$output_dir/build.log"
  touch "$marker_dir/environment.ok"
}

stage_data() {
  [[ ! -e "$marker_dir/data.ok" ]] || return 0
  [[ -e "$marker_dir/environment.ok" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
  rm -rf "$work_dir/raw_data" "$app_dir/build/preprocessed_data"; mkdir -p "$work_dir/raw_data" "$app_dir/build/model"
  if [[ $preflight -eq 1 ]]; then
    ln -s "$raw_input/case_00005" "$work_dir/raw_data/case_00005"
  else
    while IFS= read -r case_name; do
      [[ -n "$case_name" ]] || continue
      source_case="$case_name"; [[ "$case_name" != "case_00400" ]] || source_case="case_00185"
      ln -s "$raw_input/$source_case" "$work_dir/raw_data/$case_name"
    done < <("$venv_dir/bin/python" - "$app_dir/meta/inference_cases.json" <<'PY'
import json, sys
for item in json.load(open(sys.argv[1])): print(item)
PY
    )
  fi
  ln -sfn "$model_input" "$app_dir/build/model/3dunet_kits19_pytorch.ptc"
  cp "$repo_dir/loadgen/mlperf.conf" "$app_dir/build/mlperf.conf"
  (cd "$app_dir" && "$venv_dir/bin/python" preprocess.py --raw_data_dir "$work_dir/raw_data" --results_dir "$app_dir/build/preprocessed_data" --mode preprocess --num_proc 8) 2>&1 | tee "$output_dir/preprocess.log"
  expected=43; [[ $preflight -eq 0 ]] || expected=1
  actual="$(find "$app_dir/build/preprocessed_data" -maxdepth 1 -name 'case_*.pkl' -type f | wc -l)"
  [[ "$actual" -eq "$expected" ]] || { echo "preprocessed $actual cases, expected $expected" >&2; exit 1; }
  touch "$marker_dir/data.ok"
}

stage_gpu() {
  [[ ! -e "$marker_dir/gpu.ok" ]] || return 0
  [[ -e "$marker_dir/data.ok" ]] || { echo "data stage has not succeeded" >&2; exit 1; }
  rm -rf "$app_dir/build/logs" "$app_dir/build/postprocessed_data"; mkdir -p "$app_dir/build/logs" "$app_dir/build/postprocessed_data"
  { date -Is; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader; } | tee "$output_dir/gpu_memory.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "could not determine free GPU memory" >&2; exit 1; }
  [[ "$free_mib" -ge 12000 ]] || { echo "insufficient free GPU memory: ${free_mib} MiB (need at least 12000 MiB)" >&2; exit 1; }
  (
    cd "$app_dir"; export LOG_PATH="$app_dir/build/logs"; export SKIP_VERIFY_ACCURACY=1
    "$venv_dir/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model="$model_input" --preprocessed_data_dir="$app_dir/build/preprocessed_data" \
      --mlperf_conf="$app_dir/build/mlperf.conf" --user_conf="$app_dir/user.conf"
  ) 2>&1 | tee "$output_dir/run.log"
  grep -q '^pytorch device: ' "$output_dir/run.log"
  test -s "$app_dir/build/logs/mlperf_log_accuracy.json"; test -s "$app_dir/build/logs/mlperf_log_detail.txt"
  cp "$app_dir/build/logs/mlperf_log_accuracy.json" "$output_dir/mlperf_log_accuracy.json"
  cp "$app_dir/build/logs/mlperf_log_detail.txt" "$output_dir/mlperf_log_detail.txt"
  touch "$marker_dir/gpu.ok"
}

stage_evaluate() {
  [[ ! -e "$marker_dir/evaluate.ok" ]] || return 0
  [[ -e "$marker_dir/gpu.ok" ]] || { echo "gpu stage has not succeeded" >&2; exit 1; }
  rm -rf "$app_dir/build/postprocessed_data"; mkdir -p "$app_dir/build/postprocessed_data"
  (cd "$app_dir" && "$venv_dir/bin/python" accuracy_kits.py \
    --log_file="$output_dir/mlperf_log_accuracy.json" --preprocessed_data_dir="$app_dir/build/preprocessed_data" \
    --postprocessed_data_dir="$app_dir/build/postprocessed_data" --num_proc=8) 2>&1 | tee "$output_dir/evaluator.log"
  cp "$output_dir/evaluator.log" "$output_dir/accuracy.txt"
  grep -Eq '^Accuracy: mean = [0-9]+\.[0-9]{5}, kidney = [0-9]+\.[0-9]{4}, tumor = [0-9]+\.[0-9]{4}$' "$output_dir/accuracy.txt"
  test -s "$output_dir/mlperf_log_accuracy.json"; test -s "$output_dir/mlperf_log_detail.txt"
  touch "$marker_dir/evaluate.ok"
}

run_stage() { case "$1" in environment) stage_environment ;; data) stage_data ;; gpu) stage_gpu ;; evaluate) stage_evaluate ;; esac; }
[[ -z "$from_stage" ]] || clear_from "$from_stage"
if [[ -n "$only_stage" ]]; then
  run_stage "$only_stage"
else
  active=1; [[ -z "$from_stage" ]] || active=0
  for s in "${stages[@]}"; do [[ -z "$from_stage" || "$s" != "$from_stage" ]] || active=1; [[ $active -eq 0 ]] || run_stage "$s"; done
fi
