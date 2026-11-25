#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet [--stage setup|preprocess|run|evaluate] [--from setup|preprocess|run|evaluate]" >&2
  exit 2
}

target=""
stage=""
from=""
while (($#)); do
  case "$1" in
    --target) target="${2:-}"; shift 2 ;;
    --stage) stage="${2:-}"; shift 2 ;;
    --from) from="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$target" == "3d-unet" ]] || usage
[[ -z "$stage" || -z "$from" ]] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$script_dir/repo"
bench_src="$repo_dir/vision/medical_imaging/3d-unet-kits19"
cache_root="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
raw_src="${KITS19_RAW_DIR:-$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data}"
model_src="${MODEL_DIR:-$cache_root/3d-unet-kits19/model}"
uv_cache="${UV_CACHE_DIR:-$cache_root/uv}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
output_dir="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
target_dir="$(dirname "$output_dir")"
work_dir="$target_dir/work"
venv_dir="$work_dir/venv"
bench_dir="$work_dir/benchmark"
raw_dir="$work_dir/raw_data"
preprocessed_dir="$work_dir/preprocessed_data"
postprocessed_dir="$work_dir/postprocessed_data"
log_dir="$work_dir/loadgen_logs"
marker_dir="$work_dir/markers"
model_path="$model_src/3dunet_kits19_pytorch.ptc"
python_bin="$venv_dir/bin/python"
export UV_CACHE_DIR="$uv_cache"
export UV_EXCLUDE_NEWER=2026-11-10

stages=(setup preprocess run evaluate)
stage_index() {
  case "$1" in
    setup) echo 0 ;; preprocess) echo 1 ;; run) echo 2 ;; evaluate) echo 3 ;; *) usage ;;
  esac
}

clear_from() {
  local first="$1" idx i name
  idx="$(stage_index "$first")"
  mkdir -p "$marker_dir"
  for ((i=idx; i<${#stages[@]}; i++)); do
    name="${stages[$i]}"
    rm -f "$marker_dir/$name.ok" "$output_dir/$name.log"
  done
  if ((idx <= 1)); then rm -rf "$preprocessed_dir"; fi
  if ((idx <= 2)); then
    rm -rf "$postprocessed_dir" "$log_dir"
    rm -f "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" "$output_dir/accuracy.txt"
  fi
  if ((idx <= 3)); then
    rm -rf "$postprocessed_dir"
    rm -f "$output_dir/accuracy.txt"
  fi
}

run_setup() {
  local log="$output_dir/setup.log"
  mkdir -p "$work_dir" "$marker_dir"
  {
    echo "source repository: $repo_dir"
    echo "benchmark source: $bench_src"
    echo "raw input: $raw_src"
    echo "model input: $model_path"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    test -d "$bench_src"
    test -d "$raw_src"
    test -s "$model_path"

    rm -rf "$bench_dir" "$work_dir/loadgen_src" "$raw_dir"
    mkdir -p "$bench_dir" "$work_dir/loadgen_src" "$raw_dir" "$preprocessed_dir" "$postprocessed_dir" "$log_dir"
    cp -a "$bench_src/." "$bench_dir/"
    cp -a "$repo_dir/loadgen/." "$work_dir/loadgen_src/"
    cp "$repo_dir/mlperf.conf" "$bench_dir/mlperf.conf"

    # Required diagnostic and two compatibility repairs in the target-local copy.
    sed -i '/self.model.eval()/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$bench_dir/pytorch_SUT.py"
    sed -i 's/signal\.gaussian(/signal.windows.gaussian(/' "$bench_dir/inference_utils.py"
    sed -i '/df.set_index("case", inplace=True)/a\    df.loc["mean"] = df.mean()' "$bench_dir/accuracy_kits.py"

    if [[ -n "${REPRO_PREFLIGHT_CASES:-}" ]]; then
      REPRO_PREFLIGHT_CASES="$REPRO_PREFLIGHT_CASES" /usr/bin/python3.11 - "$bench_dir/meta/inference_cases.json" "$bench_dir/meta/checksum_inference.json" <<'PY'
import json, os, sys
manifest_path, checksum_path = sys.argv[1:]
cases = [x.strip() for x in os.environ["REPRO_PREFLIGHT_CASES"].split(",") if x.strip()]
manifest = json.load(open(manifest_path))
if not cases or any(x not in manifest for x in cases):
    raise SystemExit("invalid REPRO_PREFLIGHT_CASES")
checksums = json.load(open(checksum_path))
with open(manifest_path, "w") as f:
    json.dump(cases, f, indent=4)
with open(checksum_path, "w") as f:
    json.dump({f"{x}.pkl": checksums[f"{x}.pkl"] for x in cases}, f, indent=4, sort_keys=True)
PY
      echo "bounded preflight cases: $REPRO_PREFLIGHT_CASES"
    fi

    while IFS= read -r case_name; do
      source_case="$case_name"
      [[ "$case_name" == "case_00400" ]] && source_case="case_00185"
      test -d "$raw_src/$source_case"
      ln -s "$raw_src/$source_case" "$raw_dir/$case_name"
    done < <(/usr/bin/python3.11 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))))' "$bench_dir/meta/inference_cases.json")

    if [[ ! -x "$python_bin" ]]; then
      uv venv --python /usr/bin/python3.11 "$venv_dir"
    fi
    uv pip install --python "$python_bin" \
      'numpy==1.26.4' 'scipy==1.13.1' 'nibabel==5.4.2' 'pandas==1.5.3' \
      'python-dateutil==2.9.0.post0' 'torch==2.7.1' 'pybind11==2.13.6' 'setuptools==84.2.1' 'wheel==0.48.0'
    uv pip install --python "$python_bin" 'mlcommons-loadgen==6.0.17'
    "$python_bin" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("python packages:", "numpy", numpy.__version__, "scipy", scipy.__version__, "nibabel", nibabel.__version__, "pandas", pandas.__version__, "torch", torch.__version__)
print("cuda available:", torch.cuda.is_available(), "cuda runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "PyTorch cannot access the required GPU"
PY
  } 2>&1 | tee "$log"
  touch "$marker_dir/setup.ok"
}

run_preprocess() {
  local log="$output_dir/preprocess.log"
  test -f "$marker_dir/setup.ok"
  mkdir -p "$preprocessed_dir"
  {
    cd "$bench_dir"
    /usr/bin/time -v "$python_bin" preprocess.py \
      --raw_data_dir "$raw_dir" --results_dir "$preprocessed_dir" --mode preprocess --num_proc 8
  } 2>&1 | tee "$log"
  touch "$marker_dir/preprocess.ok"
}

wait_for_gpu_memory() {
  local need_mib=16000 tries=0 free_mib
  while true; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')"
    echo "GPU memory before run: ${free_mib} MiB free; required: ${need_mib} MiB"
    if ((free_mib >= need_mib)); then return 0; fi
    tries=$((tries + 1))
    if ((tries >= 30)); then
      echo "GPU did not have ${need_mib} MiB free within 300 seconds" >&2
      return 1
    fi
    sleep 10
  done
}

run_inference() {
  local log="$output_dir/run.log"
  test -f "$marker_dir/preprocess.ok"
  mkdir -p "$log_dir"
  {
    wait_for_gpu_memory
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
    cd "$bench_dir"
    LOG_PATH="$log_dir" SKIP_VERIFY_ACCURACY=1 /usr/bin/time -v "$python_bin" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$model_path" --preprocessed_data_dir="$preprocessed_dir" \
      --mlperf_conf="$bench_dir/mlperf.conf" --user_conf="$bench_dir/user.conf"
  } 2>&1 | tee "$log"
  cp "$log_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_accuracy.json"
  cp "$log_dir/mlperf_log_detail.txt" "$output_dir/mlperf_log_detail.txt"
  touch "$marker_dir/run.ok"
}

run_evaluate() {
  local log="$output_dir/evaluator.log"
  local stderr_log="$output_dir/evaluator.stderr.log"
  test -f "$marker_dir/run.ok"
  mkdir -p "$postprocessed_dir"
  : > "$stderr_log"
  (
    cd "$bench_dir"
    "$python_bin" accuracy_kits.py \
      --log_file="$output_dir/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="$preprocessed_dir" \
      --postprocessed_data_dir="$postprocessed_dir" --num_proc=8
  ) > "$output_dir/accuracy.txt" 2> "$stderr_log"
  tee "$log" < "$output_dir/accuracy.txt"
  tee -a "$log" < "$stderr_log" >&2
  touch "$marker_dir/evaluate.ok"
}

if [[ -n "$from" ]]; then clear_from "$from"; fi

if [[ -n "$stage" ]]; then
  selected=("$stage")
elif [[ -n "$from" ]]; then
  first_idx="$(stage_index "$from")"
  selected=("${stages[@]:first_idx}")
else
  selected=("${stages[@]}")
fi

for current in "${selected[@]}"; do
  if [[ -f "$marker_dir/$current.ok" ]]; then
    echo "stage $current already complete"
    continue
  fi
  case "$current" in
    setup) run_setup ;; preprocess) run_preprocess ;; run) run_inference ;; evaluate) run_evaluate ;;
  esac
done
