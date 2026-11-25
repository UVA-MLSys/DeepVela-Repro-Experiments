#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target 3d-unet [--stage setup|preprocess|run|evaluate] [--from setup|preprocess|run|evaluate]" >&2
  exit 2
}

target=""
only_stage=""
from_stage=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="${2:-}"; shift 2 ;;
    --stage) only_stage="${2:-}"; shift 2 ;;
    --from) from_stage="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$target" == "3d-unet" ]] || usage
[[ -z "$only_stage" || -z "$from_stage" ]] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$script_dir/repo"
bench_src="$repo_dir/vision/medical_imaging/3d-unet-kits19"
cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
raw_input="$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data"
model_input="$cache_root/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
uv_cache="$cache_root/uv"
work_dir="${REPRO_WORK_DIR:-$script_dir/$target}"
output_dir="${OUTPUT_DIR:-$work_dir/out}"
mkdir -p "$output_dir" "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
output_dir="$(cd "$output_dir" && pwd)"
venv="$work_dir/.venv"
bench="$work_dir/benchmark"
build_dir="$work_dir/build"
preprocessed="$build_dir/preprocessed_data"
postprocessed="$build_dir/postprocessed_data"
log_dir="$output_dir"
num_proc="${NUM_PROC:-4}"
preflight_cases="${PREFLIGHT_CASES:-}"

stage_index() {
  case "$1" in
    setup) echo 0 ;;
    preprocess) echo 1 ;;
    run) echo 2 ;;
    evaluate) echo 3 ;;
    *) usage ;;
  esac
}

should_run() {
  local name="$1"
  if [[ -n "$only_stage" ]]; then
    [[ "$name" == "$only_stage" ]]
  elif [[ -n "$from_stage" ]]; then
    [[ "$(stage_index "$name")" -ge "$(stage_index "$from_stage")" ]]
  else
    [[ ! -f "$work_dir/.${name}.success" ]]
  fi
}

invalidate_from() {
  local start="$1" idx name
  idx="$(stage_index "$start")"
  for name in setup preprocess run evaluate; do
    if [[ "$(stage_index "$name")" -ge "$idx" ]]; then
      rm -f "$work_dir/.${name}.success"
    fi
  done
  if [[ "$idx" -le 1 ]]; then
    rm -rf "$preprocessed" "$postprocessed"
  elif [[ "$idx" -le 2 ]]; then
    rm -rf "$postprocessed"
  fi
  if [[ "$idx" -le 2 ]]; then
    rm -f "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" \
      "$output_dir/mlperf_log_summary.txt" "$output_dir/accuracy.txt"
  elif [[ "$idx" -le 3 ]]; then
    rm -rf "$postprocessed"
    rm -f "$output_dir/accuracy.txt"
  fi
}

if [[ -n "$only_stage" ]]; then
  invalidate_from "$only_stage"
elif [[ -n "$from_stage" ]]; then
  invalidate_from "$from_stage"
fi

run_setup() {
  : > "$output_dir/build.log"
  {
    echo "source repository: $repo_dir"
    echo "benchmark source commit supplied by task: 3fbc329939999c13d0a7b5e67fb2092287e06047"
    echo "model sha256: $(sha256sum "$model_input" | awk '{print $1}')"
    export UV_CACHE_DIR="$uv_cache"
    export UV_EXCLUDE_NEWER=2026-11-10
    uv venv --python 3.11 --clear "$venv"
    uv pip install --python "$venv/bin/python" \
      "torch==2.4.1" "numpy==1.26.4" "scipy==1.11.4" \
      "nibabel==3.2.1" "pandas==1.5.3" "pybind11==2.11.1"
    CFLAGS="-std=c++14" CXXFLAGS="-std=c++14" \
      uv pip install --python "$venv/bin/python" "$repo_dir/loadgen"

    rm -rf "$bench"
    cp -a "$bench_src" "$bench"
    cp "$repo_dir/loadgen/mlperf.conf" "$bench/mlperf.conf"
    "$venv/bin/python" - "$bench/pytorch_SUT.py" "$bench/run.py" "$bench/mlperf.conf" "$bench/user.conf" "$bench/combined.conf" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
needle = "        self.model.eval()\n"
addition = needle + '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n'
if "pytorch device: " not in s:
    if needle not in s:
        raise RuntimeError("could not locate model.eval() for device evidence patch")
    p.write_text(s.replace(needle, addition, 1))

run_py = Path(sys.argv[2])
s = run_py.read_text()
needle = '    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)\n'
if needle not in s:
    raise RuntimeError("could not locate second LoadGen config call")
run_py.write_text(s.replace(needle, "", 1))

mlperf_conf, user_conf, combined_conf = map(Path, sys.argv[3:6])
combined_conf.write_text(mlperf_conf.read_text() + "\n" + user_conf.read_text())
PY

    mkdir -p "$build_dir/raw_data" "$build_dir/model"
    "$venv/bin/python" - "$bench/meta/inference_cases.json" "$bench/meta/checksum_inference.json" "$preflight_cases" <<'PY'
import json, sys
from pathlib import Path
manifest, checksums = map(Path, sys.argv[1:3])
selected = sys.argv[3]
if selected:
    wanted = selected.split(",")
    all_cases = json.loads(manifest.read_text())
    if any(case not in all_cases for case in wanted):
        raise RuntimeError("preflight case is not in the inference manifest")
    source = json.loads(checksums.read_text())
    manifest.write_text(json.dumps(wanted, indent=2) + "\n")
    checksums.write_text(json.dumps({case + ".pkl": source[case + ".pkl"] for case in wanted}, indent=2, sort_keys=True) + "\n")
PY
    while IFS= read -r case_name; do
      source_name="$case_name"
      [[ "$case_name" == "case_00400" ]] && source_name="case_00185"
      ln -sfn "$raw_input/$source_name" "$build_dir/raw_data/$case_name"
    done < <("$venv/bin/python" -c 'import json,sys; print(*json.load(open(sys.argv[1])), sep="\n")' "$bench/meta/inference_cases.json")
    ln -sfn "$model_input" "$build_dir/model/3dunet_kits19_pytorch.ptc"
    "$venv/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("python/package import check: ok")
print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
print("numpy:", numpy.__version__, "scipy:", scipy.__version__)
print("nibabel:", nibabel.__version__, "pandas:", pandas.__version__)
PY
  } 2>&1 | tee -a "$output_dir/build.log"
  touch "$work_dir/.setup.success"
}

run_preprocess() {
  : > "$output_dir/preprocess.log"
  mkdir -p "$preprocessed"
  (
    cd "$bench"
    /usr/bin/time -v "$venv/bin/python" preprocess.py \
      --raw_data_dir "$build_dir/raw_data" \
      --results_dir "$preprocessed" --mode preprocess --num_proc "$num_proc"
  ) 2>&1 | tee "$output_dir/preprocess.log"
  touch "$work_dir/.preprocess.success"
}

run_inference() {
  : > "$output_dir/gpu_check.log"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader 2>&1 | tee "$output_dir/gpu_check.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ "$free_mib" -lt 20000 ]]; then
    echo "insufficient free GPU memory: ${free_mib} MiB; need at least 20000 MiB" | tee -a "$output_dir/gpu_check.log" >&2
    exit 1
  fi
  rm -f "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" "$output_dir/mlperf_log_summary.txt"
  (
    cd "$bench"
    export LOG_PATH="$log_dir"
    export SKIP_VERIFY_ACCURACY=1
    export PYTHONUNBUFFERED=1
    /usr/bin/time -v "$venv/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$model_input" --preprocessed_data_dir="$preprocessed" \
      --mlperf_conf="$bench/combined.conf" --user_conf="$bench/user.conf"
  ) 2>&1 | tee "$output_dir/run.log"
  test -s "$output_dir/mlperf_log_accuracy.json"
  test -s "$output_dir/mlperf_log_detail.txt"
  touch "$work_dir/.run.success"
}

run_evaluator() {
  rm -rf "$postprocessed"
  mkdir -p "$postprocessed"
  (
    cd "$bench"
    "$venv/bin/python" accuracy_kits.py \
      --preprocessed_data_dir="$preprocessed" \
      --postprocessed_data_dir="$postprocessed" \
      --log_file="$output_dir/mlperf_log_accuracy.json" --num_proc="$num_proc"
  ) 2>&1 | tee "$output_dir/accuracy.txt" "$output_dir/evaluator.log"
  grep -q '^Accuracy: mean = ' "$output_dir/accuracy.txt"
  touch "$work_dir/.evaluate.success"
}

if should_run setup; then run_setup; fi
if should_run preprocess; then run_preprocess; fi
if should_run run; then run_inference; fi
if should_run evaluate; then run_evaluator; fi

test -s "$output_dir/mlperf_log_accuracy.json" || [[ -n "$only_stage" && "$(stage_index "$only_stage")" -lt 2 ]]
test -s "$output_dir/mlperf_log_detail.txt" || [[ -n "$only_stage" && "$(stage_index "$only_stage")" -lt 2 ]]
test -s "$output_dir/accuracy.txt" || [[ -n "$only_stage" && "$(stage_index "$only_stage")" -lt 3 ]]
