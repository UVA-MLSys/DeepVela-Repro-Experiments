#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: OUTPUT_DIR=... $0 --target 3d-unet [--stage environment|preprocess|run|evaluate] [--from environment|preprocess|run|evaluate]" >&2
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
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$script_dir/repo"
target_dir="$script_dir/$target"
work_dir="$target_dir/work"
out_dir="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
venv_dir="$work_dir/venv"
bench_dir="$work_dir/benchmark"
raw_links="$work_dir/raw_data"
preprocessed_dir="$work_dir/preprocessed_data"
postprocessed_dir="$work_dir/postprocessed_data"
marker_dir="$target_dir/markers"

cache_root=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
raw_source="$cache_root/3d-unet-kits19/kits19_raw_data/kits19/data"
model_source="$cache_root/3d-unet-kits19/model/3dunet_kits19_pytorch.ptc"
export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$work_dir" "$marker_dir" "$out_dir"
[[ -d "$repo_dir" ]] || { echo "fresh repository copy not found at $repo_dir" >&2; exit 1; }
[[ -d "$raw_source" ]] || { echo "raw input not found at $raw_source" >&2; exit 1; }
[[ -s "$model_source" ]] || { echo "model not found at $model_source" >&2; exit 1; }

invalidate_from() {
  local first="$1"
  case "$first" in
    environment)
      rm -rf "$venv_dir" "$bench_dir" "$work_dir/loadgen"
      rm -f "$marker_dir/environment.ok"
      ;&
    preprocess)
      rm -rf "$raw_links" "$preprocessed_dir"
      rm -f "$marker_dir/preprocess.ok" "$out_dir/preprocess.log"
      ;&
    run)
      rm -rf "$postprocessed_dir"
      rm -f "$marker_dir/run.ok" "$out_dir/run.log" \
        "$out_dir/mlperf_log_accuracy.json" "$out_dir/mlperf_log_detail.txt" \
        "$out_dir/mlperf_log_summary.txt" "$out_dir/mlperf_log_trace.json"
      ;&
    evaluate)
      rm -f "$marker_dir/evaluate.ok" "$out_dir/evaluator.log" "$out_dir/accuracy.txt"
      ;;
    *) usage ;;
  esac
}

run_environment() {
  [[ -f "$marker_dir/environment.ok" ]] && return
  {
    echo "Creating isolated Python environment"
    uv venv --python 3.11 "$venv_dir"
    uv pip install --python "$venv_dir/bin/python" \
      'numpy==1.26.4' 'scipy==1.10.1' 'pandas==1.5.3' \
      'nibabel==5.3.2' 'torch==2.7.1'
    cp -a "$repo_dir/vision/medical_imaging/3d-unet-kits19" "$bench_dir"
    cp -a "$repo_dir/loadgen" "$work_dir/loadgen"
    uv pip install --python "$venv_dir/bin/python" "$work_dir/loadgen"
    "$venv_dir/bin/python" - <<'PY'
import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("loadgen", mlperf_loadgen.__file__)
PY
  } 2>&1 | tee "$out_dir/environment.log"
  touch "$marker_dir/environment.ok"
}

run_preprocess() {
  [[ -f "$marker_dir/preprocess.ok" ]] && return
  [[ -f "$marker_dir/environment.ok" ]] || run_environment
  {
    rm -rf "$raw_links" "$preprocessed_dir"
    mkdir -p "$raw_links" "$preprocessed_dir"
    cp "$repo_dir/vision/medical_imaging/3d-unet-kits19/meta/inference_cases.json" "$bench_dir/meta/inference_cases.json"
    cp "$repo_dir/vision/medical_imaging/3d-unet-kits19/meta/checksum_inference.json" "$bench_dir/meta/checksum_inference.json"
    if [[ -n "${REPRO_CASES:-}" ]]; then
      REPRO_CASES="$REPRO_CASES" BENCH_DIR="$bench_dir" "$venv_dir/bin/python" - <<'PY'
import json, os
from pathlib import Path
b = Path(os.environ["BENCH_DIR"])
requested = os.environ["REPRO_CASES"].split(",")
full = json.loads((b / "meta/inference_cases.json").read_text())
if not requested or any(x not in full for x in requested):
    raise SystemExit("REPRO_CASES contains an unknown case")
(b / "meta/inference_cases.json").write_text(json.dumps(requested, indent=2) + "\n")
checks = json.loads((b / "meta/checksum_inference.json").read_text())
reduced = {f"{x}.pkl": checks[f"{x}.pkl"] for x in requested}
(b / "meta/checksum_inference.json").write_text(json.dumps(reduced, indent=4, sort_keys=True) + "\n")
print("Bounded preflight cases:", requested)
PY
    fi
    BENCH_DIR="$bench_dir" RAW_SOURCE="$raw_source" RAW_LINKS="$raw_links" "$venv_dir/bin/python" - <<'PY'
import json, os
from pathlib import Path
b = Path(os.environ["BENCH_DIR"])
src = Path(os.environ["RAW_SOURCE"])
dst = Path(os.environ["RAW_LINKS"])
cases = json.loads((b / "meta/inference_cases.json").read_text())
for case in cases:
    actual = "case_00185" if case == "case_00400" else case
    target = src / actual
    for name in ("imaging.nii.gz", "segmentation.nii.gz"):
        if not (target / name).is_file():
            raise SystemExit(f"missing input {target / name}")
    (dst / case).symlink_to(target, target_is_directory=True)
print(f"Resolved {len(cases)} manifest cases")
PY
    cd "$bench_dir"
    "$venv_dir/bin/python" preprocess.py --raw_data_dir "$raw_links" \
      --results_dir "$preprocessed_dir" --mode preprocess --num_proc 4
    "$venv_dir/bin/python" preprocess.py --raw_data_dir "$raw_links" \
      --results_dir "$preprocessed_dir" --mode verify --num_proc 4
  } 2>&1 | tee "$out_dir/preprocess.log"
  touch "$marker_dir/preprocess.ok"
}

run_gpu() {
  [[ -f "$marker_dir/run.ok" ]] && return
  [[ -f "$marker_dir/preprocess.ok" ]] || run_preprocess
  {
    echo "GPU state immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    if (( free_mib < 24000 )); then
      echo "insufficient free GPU memory: ${free_mib} MiB (need 24000 MiB)" >&2
      exit 1
    fi
    MODEL_SOURCE="$model_source" "$venv_dir/bin/python" - <<'PY'
import os, torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise SystemExit("CUDA is unavailable")
model = torch.jit.load(os.environ["MODEL_SOURCE"], map_location=device)
print("pytorch device: ", next(model.parameters()).device)
del model
torch.cuda.empty_cache()
PY
    rm -rf "$postprocessed_dir"
    mkdir -p "$postprocessed_dir"
    cd "$bench_dir"
    SKIP_VERIFY_ACCURACY=1 LOG_PATH="$out_dir" "$venv_dir/bin/python" run.py \
      --backend=pytorch --scenario=Offline --accuracy \
      --model="$model_source" --preprocessed_data_dir="$preprocessed_dir" \
      --mlperf_conf="$repo_dir/mlperf.conf" --user_conf="$bench_dir/user.conf"
  } 2>&1 | tee "$out_dir/run.log"
  [[ -s "$out_dir/mlperf_log_accuracy.json" ]] || { echo "accuracy log missing" >&2; exit 1; }
  [[ -s "$out_dir/mlperf_log_detail.txt" ]] || { echo "detail log missing" >&2; exit 1; }
  touch "$marker_dir/run.ok"
}

run_evaluate() {
  [[ -f "$marker_dir/evaluate.ok" ]] && return
  [[ -f "$marker_dir/run.ok" ]] || run_gpu
  {
    cd "$bench_dir"
    "$venv_dir/bin/python" accuracy_kits.py \
      --preprocessed_data_dir="$preprocessed_dir" \
      --postprocessed_data_dir="$postprocessed_dir" \
      --log_file="$out_dir/mlperf_log_accuracy.json" --num_proc 4
  } 2>&1 | tee "$out_dir/evaluator.log" "$out_dir/accuracy.txt"
  touch "$marker_dir/evaluate.ok"
}

if [[ -n "$from" ]]; then
  invalidate_from "$from"
fi

run_named() {
  case "$1" in
    environment) run_environment ;;
    preprocess) run_preprocess ;;
    run) run_gpu ;;
    evaluate) run_evaluate ;;
    *) usage ;;
  esac
}

if [[ -n "$stage" ]]; then
  run_named "$stage"
elif [[ -n "$from" ]]; then
  case "$from" in
    environment) run_environment; run_preprocess; run_gpu; run_evaluate ;;
    preprocess) run_preprocess; run_gpu; run_evaluate ;;
    run) run_gpu; run_evaluate ;;
    evaluate) run_evaluate ;;
    *) usage ;;
  esac
else
  run_environment
  run_preprocess
  run_gpu
  run_evaluate
fi
