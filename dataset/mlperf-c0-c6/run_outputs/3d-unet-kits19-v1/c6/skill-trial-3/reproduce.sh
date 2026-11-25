#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
KITS19_RAW_DIR="${CACHE_ROOT}/3d-unet-kits19/kits19_raw_data/kits19/data"
MODEL_DIR="${CACHE_ROOT}/3d-unet-kits19/model"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_EXCLUDE_NEWER="2026-11-10"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/repo"
TARGET=""
MODE="all"
START_STAGE=""

usage() {
  echo "usage: $0 --target 3d-unet [--stage setup|preprocess|run] [--from setup|preprocess|run]" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE="stage"; START_STAGE="${2:-}"; shift 2 ;;
    --from) MODE="from"; START_STAGE="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "${TARGET}" == "3d-unet" ]] || usage
[[ -d "${REPO_DIR}/vision/medical_imaging/3d-unet-kits19" ]] || { echo "repository not found beside reproduce.sh" >&2; exit 1; }
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
[[ "${OUTPUT_DIR}" != "/" ]] || { echo "unsafe OUTPUT_DIR" >&2; exit 1; }

TARGET_DIR="${SCRIPT_DIR}/${TARGET}"
WORK_DIR="${REPRO_WORK_DIR:-${TARGET_DIR}/work}"
WORK_DIR="$(mkdir -p "${WORK_DIR}" && cd "${WORK_DIR}" && pwd)"
MARKER_DIR="${WORK_DIR}/markers"
VENV_DIR="${WORK_DIR}/venv"
BENCH_DIR="${WORK_DIR}/bench"
LOADGEN_DIR="${WORK_DIR}/loadgen"
RAW_LINKS="${WORK_DIR}/raw_data"
PREPROCESSED_DIR="${WORK_DIR}/preprocessed_data"
POSTPROCESSED_DIR="${WORK_DIR}/postprocessed_data"
MODEL_PATH="${MODEL_DIR}/3dunet_kits19_pytorch.ptc"
STAGES=(setup preprocess run)

stage_index() {
  case "$1" in setup) echo 0 ;; preprocess) echo 1 ;; run) echo 2 ;; *) usage ;; esac
}

if [[ "${MODE}" != "all" ]]; then
  stage_index "${START_STAGE}" >/dev/null
fi

clear_from() {
  local start="$1" idx i
  idx="$(stage_index "${start}")"
  if ((idx == 0)); then
    rm -rf "${WORK_DIR}"
    mkdir -p "${MARKER_DIR}"
    rm -f "${OUTPUT_DIR}/mlperf_log_accuracy.json" "${OUTPUT_DIR}/mlperf_log_detail.txt" \
      "${OUTPUT_DIR}/mlperf_log_summary.txt" "${OUTPUT_DIR}/mlperf_log_trace.json" \
      "${OUTPUT_DIR}/accuracy.txt" "${OUTPUT_DIR}/build.log" "${OUTPUT_DIR}/preprocess.log" \
      "${OUTPUT_DIR}/run.log" "${OUTPUT_DIR}/evaluator.log" "${OUTPUT_DIR}/gpu-check.log"
  else
    mkdir -p "${MARKER_DIR}"
    for ((i=idx; i<${#STAGES[@]}; i++)); do rm -f "${MARKER_DIR}/${STAGES[i]}.ok"; done
    if ((idx <= 1)); then rm -rf "${PREPROCESSED_DIR}"; fi
    if ((idx <= 2)); then
      rm -rf "${POSTPROCESSED_DIR}"
      rm -f "${OUTPUT_DIR}/mlperf_log_accuracy.json" "${OUTPUT_DIR}/mlperf_log_detail.txt" \
        "${OUTPUT_DIR}/mlperf_log_summary.txt" "${OUTPUT_DIR}/accuracy.txt" "${OUTPUT_DIR}/run.log" \
        "${OUTPUT_DIR}/evaluator.log" "${OUTPUT_DIR}/gpu-check.log"
    fi
  fi
}

run_setup() {
  mkdir -p "${MARKER_DIR}" "${OUTPUT_DIR}"
  rm -rf "${VENV_DIR}" "${BENCH_DIR}" "${LOADGEN_DIR}"
  {
    echo "Building Python environment"
    uv venv --python 3.11 "${VENV_DIR}"
    uv pip install --python "${VENV_DIR}/bin/python" --offline \
      numpy==1.26.4 scipy==1.10.1 nibabel==5.3.2 pandas==1.5.3 \
      torch==2.4.1 pybind11==2.12.0 setuptools wheel
    cp -a "${REPO_DIR}/loadgen" "${LOADGEN_DIR}"
    uv pip install --python "${VENV_DIR}/bin/python" --offline --no-build-isolation "${LOADGEN_DIR}"
    cp -a "${REPO_DIR}/vision/medical_imaging/3d-unet-kits19" "${BENCH_DIR}"
    cp "${REPO_DIR}/loadgen/mlperf.conf" "${WORK_DIR}/mlperf.conf"
    [[ -s "${MODEL_PATH}" ]]
    "${VENV_DIR}/bin/python" -c 'import mlperf_loadgen, nibabel, numpy, pandas, scipy, torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'
  } 2>&1 | tee "${OUTPUT_DIR}/build.log"

  if [[ -n "${PREFLIGHT_CASES:-}" ]]; then
    PREFLIGHT_CASES="${PREFLIGHT_CASES}" "${VENV_DIR}/bin/python" - "${BENCH_DIR}" <<'PY'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1]) / "meta"
wanted = [x for x in os.environ["PREFLIGHT_CASES"].split(",") if x]
manifest = json.loads((root / "inference_cases.json").read_text())
assert wanted and set(wanted) <= set(manifest), (wanted, manifest)
(root / "inference_cases.json").write_text(json.dumps(wanted, indent=4) + "\n")
checks = json.loads((root / "checksum_inference.json").read_text())
names = {f"{case}.pkl" for case in wanted}
(root / "checksum_inference.json").write_text(json.dumps({k: v for k, v in checks.items() if k in names}, indent=4, sort_keys=True) + "\n")
PY
  fi
  touch "${MARKER_DIR}/setup.ok"
}

run_preprocess() {
  [[ -f "${MARKER_DIR}/setup.ok" ]] || { echo "setup stage has not succeeded" >&2; exit 1; }
  mkdir -p "${RAW_LINKS}" "${PREPROCESSED_DIR}"
  "${VENV_DIR}/bin/python" - "${BENCH_DIR}/meta/inference_cases.json" "${RAW_LINKS}" "${KITS19_RAW_DIR}" <<'PY'
import json, os, pathlib, sys
manifest, destination, source = map(pathlib.Path, sys.argv[1:])
for case in json.loads(manifest.read_text()):
    origin = source / ("case_00185" if case == "case_00400" else case)
    assert (origin / "imaging.nii.gz").is_file(), origin
    assert (origin / "segmentation.nii.gz").is_file(), origin
    link = destination / case
    if not link.exists():
        os.symlink(origin, link, target_is_directory=True)
PY
  {
    cd "${BENCH_DIR}"
    "${VENV_DIR}/bin/python" preprocess.py --raw_data_dir "${RAW_LINKS}" \
      --results_dir "${PREPROCESSED_DIR}" --mode preprocess --num_proc "${PREPROCESS_PROCS:-4}"
    "${VENV_DIR}/bin/python" preprocess.py --raw_data_dir "${RAW_LINKS}" --results_dir "${PREPROCESSED_DIR}" \
      --mode verify --num_proc "${PREPROCESS_PROCS:-4}"
  } 2>&1 | tee "${OUTPUT_DIR}/preprocess.log"
  touch "${MARKER_DIR}/preprocess.ok"
}

run_gpu() {
  [[ -f "${MARKER_DIR}/preprocess.ok" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  mkdir -p "${POSTPROCESSED_DIR}"
  {
    echo "GPU state immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  } 2>&1 | tee "${OUTPUT_DIR}/gpu-check.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  [[ "${free_mib}" =~ ^[0-9]+$ ]] && ((free_mib >= 16000)) || { echo "less than 16000 MiB GPU memory free" >&2; exit 1; }

  {
    cd "${BENCH_DIR}"
    "${VENV_DIR}/bin/python" - "${MODEL_PATH}" <<'PY'
import sys, torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(sys.argv[1], map_location=device)
print("pytorch device: ", next(model.parameters()).device, flush=True)
del model
torch.cuda.empty_cache()
PY
    export LOG_PATH="${OUTPUT_DIR}"
    export SKIP_VERIFY_ACCURACY=1
    "${VENV_DIR}/bin/python" run.py --backend=pytorch --scenario=Offline --accuracy \
      --model="${MODEL_PATH}" --preprocessed_data_dir="${PREPROCESSED_DIR}" \
      --mlperf_conf="${WORK_DIR}/mlperf.conf" --user_conf="${BENCH_DIR}/user.conf"
  } 2>&1 | tee "${OUTPUT_DIR}/run.log"

  {
    cd "${BENCH_DIR}"
    "${VENV_DIR}/bin/python" accuracy_kits.py --log_file="${OUTPUT_DIR}/mlperf_log_accuracy.json" \
      --preprocessed_data_dir="${PREPROCESSED_DIR}" --postprocessed_data_dir="${POSTPROCESSED_DIR}" \
      --num_proc="${EVALUATOR_PROCS:-4}"
  } 2>&1 | tee "${OUTPUT_DIR}/accuracy.txt" | tee "${OUTPUT_DIR}/evaluator.log"
  [[ -s "${OUTPUT_DIR}/mlperf_log_accuracy.json" && -s "${OUTPUT_DIR}/mlperf_log_detail.txt" && -s "${OUTPUT_DIR}/accuracy.txt" ]]
  touch "${MARKER_DIR}/run.ok"
}

if [[ "${MODE}" == "from" ]]; then clear_from "${START_STAGE}"; fi
if [[ "${MODE}" == "all" && ! -d "${WORK_DIR}" ]]; then mkdir -p "${MARKER_DIR}"; fi

for stage in "${STAGES[@]}"; do
  if [[ "${MODE}" == "stage" && "${stage}" != "${START_STAGE}" ]]; then continue; fi
  if [[ "${MODE}" == "from" ]] && (( $(stage_index "${stage}") < $(stage_index "${START_STAGE}") )); then continue; fi
  if [[ -f "${MARKER_DIR}/${stage}.ok" ]]; then echo "Skipping completed stage: ${stage}"; continue; fi
  case "${stage}" in setup) run_setup ;; preprocess) run_preprocess ;; run) run_gpu ;; esac
done
