#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad [--stage build|prepare|run|evaluate] [--from build|prepare|run|evaluate]" >&2
  exit 2
}

target=""
only_stage=""
from_stage=""
while (($#)); do
  case "$1" in
    --target) target="${2:-}"; shift 2 ;;
    --stage) only_stage="${2:-}"; shift 2 ;;
    --from) from_stage="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$target" == "bert-squad" ]] || usage
[[ -z "$only_stage" || -z "$from_stage" ]] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$script_dir/repo"
cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
squad_dir="$cache_root/bert-squad/data"
model_dir="$cache_root/bert-squad/model"
dle_dir="$cache_root/bert-squad/DeepLearningExamples"
uv_cache="$cache_root/uv"
target_dir="$script_dir/$target"
out_dir="${OUTPUT_DIR:-$target_dir/out}"
case "$out_dir" in /*) ;; *) out_dir="$script_dir/${out_dir#./}" ;; esac
work_dir="$target_dir/work"
venv_dir="$target_dir/venv"
bert_work="$work_dir/bert"
loadgen_work="$work_dir/loadgen"

mkdir -p "$target_dir" "$out_dir" "$work_dir"

for required in \
  "$repo_dir/language/bert/run.py" \
  "$repo_dir/loadgen/setup.py" \
  "$squad_dir/dev-v1.1.json" \
  "$model_dir/model.pytorch" \
  "$model_dir/vocab.txt" \
  "$dle_dir/PyTorch/LanguageModeling/BERT/tokenization.py"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

stages=(build prepare run evaluate)
stage_index() {
  case "$1" in
    build) echo 0 ;; prepare) echo 1 ;; run) echo 2 ;; evaluate) echo 3 ;; *) usage ;;
  esac
}

if [[ -n "$from_stage" ]]; then
  start="$(stage_index "$from_stage")"
  if ((start == 0)); then
    rm -rf -- "$venv_dir" "$work_dir"
    mkdir -p "$work_dir"
  fi
  for ((i=start; i<${#stages[@]}; i++)); do
    rm -f -- "$out_dir/.${stages[$i]}.success"
  done
  if ((start <= 1)); then rm -f -- "$work_dir/eval_features.pickle"; fi
  if ((start <= 2)); then
    rm -f -- "$out_dir/mlperf_log_accuracy.json" "$out_dir/mlperf_log_detail.txt" \
      "$out_dir/mlperf_log_summary.txt" "$out_dir/mlperf_log_trace.json"
  fi
  if ((start <= 3)); then
    rm -f -- "$out_dir/accuracy.txt" "$work_dir/predictions.json"
  fi
fi

run_build() {
  local log="$out_dir/build.log"
  : >"$log"
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    rm -rf -- "$venv_dir"
    UV_CACHE_DIR="$uv_cache" UV_EXCLUDE_NEWER=2026-11-10 \
      uv venv --python "${PYTHON_BIN:-python3.11}" "$venv_dir" >>"$log" 2>&1
  fi
  UV_CACHE_DIR="$uv_cache" UV_EXCLUDE_NEWER=2026-11-10 \
    uv pip install --python "$venv_dir/bin/python" \
      'numpy==1.26.4' 'torch==2.5.1' 'transformers==4.37.2' \
      'absl-py==2.1.0' 'six==1.16.0' 'setuptools==75.6.0' 'boto3==1.40.73' \
      >>"$log" 2>&1

  rm -rf -- "$loadgen_work" "$bert_work"
  cp -a "$repo_dir/loadgen" "$loadgen_work"
  cp -a "$repo_dir/language/bert" "$bert_work"
  cp -a "$dle_dir/." "$bert_work/DeepLearningExamples/"

  "$venv_dir/bin/python" - "$bert_work/pytorch_SUT.py" >>"$log" 2>&1 <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
needle = "        self.model.load_state_dict(torch.load(model_file), strict=False)\n"
replacement = needle + "        print(\"pytorch device: {}\".format(next(self.model.parameters()).device))\n"
if needle not in s:
    raise SystemExit("could not apply device-log compatibility patch")
p.write_text(s.replace(needle, replacement, 1))
PY
  UV_CACHE_DIR="$uv_cache" UV_EXCLUDE_NEWER=2026-11-10 \
    uv pip install --python "$venv_dir/bin/python" "$loadgen_work" >>"$log" 2>&1
  "$venv_dir/bin/python" - <<'PY' >>"$log" 2>&1
import mlperf_loadgen, numpy, torch, transformers
print("mlperf_loadgen:", mlperf_loadgen.__file__)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
PY
  touch "$out_dir/.build.success"
}

run_prepare() {
  local log="$out_dir/prepare.log"
  : >"$log"
  (
    cd "$bert_work"
    PYTHONPATH="$bert_work/DeepLearningExamples/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
      DATASET_FILE="$squad_dir/dev-v1.1.json" VOCAB_FILE="$model_dir/vocab.txt" \
      "$venv_dir/bin/python" - "$work_dir/eval_features.pickle" <<'PY'
import sys
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL(cache_path=sys.argv[1])
print("prepared feature count:", qsl.count)
PY
  ) >>"$log" 2>&1
  [[ -s "$work_dir/eval_features.pickle" ]]
  touch "$out_dir/.prepare.success"
}

run_gpu() {
  local log="$out_dir/run.log"
  : >"$log"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used \
    --format=csv,noheader >>"$log" 2>&1
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || ((free_mib < 8192)); then
    echo "insufficient free GPU memory: ${free_mib:-unknown} MiB (need at least 8192 MiB)" | tee -a "$log" >&2
    exit 1
  fi
  local max_arg=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    max_arg=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  (
    cd "$bert_work"
    ln -sfn "$work_dir/eval_features.pickle" eval_features.pickle
    LOG_PATH="$out_dir" DATASET_FILE="$squad_dir/dev-v1.1.json" \
      VOCAB_FILE="$model_dir/vocab.txt" ML_MODEL_FILE_WITH_PATH="$model_dir/model.pytorch" \
      SKIP_VERIFY_ACCURACY=1 "$venv_dir/bin/python" run.py \
      --backend=pytorch --accuracy "${max_arg[@]}"
  ) >>"$log" 2>&1
  [[ -s "$out_dir/mlperf_log_accuracy.json" && -s "$out_dir/mlperf_log_detail.txt" ]]
  touch "$out_dir/.run.success"
}

run_evaluate() {
  local log="$out_dir/evaluator.log"
  local max_arg=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    max_arg=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  (
    cd "$bert_work"
    "$venv_dir/bin/python" accuracy-squad.py \
      --vocab_file "$model_dir/vocab.txt" \
      --val_data "$squad_dir/dev-v1.1.json" \
      --log_file "$out_dir/mlperf_log_accuracy.json" \
      --out_file "$work_dir/predictions.json" \
      --features_cache_file "$work_dir/eval_features.pickle" \
      "${max_arg[@]}"
  ) 2>&1 | tee "$out_dir/accuracy.txt" >"$log"
  [[ -s "$out_dir/accuracy.txt" ]]
  touch "$out_dir/.evaluate.success"
}

run_one() {
  local stage="$1"
  [[ ! -e "$out_dir/.$stage.success" ]] || return 0
  case "$stage" in
    build) run_build ;;
    prepare) run_prepare ;;
    run) run_gpu ;;
    evaluate) run_evaluate ;;
  esac
}

if [[ -n "$only_stage" ]]; then
  run_one "$only_stage"
elif [[ -n "$from_stage" ]]; then
  start="$(stage_index "$from_stage")"
  for ((i=start; i<${#stages[@]}; i++)); do run_one "${stages[$i]}"; done
else
  for stage in "${stages[@]}"; do run_one "$stage"; done
fi
