#!/usr/bin/env bash
set -euo pipefail

target=""
while (($#)); do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { echo "--target requires an argument" >&2; exit 2; }
      target="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$target" != "bert-squad" ]]; then
  echo "unsupported target: ${target:-<none>}" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_repo="$root_dir/repo"
target_dir="$root_dir/$target"
work_repo="$target_dir/work/repo"
venv_dir="$target_dir/venv"
output_dir="${OUTPUT_DIR:-$target_dir/out}"

cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
squad_dir="$cache_root/bert-squad/data"
model_dir="$cache_root/bert-squad/model"
dle_dir="$cache_root/bert-squad/DeepLearningExamples"

export UV_CACHE_DIR="$cache_root/uv"
export UV_EXCLUDE_NEWER="2026-11-10"
export HF_HOME="$target_dir/huggingface"
export TORCH_HOME="$target_dir/torch"
export XDG_CACHE_HOME="$target_dir/xdg-cache"

mkdir -p "$output_dir" "$target_dir"
: > "$output_dir/build.log"
: > "$output_dir/run.log"
: > "$output_dir/evaluator.log"
: > "$output_dir/evaluator-stderr.log"

for required in \
  "$source_repo/language/bert/run.py" \
  "$source_repo/loadgen/setup.py" \
  "$squad_dir/dev-v1.1.json" \
  "$model_dir/model.pytorch" \
  "$model_dir/vocab.txt" \
  "$dle_dir/PyTorch/LanguageModeling/BERT/tokenization.py"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" | tee -a "$output_dir/build.log" >&2; exit 1; }
done

if [[ ! -x "$venv_dir/bin/python" ]]; then
  uv venv --python 3.12 "$venv_dir" 2>&1 | tee -a "$output_dir/build.log"
fi

uv pip install --python "$venv_dir/bin/python" \
  'torch==2.4.1' \
  'transformers==4.44.2' \
  'numpy==1.26.4' \
  'setuptools<81' \
  'absl-py' \
  'boto3' \
  'six' \
  'pybind11==2.11.1' \
  'wheel' \
  2>&1 | tee -a "$output_dir/build.log"

if [[ ! -d "$work_repo" ]]; then
  mkdir -p "$(dirname "$work_repo")"
  cp -a "$source_repo" "$work_repo"
fi

bert_dir="$work_repo/language/bert"
if [[ ! -L "$bert_dir/DeepLearningExamples" ]]; then
  if [[ -d "$bert_dir/DeepLearningExamples" ]]; then
    rmdir "$bert_dir/DeepLearningExamples"
  fi
  ln -s "$dle_dir" "$bert_dir/DeepLearningExamples"
fi

# The target requires this diagnostic prefix, absent from the pinned source.
if ! rg -q 'pytorch device: ' "$bert_dir/pytorch_SUT.py"; then
  python="$venv_dir/bin/python"
  "$python" - "$bert_dir/pytorch_SUT.py" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
needle = '        self.version = transformers.__version__\n'
replacement = needle + '        print("pytorch device: {}".format(self.dev))\n'
if needle not in text:
    raise SystemExit("could not locate device-selection insertion point")
path.write_text(text.replace(needle, replacement, 1))
PY
fi

uv pip install --python "$venv_dir/bin/python" "$work_repo/loadgen" \
  2>&1 | tee -a "$output_dir/build.log"

rm -f "$bert_dir/eval_features.pickle"
mkdir -p "$bert_dir/build/result"

export ML_MODEL_FILE_WITH_PATH="$model_dir/model.pytorch"
export VOCAB_FILE="$model_dir/vocab.txt"
export DATASET_FILE="$squad_dir/dev-v1.1.json"
export LOG_PATH="$output_dir"
export SKIP_VERIFY_ACCURACY=1

gpu_status="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader)"
printf 'GPU before run: %s\n' "$gpu_status" | tee -a "$output_dir/run.log"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 12000 )); then
  echo "insufficient free GPU memory: ${free_mib:-unknown} MiB (need 12000 MiB)" | tee -a "$output_dir/run.log" >&2
  exit 1
fi

(
  cd "$bert_dir"
  "$venv_dir/bin/python" run.py --backend=pytorch --accuracy
) 2>&1 | tee -a "$output_dir/run.log"

(
  cd "$bert_dir"
  "$venv_dir/bin/python" accuracy-squad.py \
    --vocab_file "$model_dir/vocab.txt" \
    --val_data "$squad_dir/dev-v1.1.json" \
    --log_file "$output_dir/mlperf_log_accuracy.json" \
    --out_file "$bert_dir/build/result/predictions.json" \
    --features_cache_file "$bert_dir/eval_features.pickle"
) 2> "$output_dir/evaluator-stderr.log" | tee "$output_dir/evaluator.log" "$output_dir/accuracy.txt"

for artifact in mlperf_log_accuracy.json mlperf_log_detail.txt accuracy.txt; do
  [[ -s "$output_dir/$artifact" ]] || { echo "missing or empty output: $output_dir/$artifact" >&2; exit 1; }
done
