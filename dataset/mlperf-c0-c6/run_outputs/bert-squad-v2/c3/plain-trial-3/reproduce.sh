#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad" >&2
  exit 2
}

target=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || usage
      target="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done
[[ "$target" == "bert-squad" ]] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$script_dir/repo"
cache_root="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
squad_dir="$cache_root/bert-squad/data"
model_dir="$cache_root/bert-squad/model"
dle_dir="$cache_root/bert-squad/DeepLearningExamples"
uv_cache_dir="$cache_root/uv"

[[ -d "$repo_dir/language/bert" ]] || { echo "missing repository beside reproduce.sh: $repo_dir" >&2; exit 1; }
[[ -f "$squad_dir/dev-v1.1.json" ]] || { echo "missing SQuAD input" >&2; exit 1; }
[[ -f "$model_dir/model.pytorch" && -f "$model_dir/vocab.txt" ]] || { echo "missing model inputs" >&2; exit 1; }
[[ -d "$dle_dir/PyTorch/LanguageModeling/BERT" ]] || { echo "missing DeepLearningExamples input" >&2; exit 1; }

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$script_dir/$target/out"
elif [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
fi
target_dir="$(dirname "$OUTPUT_DIR")"
work_dir="$target_dir/work"
venv_dir="$target_dir/.venv"
mkdir -p "$target_dir" "$OUTPUT_DIR"

export UV_CACHE_DIR="$uv_cache_dir"
export UV_EXCLUDE_NEWER="2026-11-10"
export UV_NO_PROGRESS=1

{
  echo "Creating Python 3.11 environment"
  uv venv --python python3.11 --clear "$venv_dir"
  uv pip install --python "$venv_dir/bin/python" --offline \
    'torch==2.4.1' 'transformers==4.31.0' 'numpy==1.26.4' \
    'absl-py==2.1.0' 'six==1.16.0' 'boto3==1.44.0' 'pybind11==2.11.1' \
    'setuptools==69.5.1'
  uv pip install --python "$venv_dir/bin/python" --offline "$repo_dir/loadgen"
} >"$target_dir/build.log" 2>&1

rm -rf "$work_dir"
mkdir -p "$work_dir"
cp -a "$repo_dir/language/bert" "$work_dir/bert"
rmdir "$work_dir/bert/DeepLearningExamples"
ln -s "$dle_dir" "$work_dir/bert/DeepLearningExamples"

# The target requires this diagnostic prefix, which the pinned harness lacks.
# Patch only the disposable work copy, leaving the supplied source untouched.
sed -i '/self.version = transformers.__version__/a\        print("pytorch device: {}".format(self.dev))' \
  "$work_dir/bert/pytorch_SUT.py"

bert_dir="$work_dir/bert"
export ML_MODEL_FILE_WITH_PATH="$model_dir/model.pytorch"
export VOCAB_FILE="$model_dir/vocab.txt"
export DATASET_FILE="$squad_dir/dev-v1.1.json"
export LOG_PATH="$OUTPUT_DIR"
export SKIP_VERIFY_ACCURACY=1
export PYTHONUNBUFFERED=1

# Record immediately before the sole GPU run so sharing is auditable.
{
  date -Is
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits
} >"$target_dir/gpu-before.log" 2>&1

(
  cd "$bert_dir"
  "$venv_dir/bin/python" run.py --backend=pytorch --accuracy
) >"$target_dir/run.log" 2>&1

(
  cd "$bert_dir"
  mkdir -p result
  "$venv_dir/bin/python" accuracy-squad.py \
    --vocab_file "$model_dir/vocab.txt" \
    --val_data "$squad_dir/dev-v1.1.json" \
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --out_file "$bert_dir/result/predictions.json" \
    --features_cache_file "$bert_dir/eval_features.pickle"
) > >(tee "$target_dir/evaluator.log" "$OUTPUT_DIR/accuracy.txt") \
  2> >(tee "$target_dir/evaluator-stderr.log" >&2)

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
