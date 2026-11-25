#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

target=""
stage=""
from=""
while (($#)); do
  case "$1" in
    --target) target=${2:?missing target id}; shift 2 ;;
    --stage) stage=${2:?missing stage}; shift 2 ;;
    --from) from=${2:?missing stage}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$target" != bert-squad ]]; then
  echo "unsupported target: ${target:-<none>}" >&2
  exit 2
fi
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

repo_root=$(cd "$(dirname "$0")" && pwd)
work_root="$repo_root/$target/work"
venv="$work_root/venv"
src="$work_root/bert"
mkdir -p "$OUTPUT_DIR" "$work_root"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

stages=(build preprocess run)
run_stage() {
  local wanted=$1
  [[ -z "$stage" || "$stage" == "$wanted" ]] || return 0
  if [[ -n "$from" ]]; then
    local found=0 s
    for s in "${stages[@]}"; do
      [[ "$s" == "$from" ]] && found=1
      [[ "$s" == "$wanted" && $found -eq 0 ]] && return 0
    done
  fi
  return 1
}

if run_stage build; then :; else
  rm -f "$work_root/.build.ok" "$work_root/.preprocess.ok" "$work_root/.run.ok"
  rm -rf "$venv" "$src"
  mkdir -p "$src"
  {
    echo "Building isolated Python environment"
    uv venv --python /usr/bin/python3.11 "$venv"
    uv pip install --python "$venv/bin/python" \
      'torch==2.4.1' 'transformers==4.44.2' 'numpy==1.26.4' \
      'absl-py==2.1.0' 'six==1.16.0' 'setuptools==75.6.0' \
      'mlcommons-loadgen==6.0.17' 'boto3==1.40.70'
    cp -a "$repo_root/repo/language/bert/." "$src/"
    cp -a "$DEEPLEARNING_EXAMPLES_DIR/." "$src/DeepLearningExamples/"
    "$venv/bin/python" - <<'PY' "$src/pytorch_SUT.py"
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
needle = "        self.model.load_state_dict(torch.load(model_file), strict=False)\n"
replacement = needle + '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n'
if needle not in s:
    raise SystemExit("device logging patch point not found")
p.write_text(s.replace(needle, replacement, 1))
PY
    "$venv/bin/python" - <<'PY'
from importlib.metadata import version
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("loadgen", version("mlcommons-loadgen"))
PY
  } 2>&1 | tee "$OUTPUT_DIR/build.log"
  touch "$work_root/.build.ok"
fi

if run_stage preprocess; then :; else
  [[ -f "$work_root/.build.ok" ]] || { echo "build stage has not succeeded" >&2; exit 1; }
  rm -f "$work_root/.preprocess.ok" "$work_root/.run.ok"
  rm -f "$src/eval_features.pickle"
  {
    echo "Preprocessing SQuAD v1.1 validation set"
    cd "$src"
    PYTHONPATH="$src/DeepLearningExamples/PyTorch/LanguageModeling/BERT:$src" \
    DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
    VOCAB_FILE="$MODEL_DIR/vocab.txt" \
    "$venv/bin/python" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL()
print("preprocessed_features", qsl.count)
PY
  } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  touch "$work_root/.preprocess.ok"
fi

if run_stage run; then :; else
  [[ -f "$work_root/.preprocess.ok" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
  rm -f "$work_root/.run.ok"
  rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" \
        "$OUTPUT_DIR/accuracy.txt" "$work_root/predictions.json"
  run_args=(--backend=pytorch --accuracy)
  eval_args=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    run_args+=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
    eval_args+=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  {
    echo "GPU memory immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    cd "$src"
    DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
    VOCAB_FILE="$MODEL_DIR/vocab.txt" \
    ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch" \
    LOG_PATH="$OUTPUT_DIR" \
    SKIP_VERIFY_ACCURACY=1 \
    "$venv/bin/python" run.py "${run_args[@]}"
  } 2>&1 | tee "$OUTPUT_DIR/run.log"
  {
    cd "$src"
    "$venv/bin/python" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --features_cache_file "$src/eval_features.pickle" \
      --out_file "$work_root/predictions.json" \
      "${eval_args[@]}"
  } 2> >(tee "$OUTPUT_DIR/evaluator-stderr.log" >&2) \
    | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  touch "$work_root/.run.ok"
fi
