#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT_DEFAULT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
CACHE_ROOT=${CACHE_ROOT:-$CACHE_ROOT_DEFAULT}
SQUAD_DIR=${SQUAD_DIR:-$CACHE_ROOT/bert-squad/data}
MODEL_DIR=${MODEL_DIR:-$CACHE_ROOT/bert-squad/model}
DEEPLEARNING_EXAMPLES_DIR=${DEEPLEARNING_EXAMPLES_DIR:-$CACHE_ROOT/bert-squad/DeepLearningExamples}
UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
UV_EXCLUDE_NEWER=${UV_EXCLUDE_NEWER:-2026-11-10}
export UV_CACHE_DIR UV_EXCLUDE_NEWER

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=
STAGE=
FROM=
while (($#)); do
  case "$1" in
    --target) TARGET=${2:-}; shift 2 ;;
    --stage) STAGE=${2:-}; shift 2 ;;
    --from) FROM=${2:-}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == bert-squad ]] || { echo "target must be bert-squad" >&2; exit 2; }
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "OUTPUT_DIR must be set" >&2; exit 2; }

TARGET_DIR="$SCRIPT_DIR/$TARGET"
WORK_DIR="$TARGET_DIR/work"
ENV_DIR="$TARGET_DIR/venv"
mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
ENV_MARKER="$TARGET_DIR/.environment.success"
PREP_MARKER="$TARGET_DIR/.prepare.success"
RUN_MARKER="$TARGET_DIR/.run.success"

clear_from() {
  case "$1" in
    environment)
      rm -rf "$ENV_DIR" "$WORK_DIR"
      rm -f "$ENV_MARKER" "$PREP_MARKER" "$RUN_MARKER"
      rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" "$OUTPUT_DIR/accuracy.txt"
      ;;
    prepare)
      rm -rf "$WORK_DIR"
      rm -f "$PREP_MARKER" "$RUN_MARKER"
      rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" "$OUTPUT_DIR/accuracy.txt"
      ;;
    run)
      rm -f "$RUN_MARKER"
      rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" "$OUTPUT_DIR/accuracy.txt"
      ;;
    *) echo "stage must be environment, prepare, or run" >&2; exit 2 ;;
  esac
}

stage_environment() {
  [[ -f "$ENV_MARKER" ]] && return
  {
    date -Ins
    echo "python base: /usr/bin/python3.11"
    uv venv --python /usr/bin/python3.11 "$ENV_DIR"
    uv pip install --python "$ENV_DIR/bin/python" \
      'torch==2.7.1' 'transformers==4.40.2' 'numpy==1.26.4' \
      'absl-py==2.3.1' 'six==1.17.0' 'setuptools==80.9.0' \
      'wheel==0.45.1' 'pybind11==2.13.6' 'boto3==1.44.0'
    # The host Python installations omit development headers. Install the
    # matching prebuilt wheel for the pinned source version instead.
    uv pip install --python "$ENV_DIR/bin/python" 'mlcommons-loadgen==6.0.17'
    "$ENV_DIR/bin/python" - <<'PY'
import torch, transformers, numpy, mlperf_loadgen
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
    date -Ins
  } 2>&1 | tee "$TARGET_DIR/build.log"
  touch "$ENV_MARKER"
}

stage_prepare() {
  [[ -f "$PREP_MARKER" ]] && return
  {
    date -Ins
    test -s "$SQUAD_DIR/dev-v1.1.json"
    test -s "$MODEL_DIR/model.pytorch"
    test -s "$MODEL_DIR/vocab.txt"
    test -s "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    mkdir -p "$WORK_DIR"
    cp -a "$SCRIPT_DIR/repo/." "$WORK_DIR/repo/"
    rmdir "$WORK_DIR/repo/language/bert/DeepLearningExamples"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$WORK_DIR/repo/language/bert/DeepLearningExamples"
    test -s "$WORK_DIR/repo/language/bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT/tokenization.py"
    "$ENV_DIR/bin/python" - "$WORK_DIR/repo/language/bert/pytorch_SUT.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
needle = '        self.version = transformers.__version__\n'
assert needle in s
s = s.replace(needle, needle + '        print("pytorch device: {}".format(self.dev))\n', 1)
p.write_text(s)
PY
    "$ENV_DIR/bin/python" - <<PY
import json
p = "$SQUAD_DIR/dev-v1.1.json"
d = json.load(open(p))
print("dataset version", d["version"])
print("articles", len(d["data"]))
print("questions", sum(len(p["qas"]) for a in d["data"] for p in a["paragraphs"]))
PY
    date -Ins
  } 2>&1 | tee "$TARGET_DIR/prepare.log"
  touch "$PREP_MARKER"
}

stage_run() {
  [[ -f "$RUN_MARKER" ]] && return
  local bert_dir="$WORK_DIR/repo/language/bert"
  local max_args=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  {
    date -Ins
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu --format=csv,noheader
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [[ "$free_mib" =~ ^[0-9]+$ ]]
    if (( free_mib < 8192 )); then
      echo "insufficient free GPU memory: ${free_mib} MiB (need 8192 MiB)" >&2
      exit 1
    fi
    echo "launch free GPU memory: ${free_mib} MiB"
  } 2>&1 | tee "$TARGET_DIR/gpu-check.log"

  (
    cd "$bert_dir"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export LOG_PATH="$OUTPUT_DIR"
    export SKIP_VERIFY_ACCURACY=1
    "$ENV_DIR/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
  ) 2>&1 | tee "$TARGET_DIR/run.log"

  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  (
    cd "$bert_dir"
    "$ENV_DIR/bin/python" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --out_file "$WORK_DIR/predictions.json" \
      --features_cache_file "$bert_dir/eval_features.pickle" \
      "${max_args[@]}"
  ) 2>&1 | tee "$TARGET_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  grep -q '^pytorch device: ' "$TARGET_DIR/run.log"
  grep '^{' "$OUTPUT_DIR/accuracy.txt" | tail -1 | "$ENV_DIR/bin/python" -c 'import json,sys; d=json.load(sys.stdin); assert set(d)=={"exact_match","f1"}; print(d)'
  touch "$RUN_MARKER"
}

if [[ -n "$FROM" ]]; then clear_from "$FROM"; fi
if [[ -n "$STAGE" ]]; then
  case "$STAGE" in
    environment) stage_environment ;;
    prepare) stage_environment; stage_prepare ;;
    run) stage_environment; stage_prepare; stage_run ;;
    *) echo "stage must be environment, prepare, or run" >&2; exit 2 ;;
  esac
else
  stage_environment
  stage_prepare
  stage_run
fi
