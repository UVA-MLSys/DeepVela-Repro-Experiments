#!/usr/bin/env bash
set -euo pipefail

TARGET=""
ONLY_STAGE=""
FROM_STAGE=""
while (($#)); do
  case "$1" in
    --target) TARGET="${2:?missing target id}"; shift 2 ;;
    --stage) ONLY_STAGE="${2:?missing stage}"; shift 2 ;;
    --from) FROM_STAGE="${2:?missing stage}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "unsupported target: ${TARGET:-<missing>}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER="2026-11-10"
export UV_OFFLINE=1

OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/$TARGET/out}"
mkdir -p "$OUTPUT_DIR"
TARGET_DIR="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)"
WORK_DIR="$TARGET_DIR/work"
VENV_DIR="$TARGET_DIR/venv"
BERT_DIR="$WORK_DIR/language/bert"

stages=(environment prepare run)
valid_stage() {
  local wanted="$1" item
  for item in "${stages[@]}"; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}
if [[ -n "$ONLY_STAGE" ]] && ! valid_stage "$ONLY_STAGE"; then echo "invalid stage: $ONLY_STAGE" >&2; exit 2; fi
if [[ -n "$FROM_STAGE" ]] && ! valid_stage "$FROM_STAGE"; then echo "invalid stage: $FROM_STAGE" >&2; exit 2; fi

if [[ -n "$FROM_STAGE" ]]; then
  removing=0
  for stage in "${stages[@]}"; do
    [[ "$stage" == "$FROM_STAGE" ]] && removing=1
    if ((removing)); then
      rm -f "$TARGET_DIR/.${stage}.success"
      case "$stage" in
        environment) rm -rf "$VENV_DIR" ;;
        prepare) rm -rf "$WORK_DIR" ;;
        run)
          rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
            "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/accuracy.txt" \
            "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/gpu-memory.log"
          ;;
      esac
    fi
  done
fi

run_environment() {
  local log="$OUTPUT_DIR/build.log"
  {
    echo "Creating Python 3.11 environment"
    uv venv "$VENV_DIR" --python 3.11
    uv pip install --offline --python "$VENV_DIR/bin/python" \
      "setuptools<81" wheel pybind11 absl-py six boto3 "numpy==1.26.4" \
      "torch==2.5.1" "transformers==4.44.2"
    mkdir -p "$WORK_DIR"
    cp -a "$REPO_DIR/loadgen" "$WORK_DIR/loadgen"
    uv pip install --offline --python "$VENV_DIR/bin/python" --no-build-isolation "$WORK_DIR/loadgen"
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
PY
  } 2>&1 | tee "$log"
  touch "$TARGET_DIR/.environment.success"
}

run_prepare() {
  local log="$OUTPUT_DIR/prepare.log"
  {
    rm -rf "$BERT_DIR"
    mkdir -p "$WORK_DIR/language"
    cp -a "$REPO_DIR/language/bert" "$BERT_DIR"
    rm -rf "$BERT_DIR/DeepLearningExamples"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"
    mkdir -p "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2" "$BERT_DIR/build/result"
    ln -s "$SQUAD_DIR/dev-v1.1.json" "$BERT_DIR/build/data/dev-v1.1.json"
    ln -s "$MODEL_DIR/model.pytorch" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
    ln -s "$MODEL_DIR/vocab.txt" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
    # Required observability line, added only to the target-local working copy.
    sed -i '/self.model.to(self.dev)/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$BERT_DIR/pytorch_SUT.py"
    cd "$BERT_DIR"
    PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
      DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" VOCAB_FILE="$MODEL_DIR/vocab.txt" \
      "$VENV_DIR/bin/python" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL(total_count_override=1, perf_count_override=1)
print("prepared_features", len(qsl.eval_features))
PY
    test -s eval_features.pickle
  } 2>&1 | tee "$log"
  touch "$TARGET_DIR/.prepare.success"
}

run_benchmark() {
  local max_args=()
  if [[ -n "${MAX_EXAMPLES:-}" ]]; then max_args=(--max_examples "$MAX_EXAMPLES"); fi
  local free_mib
  {
    echo "GPU memory immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
  } | tee "$OUTPUT_DIR/gpu-memory.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if ((free_mib < 8192)); then
    echo "insufficient free GPU memory: ${free_mib} MiB (need 8192 MiB)" | tee -a "$OUTPUT_DIR/gpu-memory.log" >&2
    return 1
  fi
  cd "$BERT_DIR"
  export LOG_PATH="$OUTPUT_DIR"
  export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
  export VOCAB_FILE="$MODEL_DIR/vocab.txt"
  export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
  export SKIP_VERIFY_ACCURACY=1
  "$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}" 2>&1 | tee "$OUTPUT_DIR/run.log"
  "$VENV_DIR/bin/python" accuracy-squad.py \
    --vocab_file "$MODEL_DIR/vocab.txt" \
    --val_data "$SQUAD_DIR/dev-v1.1.json" \
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --out_file "$BERT_DIR/build/result/predictions.json" \
    --features_cache_file "$BERT_DIR/eval_features.pickle" \
    "${max_args[@]}" 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  test -s "$OUTPUT_DIR/accuracy.txt"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  touch "$TARGET_DIR/.run.success"
}

run_one() {
  local stage="$1"
  local marker="$TARGET_DIR/.${stage}.success"
  if [[ -e "$marker" ]]; then echo "Skipping completed stage: $stage"; return; fi
  case "$stage" in
    environment) run_environment ;;
    prepare) test -e "$TARGET_DIR/.environment.success"; run_prepare ;;
    run) test -e "$TARGET_DIR/.prepare.success"; run_benchmark ;;
  esac
}

if [[ -n "$ONLY_STAGE" ]]; then
  run_one "$ONLY_STAGE"
else
  started=0
  [[ -z "$FROM_STAGE" ]] && started=1
  for stage in "${stages[@]}"; do
    [[ "$stage" == "$FROM_STAGE" ]] && started=1
    ((started)) && run_one "$stage"
  done
fi
