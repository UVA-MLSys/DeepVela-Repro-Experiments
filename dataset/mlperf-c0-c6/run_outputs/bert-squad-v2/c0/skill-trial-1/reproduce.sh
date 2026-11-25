#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
export UV_OFFLINE=1
export UV_LINK_MODE=copy

TARGET=
STAGE=
FROM=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET=${2:-}; shift 2 ;;
    --stage) STAGE=${2:-}; shift 2 ;;
    --from) FROM=${2:-}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == bert-squad ]] || { echo "target must be bert-squad" >&2; exit 2; }
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "OUTPUT_DIR must be set" >&2; exit 2; }
[[ -d repo/language/bert && -d repo/loadgen ]] || { echo "run beside the fresh repo directory" >&2; exit 2; }

ROOT=$PWD
OUT=$(realpath -m "$OUTPUT_DIR")
[[ "$OUT" == "$ROOT"/* ]] || { echo "OUTPUT_DIR must be inside the current workspace" >&2; exit 2; }
WORK=$(dirname "$OUT")
ENV_DIR="$WORK/.venv"
BERT_DIR="$WORK/bert"
MARKERS="$WORK/.markers"
PY="$ENV_DIR/bin/python"
mkdir -p "$OUT" "$MARKERS"

stages=(setup preprocess run)
valid_stage() {
  local wanted=$1 item
  for item in "${stages[@]}"; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}

if [[ -n "$STAGE" && -n "$FROM" ]]; then
  echo "use only one of --stage and --from" >&2
  exit 2
fi
if [[ -n "$STAGE" ]]; then valid_stage "$STAGE" || { echo "invalid stage: $STAGE" >&2; exit 2; }; fi
if [[ -n "$FROM" ]]; then valid_stage "$FROM" || { echo "invalid stage: $FROM" >&2; exit 2; }; fi

if [[ -n "$FROM" ]]; then
  clearing=0
  for item in "${stages[@]}"; do
    [[ "$item" == "$FROM" ]] && clearing=1
    if (( clearing )); then
      rm -f "$MARKERS/$item.done"
      case "$item" in
        setup) rm -rf "$ENV_DIR" "$BERT_DIR" ;;
        preprocess) rm -f "$WORK/eval_features.pickle" ;;
        run)
          rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
            "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json" \
            "$OUT/accuracy.txt" "$OUT/predictions.json" "$OUT/run.log" \
            "$OUT/accuracy-evaluator.log" "$OUT/gpu-before-run.log"
          ;;
      esac
    fi
  done
fi

run_setup() {
  {
    echo "stage: setup"
    echo "python: python3.11"
    uv venv --python python3.11 "$ENV_DIR"
    uv pip install --python "$PY" \
      torch==2.4.1 transformers==4.44.2 numpy==1.26.4 \
      absl-py==2.1.0 six==1.17.0 boto3==1.43.121 pybind11==2.13.6 \
      setuptools==75.8.0 wheel==0.44.0
    uv pip install --python "$PY" --no-build-isolation "$ROOT/repo/loadgen"
    mkdir -p "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2" "$BERT_DIR/build/result"
    cp -a "$ROOT/repo/language/bert/." "$BERT_DIR/"
    rm -rf "$BERT_DIR/DeepLearningExamples"
    ln -sfn "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"
    ln -sfn "$SQUAD_DIR/dev-v1.1.json" "$BERT_DIR/build/data/dev-v1.1.json"
    ln -sfn "$MODEL_DIR/model.pytorch" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
    ln -sfn "$MODEL_DIR/vocab.txt" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
    "$PY" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
print("loadgen:", mlperf_loadgen.__file__)
PY
  } 2>&1 | tee "$OUT/setup.log"
  touch "$MARKERS/setup.done"
}

run_preprocess() {
  [[ -f "$MARKERS/setup.done" ]] || run_setup
  {
    echo "stage: preprocess"
    cd "$BERT_DIR"
    PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
      DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" VOCAB_FILE="$MODEL_DIR/vocab.txt" \
      "$PY" -c 'from squad_QSL import get_squad_QSL; q=get_squad_QSL(); print("feature_count:", q.count)'
    cp -f eval_features.pickle "$WORK/eval_features.pickle"
  } 2>&1 | tee "$OUT/preprocess.log"
  touch "$MARKERS/preprocess.done"
}

run_benchmark() {
  [[ -f "$MARKERS/preprocess.done" ]] || run_preprocess
  cp -f "$WORK/eval_features.pickle" "$BERT_DIR/eval_features.pickle"
  RUN_DIR="$WORK/.run-$BASHPID"
  mkdir -p "$RUN_DIR"
  nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader > "$RUN_DIR/gpu-before-run.log"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 10000 )); then
    echo "insufficient free GPU memory before run: ${free_mib:-unknown} MiB" | tee "$RUN_DIR/run.log" >&2
    exit 1
  fi
  max_args=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  {
    cat "$RUN_DIR/gpu-before-run.log"
    cd "$BERT_DIR"
    LOG_PATH="$RUN_DIR" DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
      VOCAB_FILE="$MODEL_DIR/vocab.txt" ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch" \
      SKIP_VERIFY_ACCURACY=1 "$PY" -c $'import sys\nimport pytorch_SUT\n_original = pytorch_SUT.get_pytorch_sut\ndef _with_device(args):\n    sut = _original(args)\n    print("pytorch device: " + str(next(sut.model.parameters()).device), flush=True)\n    return sut\npytorch_SUT.get_pytorch_sut = _with_device\nimport run\nrun.main()' \
      --backend=pytorch --accuracy "${max_args[@]}"
  } 2>&1 | tee "$RUN_DIR/run.log"
  {
    cd "$BERT_DIR"
    "$PY" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$RUN_DIR/mlperf_log_accuracy.json" \
      --out_file "$RUN_DIR/predictions.json" \
      --features_cache_file "$WORK/eval_features.pickle" \
      "${max_args[@]}"
  } 2>&1 | tee "$RUN_DIR/accuracy.txt" "$RUN_DIR/accuracy-evaluator.log"
  test -s "$RUN_DIR/mlperf_log_accuracy.json"
  test -s "$RUN_DIR/mlperf_log_detail.txt"
  test -s "$RUN_DIR/accuracy.txt"
  rg -q '^pytorch device: cuda:0$' "$RUN_DIR/run.log"
  for artifact in mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt \
    mlperf_log_trace.json predictions.json accuracy.txt accuracy-evaluator.log run.log \
    gpu-before-run.log; do
    [[ ! -e "$RUN_DIR/$artifact" ]] || cp -f "$RUN_DIR/$artifact" "$OUT/$artifact"
  done
  touch "$MARKERS/run.done"
}

run_one() {
  case "$1" in
    setup) [[ -f "$MARKERS/setup.done" ]] || run_setup ;;
    preprocess) [[ -f "$MARKERS/preprocess.done" ]] || run_preprocess ;;
    run) [[ -f "$MARKERS/run.done" ]] || run_benchmark ;;
  esac
}

if [[ -n "$STAGE" ]]; then
  run_one "$STAGE"
elif [[ -n "$FROM" ]]; then
  active=0
  for item in "${stages[@]}"; do
    [[ "$item" == "$FROM" ]] && active=1
    (( active )) && run_one "$item"
  done
else
  for item in "${stages[@]}"; do run_one "$item"; done
fi
