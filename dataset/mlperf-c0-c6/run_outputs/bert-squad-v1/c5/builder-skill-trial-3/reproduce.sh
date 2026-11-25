#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START_STAGE="environment"
STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) MODE="stage"; STAGE="${2:-}"; shift 2 ;;
    --from) MODE="from"; START_STAGE="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "unsupported target: ${TARGET:-<missing>}" >&2
  exit 2
fi

for required in CACHE_ROOT SQUAD_DIR MODEL_DIR DEEPLEARNING_EXAMPLES_DIR OUTPUT_DIR; do
  if [[ -z "${!required:-}" ]]; then
    echo "required environment variable is unset: $required" >&2
    exit 2
  fi
done

case "$OUTPUT_DIR" in
  ""|/|"$PWD") echo "unsafe OUTPUT_DIR: $OUTPUT_DIR" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="$SCRIPT_DIR/repo"
TARGET_DIR="$SCRIPT_DIR/$TARGET"
WORK_DIR="$TARGET_DIR/work"
SRC_DIR="$WORK_DIR/src"
VENV_DIR="$WORK_DIR/venv"
MARKER_DIR="$WORK_DIR/markers"
FEATURE_CACHE="$WORK_DIR/eval_features.pickle"
PREDICTIONS="$WORK_DIR/predictions.json"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
export PYTHONUNBUFFERED=1

if [[ ! -d "$SOURCE_REPO/language/bert" ]]; then
  echo "fresh repository copy not found at $SOURCE_REPO" >&2
  exit 2
fi
for path in "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" \
            "$MODEL_DIR/vocab.txt" "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/modeling.py"; do
  if [[ ! -r "$path" ]]; then
    echo "required input is not readable: $path" >&2
    exit 2
  fi
done

stages=(environment preprocess run evaluate)
stage_index() {
  local wanted="$1" i
  for i in "${!stages[@]}"; do
    [[ "${stages[$i]}" == "$wanted" ]] && { echo "$i"; return 0; }
  done
  return 1
}

clear_from() {
  local first="$1" index i name
  index="$(stage_index "$first")" || { echo "unknown stage: $first" >&2; exit 2; }
  mkdir -p "$OUTPUT_DIR" "$MARKER_DIR"
  for ((i=index; i<${#stages[@]}; i++)); do
    name="${stages[$i]}"
    rm -f "$MARKER_DIR/$name.ok"
  done
  if (( index == 0 )); then
    rm -rf "$WORK_DIR"
    mkdir -p "$MARKER_DIR"
    rm -f "$OUTPUT_DIR/environment.log" "$OUTPUT_DIR/preprocess.log" \
      "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" \
      "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" \
      "$OUTPUT_DIR/accuracy.txt"
  elif (( index <= 1 )); then
    rm -f "$FEATURE_CACHE" "$PREDICTIONS" "$OUTPUT_DIR/preprocess.log" \
      "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" \
      "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" \
      "$OUTPUT_DIR/accuracy.txt"
  elif (( index <= 2 )); then
    rm -f "$PREDICTIONS" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" \
      "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" \
      "$OUTPUT_DIR/accuracy.txt"
  else
    rm -f "$PREDICTIONS" "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  fi
}

write_sut() {
  local sut_path="$SRC_DIR/language/bert/pytorch_SUT.py"
  if [[ -f "$sut_path" ]]; then
    return
  fi
  cat > "$sut_path" <<'PY'
import array
import json
import os
import sys

import mlperf_loadgen as lg
import numpy as np
import torch
from transformers import BertConfig, BertForQuestionAnswering

sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())
from squad_QSL import SQuAD_v1_QSL


class BERT_PyTorch_SUT:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this target")
        self.device = torch.device("cuda:0")
        self.batch_size = int(os.environ.get("BERT_BATCH_SIZE", "16"))
        with open("bert_config.json") as config_file:
            config = BertConfig(**json.load(config_file))
        self.model = BertForQuestionAnswering(config)
        model_file = os.environ["ML_MODEL_FILE_WITH_PATH"]
        state = torch.load(model_file, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        incompatible = self.model.load_state_dict(state, strict=False)
        allowed_extra = {"bert.pooler.dense.weight", "bert.pooler.dense.bias"}
        bad_extra = set(incompatible.unexpected_keys) - allowed_extra
        if incompatible.missing_keys or bad_extra:
            raise RuntimeError(
                "checkpoint mismatch: missing=%r unexpected=%r"
                % (incompatible.missing_keys, sorted(bad_extra))
            )
        print("ignored unused checkpoint keys: %s" % sorted(set(incompatible.unexpected_keys)), flush=True)
        self.model.eval().to(self.device)
        print("pytorch device: %s" % next(self.model.parameters()).device, flush=True)
        print("pytorch batch size: %d" % self.batch_size, flush=True)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.qsl = SQuAD_v1_QSL(
            args.max_examples,
            args.max_examples,
            cache_path=os.environ["FEATURE_CACHE_FILE"],
        )

    def issue_queries(self, query_samples):
        for offset in range(0, len(query_samples), self.batch_size):
            queries = query_samples[offset:offset + self.batch_size]
            features = [self.qsl.get_features(query.index) for query in queries]
            input_ids = torch.tensor(
                [feature.input_ids for feature in features], dtype=torch.long, device=self.device
            )
            input_mask = torch.tensor(
                [feature.input_mask for feature in features], dtype=torch.long, device=self.device
            )
            segment_ids = torch.tensor(
                [feature.segment_ids for feature in features], dtype=torch.long, device=self.device
            )
            with torch.inference_mode():
                result = self.model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                start_scores, end_scores = result.start_logits, result.end_logits
                output = torch.stack((start_scores, end_scores), dim=-1).float().cpu().numpy()
            for query, logits in zip(queries, output):
                response_array = array.array("B", logits.astype(np.float32, copy=False).tobytes())
                address, length = response_array.buffer_info()
                response = lg.QuerySampleResponse(query.id, address, length)
                lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass


def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
PY
}

run_environment() {
  [[ -f "$MARKER_DIR/environment.ok" ]] && return
  mkdir -p "$OUTPUT_DIR" "$WORK_DIR" "$MARKER_DIR"
  {
    date -Is
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv --version
    uv venv --python 3.11 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" \
      "torch==2.4.0" "transformers==4.44.2" "numpy==1.26.4" \
      "absl-py==2.5.0" "six==1.17.0" "pybind11==2.11.1" \
      "setuptools==69.5.1" "boto3==1.40.61" wheel
    cp -a "$SOURCE_REPO" "$SRC_DIR"
    rm -rf "$SRC_DIR/language/bert/DeepLearningExamples"
    cp -a "$DEEPLEARNING_EXAMPLES_DIR" "$SRC_DIR/language/bert/DeepLearningExamples"
    write_sut
    uv pip install --python "$VENV_DIR/bin/python" "$SRC_DIR/loadgen"
    "$VENV_DIR/bin/python" - <<'PY'
import torch, transformers, mlperf_loadgen
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("loadgen import ok", mlperf_loadgen.__file__)
PY
    date -Is
  } 2>&1 | tee "$OUTPUT_DIR/environment.log"
  touch "$MARKER_DIR/environment.ok"
}

run_preprocess() {
  [[ -f "$MARKER_DIR/preprocess.ok" ]] && return
  [[ -f "$MARKER_DIR/environment.ok" ]] || run_environment
  {
    date -Is
    cd "$SRC_DIR/language/bert"
    PYTHONPATH="$SRC_DIR/language/bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT" \
      VOCAB_FILE="$MODEL_DIR/vocab.txt" DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
      "$VENV_DIR/bin/python" - "$FEATURE_CACHE" <<'PY'
import os
import sys
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL(cache_path=sys.argv[1])
print("preprocessed feature count:", qsl.count)
PY
    date -Is
  } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  [[ -s "$FEATURE_CACHE" ]]
  touch "$MARKER_DIR/preprocess.ok"
}

run_inference() {
  [[ -f "$MARKER_DIR/run.ok" ]] && return
  [[ -f "$MARKER_DIR/preprocess.ok" ]] || run_preprocess
  local max_args=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  {
    date -Is
    echo "GPU state immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=pid,used_memory,name \
      --format=csv,noheader,nounits || true
    cd "$SRC_DIR/language/bert"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export FEATURE_CACHE_FILE="$FEATURE_CACHE"
    export LOG_PATH="$OUTPUT_DIR"
    export SKIP_VERIFY_ACCURACY=1
    export MLPERF_SCRATCH_PATH="$WORK_DIR"
    "$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy \
      --user_conf user.conf "${max_args[@]}"
    date -Is
  } 2>&1 | tee "$OUTPUT_DIR/run.log"
  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" ]]
  [[ -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  touch "$MARKER_DIR/run.ok"
}

run_evaluate() {
  [[ -f "$MARKER_DIR/evaluate.ok" ]] && return
  [[ -f "$MARKER_DIR/run.ok" ]] || run_inference
  local max_args=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  {
    date -Is
    cd "$SRC_DIR/language/bert"
    "$VENV_DIR/bin/python" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --out_file "$PREDICTIONS" \
      --features_cache_file "$FEATURE_CACHE" \
      "${max_args[@]}" > >(tee "$OUTPUT_DIR/accuracy.txt")
    date -Is
  } > >(tee "$OUTPUT_DIR/evaluator.log") 2>&1
  "$VENV_DIR/bin/python" - "$OUTPUT_DIR/accuracy.txt" <<'PY'
import json
import math
import sys
lines = [line.strip() for line in open(sys.argv[1]) if line.strip()]
result = json.loads(next(line for line in reversed(lines) if line.startswith("{")))
assert set(result) == {"exact_match", "f1"}
assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in result.values())
print("validated evaluator result", result)
PY
  touch "$MARKER_DIR/evaluate.ok"
}

run_named_stage() {
  case "$1" in
    environment) run_environment ;;
    preprocess) run_preprocess ;;
    run) run_inference ;;
    evaluate) run_evaluate ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}

mkdir -p "$OUTPUT_DIR"
if [[ "$MODE" == "stage" ]]; then
  stage_index "$STAGE" >/dev/null || { echo "unknown stage: $STAGE" >&2; exit 2; }
  run_named_stage "$STAGE"
else
  if [[ "$MODE" == "from" ]]; then
    clear_from "$START_STAGE"
  fi
  start=0
  if [[ "$MODE" == "from" ]]; then
    start="$(stage_index "$START_STAGE")"
  fi
  for ((i=start; i<${#stages[@]}; i++)); do
    run_named_stage "${stages[$i]}"
  done
fi
