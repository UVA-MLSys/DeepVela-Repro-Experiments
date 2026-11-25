#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad [--stage environment|prepare|run] [--from environment|prepare|run] [--max-examples N]" >&2
  exit 2
}

TARGET=""
ONLY_STAGE=""
FROM_STAGE=""
MAX_EXAMPLES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; TARGET="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; ONLY_STAGE="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; FROM_STAGE="$2"; shift 2 ;;
    --max-examples) [[ $# -ge 2 ]] || usage; MAX_EXAMPLES="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$TARGET" == "bert-squad" ]] || usage
[[ -z "$ONLY_STAGE" || -z "$FROM_STAGE" ]] || usage
for stage_arg in "$ONLY_STAGE" "$FROM_STAGE"; do
  [[ -z "$stage_arg" || "$stage_arg" == environment || "$stage_arg" == prepare || "$stage_arg" == run ]] || usage
done
if [[ -n "$MAX_EXAMPLES" ]]; then
  [[ "$MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]] || usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
CACHE_ROOT="/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache"
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export SQUAD_DIR MODEL_DIR DEEPLEARNING_EXAMPLES_DIR
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER="2026-11-10"

TARGET_DIR="$SCRIPT_DIR/$TARGET"
OUTPUT_DIR="${OUTPUT_DIR:-$TARGET_DIR/out}"
WORK_DIR="$TARGET_DIR/work"
VENV_DIR="$WORK_DIR/venv"
HARNESS_DIR="$WORK_DIR/harness"
ENV_MARKER="$TARGET_DIR/.environment.success"
PREPARE_MARKER="$TARGET_DIR/.prepare.success"
RUN_MARKER="$TARGET_DIR/.run.success"

mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"

clear_run() {
  rm -f "$RUN_MARKER"
  rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" \
        "$OUTPUT_DIR/mlperf_log_trace.json" \
        "$OUTPUT_DIR/accuracy.txt" \
        "$OUTPUT_DIR/gpu_memory.log" \
        "$OUTPUT_DIR/run.log" \
        "$OUTPUT_DIR/evaluator.log"
  rm -f "$HARNESS_DIR/predictions.json"
}

clear_prepare() {
  clear_run
  rm -f "$PREPARE_MARKER"
  if [[ -d "$HARNESS_DIR" ]]; then
    rm -rf -- "$HARNESS_DIR"
  fi
  rm -f "$OUTPUT_DIR/prepare.log"
}

clear_environment() {
  clear_prepare
  rm -f "$ENV_MARKER"
  if [[ -d "$VENV_DIR" ]]; then
    rm -rf -- "$VENV_DIR"
  fi
  rm -f "$OUTPUT_DIR/build.log"
}

if [[ -n "$FROM_STAGE" ]]; then
  case "$FROM_STAGE" in
    environment) clear_environment ;;
    prepare) clear_prepare ;;
    run) clear_run ;;
  esac
fi

stage_environment() {
  mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
  if [[ -f "$ENV_MARKER" ]]; then
    return
  fi
  {
    echo "environment: python 3.10, PyTorch fp32 CUDA, matching LoadGen 6.0.17"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv "$VENV_DIR" --python 3.10
    uv pip install --python "$VENV_DIR/bin/python" \
      'torch==2.7.1' 'numpy==1.26.4' 'six==1.17.0' 'absl-py==2.3.1' \
      'mlcommons-loadgen==6.0.17' 'boto3' 'requests' 'tqdm' 'setuptools==80.9.0'
    uv pip install --python "$VENV_DIR/bin/python" 'transformers==4.57.1'
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch
print("torch", torch.__version__)
print("torch cuda runtime", torch.version.cuda)
print("numpy", numpy.__version__)
print("mlperf_loadgen", mlperf_loadgen.__file__)
PY
  } >"$OUTPUT_DIR/build.log" 2>&1
  touch "$ENV_MARKER"
}

stage_prepare() {
  [[ -f "$ENV_MARKER" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
  if [[ -f "$PREPARE_MARKER" ]]; then
    return
  fi
  {
    for required in \
      "$REPO_DIR/language/bert/run.py" \
      "$REPO_DIR/language/bert/accuracy-squad.py" \
      "$SQUAD_DIR/dev-v1.1.json" \
      "$MODEL_DIR/model.pytorch" \
      "$MODEL_DIR/vocab.txt" \
      "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/modeling.py"; do
      [[ -f "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
    done
    mkdir -p "$HARNESS_DIR"
    cp -a "$REPO_DIR/language/bert/." "$HARNESS_DIR/"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$HARNESS_DIR/DeepLearningExamples"
    cat >"$HARNESS_DIR/pytorch_SUT.py" <<'PY'
import array
import os
import sys

import mlperf_loadgen as lg
import numpy as np
import torch
from transformers import BertConfig, BertForQuestionAnswering

_bert_dir = os.path.join(
    os.environ["DEEPLEARNING_EXAMPLES_DIR"],
    "PyTorch", "LanguageModeling", "BERT"
)
if _bert_dir not in sys.path:
    sys.path.insert(0, _bert_dir)

from squad_QSL import get_squad_QSL


class BERT_PyTorch_SUT:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this target")
        model_path = os.environ["ML_MODEL_FILE_WITH_PATH"]
        print("Loading PyTorch model...")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            config = BertConfig.from_json_file(os.path.join(os.path.dirname(__file__), "bert_config.json"))
            self.model = BertForQuestionAnswering(config)
            for unused_pooler_key in ("bert.pooler.dense.weight", "bert.pooler.dense.bias"):
                if unused_pooler_key not in checkpoint:
                    raise RuntimeError("checkpoint is missing expected unused pooler tensor: " + unused_pooler_key)
                checkpoint.pop(unused_pooler_key)
            self.model.load_state_dict(checkpoint, strict=True)
        else:
            self.model = checkpoint
        self.model.eval()
        self.model.cuda()
        self.device = next(self.model.parameters()).device
        print("pytorch device: {}".format(self.device))
        self.batch_size = int(os.environ.get("BERT_BATCH_SIZE", "16"))
        self.qsl = get_squad_QSL(args.max_examples)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        with torch.inference_mode():
            for offset in range(0, len(query_samples), self.batch_size):
                samples = query_samples[offset:offset + self.batch_size]
                features = [self.qsl.get_features(sample.index) for sample in samples]
                input_ids = torch.tensor(
                    [feature.input_ids for feature in features], dtype=torch.long, device=self.device
                )
                segment_ids = torch.tensor(
                    [feature.segment_ids for feature in features], dtype=torch.long, device=self.device
                )
                input_mask = torch.tensor(
                    [feature.input_mask for feature in features], dtype=torch.long, device=self.device
                )
                start_logits, end_logits = self.model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    return_dict=False,
                )
                outputs = torch.stack((start_logits, end_logits), dim=-1).float().cpu().numpy()
                response_arrays = [array.array("B", output.tobytes()) for output in outputs]
                responses = []
                for sample, response_array in zip(samples, response_arrays):
                    address, size = response_array.buffer_info()
                    responses.append(lg.QuerySampleResponse(sample.id, address, size))
                lg.QuerySamplesComplete(responses)

    def flush_queries(self):
        pass


def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
PY
    (
      cd "$HARNESS_DIR"
      DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
      VOCAB_FILE="$MODEL_DIR/vocab.txt" \
      PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
      "$VENV_DIR/bin/python" - <<'PY'
from squad_QSL import get_squad_QSL
qsl = get_squad_QSL()
print("preprocessed features", len(qsl.eval_features))
del qsl
PY
    )
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    "$VENV_DIR/bin/python" - <<'PY'
import json, os
p = os.path.join(os.environ["SQUAD_DIR"], "dev-v1.1.json")
d = json.load(open(p, encoding="utf-8"))
print("dataset version", d["version"])
print("articles", len(d["data"]))
print("questions", sum(len(p["qas"]) for a in d["data"] for p in a["paragraphs"]))
PY
  } >"$OUTPUT_DIR/prepare.log" 2>&1
  touch "$PREPARE_MARKER"
}

stage_run() {
  [[ -f "$PREPARE_MARKER" ]] || { echo "prepare stage has not succeeded" >&2; exit 1; }
  if [[ -f "$RUN_MARKER" ]]; then
    return
  fi
  clear_run
  mkdir -p "$OUTPUT_DIR"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader >"$OUTPUT_DIR/gpu_memory.log"
  cat "$OUTPUT_DIR/gpu_memory.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "could not determine free GPU memory" >&2; exit 1; }
  if (( free_mib < 18000 )); then
    echo "insufficient free GPU memory: ${free_mib} MiB; need at least 18000 MiB" >&2
    exit 1
  fi

  export SQUAD_DIR MODEL_DIR DEEPLEARNING_EXAMPLES_DIR
  export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
  export VOCAB_FILE="$MODEL_DIR/vocab.txt"
  export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
  export LOG_PATH="$OUTPUT_DIR"
  export SKIP_VERIFY_ACCURACY=1
  export BERT_BATCH_SIZE="${BERT_BATCH_SIZE:-16}"
  run_args=(--backend=pytorch --accuracy)
  eval_args=(
    --vocab_file "$VOCAB_FILE"
    --val_data "$DATASET_FILE"
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json"
    --out_file "$HARNESS_DIR/predictions.json"
    --features_cache_file "$HARNESS_DIR/eval_features.pickle"
  )
  if [[ -n "$MAX_EXAMPLES" ]]; then
    run_args+=(--max_examples "$MAX_EXAMPLES")
    eval_args+=(--max_examples "$MAX_EXAMPLES")
  fi
  (
    cd "$HARNESS_DIR"
    "$VENV_DIR/bin/python" run.py "${run_args[@]}"
  ) >"$OUTPUT_DIR/run.log" 2>&1
  (
    cd "$HARNESS_DIR"
    PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
      "$VENV_DIR/bin/python" -u accuracy-squad.py "${eval_args[@]}"
  ) >"$OUTPUT_DIR/accuracy.txt" 2>"$OUTPUT_DIR/evaluator.log"
  cat "$OUTPUT_DIR/accuracy.txt" >>"$OUTPUT_DIR/evaluator.log"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  test -s "$OUTPUT_DIR/accuracy.txt"
  "$VENV_DIR/bin/python" - "$OUTPUT_DIR/accuracy.txt" <<'PY'
import json, math, sys
lines = [line.strip() for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
result = None
for line in reversed(lines):
    try:
        candidate = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(candidate, dict) and "exact_match" in candidate and "f1" in candidate:
        result = candidate
        break
if result is None:
    raise SystemExit("evaluator stdout contains no metric JSON object")
for key in ("exact_match", "f1"):
    if key not in result or not math.isfinite(float(result[key])):
        raise SystemExit("missing or non-finite metric: " + key)
print(result)
PY
  touch "$RUN_MARKER"
}

run_named_stage() {
  case "$1" in
    environment) stage_environment ;;
    prepare) stage_prepare ;;
    run) stage_run ;;
    *) usage ;;
  esac
}

if [[ -n "$ONLY_STAGE" ]]; then
  run_named_stage "$ONLY_STAGE"
elif [[ -n "$FROM_STAGE" ]]; then
  start=0
  for stage in environment prepare run; do
    [[ "$stage" == "$FROM_STAGE" ]] && start=1
    (( start == 0 )) || run_named_stage "$stage"
  done
else
  for stage in environment prepare run; do
    run_named_stage "$stage"
  done
fi
