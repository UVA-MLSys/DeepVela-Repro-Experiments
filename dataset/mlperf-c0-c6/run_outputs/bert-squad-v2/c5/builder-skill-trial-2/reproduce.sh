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
for value in "$ONLY_STAGE" "$FROM_STAGE"; do
  [[ -z "$value" || "$value" == environment || "$value" == prepare || "$value" == run ]] || usage
done
[[ -z "$MAX_EXAMPLES" || "$MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]] || usage

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
  rm -f -- "$RUN_MARKER" \
    "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
    "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" \
    "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/gpu_memory.log" \
    "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" \
    "$HARNESS_DIR/predictions.json"
}

clear_prepare() {
  clear_run
  rm -f -- "$PREPARE_MARKER" "$OUTPUT_DIR/prepare.log"
  if [[ -d "$HARNESS_DIR" ]]; then rm -rf -- "$HARNESS_DIR"; fi
}

clear_environment() {
  clear_prepare
  rm -f -- "$ENV_MARKER" "$OUTPUT_DIR/build.log"
  if [[ -d "$VENV_DIR" ]]; then rm -rf -- "$VENV_DIR"; fi
}

if [[ -n "$FROM_STAGE" ]]; then
  case "$FROM_STAGE" in
    environment) clear_environment ;;
    prepare) clear_prepare ;;
    run) clear_run ;;
  esac
fi

stage_environment() {
  [[ ! -f "$ENV_MARKER" ]] || return 0
  mkdir -p "$WORK_DIR"
  {
    echo "environment: Python 3.10, PyTorch fp32 CUDA, LoadGen 6.0.17"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv "$VENV_DIR" --python 3.10
    uv pip install --offline --python "$VENV_DIR/bin/python" \
      'torch==2.7.1' 'numpy==1.26.4' 'six==1.17.0' 'absl-py==2.3.1' \
      'mlcommons-loadgen==6.0.17' 'boto3' 'requests' 'tqdm' 'setuptools==80.9.0'
    uv pip install --offline --python "$VENV_DIR/bin/python" 'transformers==4.57.1'
    "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__)
print("torch cuda runtime", torch.version.cuda)
print("numpy", numpy.__version__)
print("transformers", transformers.__version__)
print("mlperf_loadgen", mlperf_loadgen.__file__)
PY
  } >"$OUTPUT_DIR/build.log" 2>&1
  touch "$ENV_MARKER"
}

stage_prepare() {
  [[ -f "$ENV_MARKER" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
  [[ ! -f "$PREPARE_MARKER" ]] || return 0
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
  } >"$OUTPUT_DIR/prepare.log" 2>&1

  cat >"$HARNESS_DIR/pytorch_SUT.py" <<'PY'
import array
import os
import sys

import mlperf_loadgen as lg
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
        print("Loading PyTorch model...")
        checkpoint = torch.load(os.environ["ML_MODEL_FILE_WITH_PATH"], map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            config = BertConfig.from_json_file(os.path.join(os.path.dirname(__file__), "bert_config.json"))
            self.model = BertForQuestionAnswering(config)
            for key in ("bert.pooler.dense.weight", "bert.pooler.dense.bias"):
                if key not in checkpoint:
                    raise RuntimeError("checkpoint is missing expected unused tensor: " + key)
                checkpoint.pop(key)
            self.model.load_state_dict(checkpoint, strict=True)
        else:
            self.model = checkpoint
        self.model.eval().cuda()
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
                input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=self.device)
                segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=self.device)
                input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long, device=self.device)
                start, end = self.model(input_ids=input_ids, token_type_ids=segment_ids,
                                        attention_mask=input_mask, return_dict=False)
                outputs = torch.stack((start, end), dim=-1).float().cpu().numpy()
                buffers = [array.array("B", output.tobytes()) for output in outputs]
                responses = []
                for sample, response in zip(samples, buffers):
                    address, size = response.buffer_info()
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
  ) >>"$OUTPUT_DIR/prepare.log" 2>&1
  sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt" \
    >>"$OUTPUT_DIR/prepare.log"
  touch "$PREPARE_MARKER"
}

stage_run() {
  [[ -f "$PREPARE_MARKER" ]] || { echo "prepare stage has not succeeded" >&2; exit 1; }
  [[ ! -f "$RUN_MARKER" ]] || return 0
  clear_run
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader >"$OUTPUT_DIR/gpu_memory.log"
  cat "$OUTPUT_DIR/gpu_memory.log"
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "could not determine free GPU memory" >&2; exit 1; }
  (( free_mib >= 18000 )) || { echo "insufficient free GPU memory: ${free_mib} MiB; need 18000 MiB" >&2; exit 1; }

  export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
  export VOCAB_FILE="$MODEL_DIR/vocab.txt"
  export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
  export LOG_PATH="$OUTPUT_DIR"
  export SKIP_VERIFY_ACCURACY=1
  export BERT_BATCH_SIZE="${BERT_BATCH_SIZE:-16}"
  run_args=(--backend=pytorch --accuracy)
  eval_args=(--vocab_file "$VOCAB_FILE" --val_data "$DATASET_FILE"
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json"
    --out_file "$HARNESS_DIR/predictions.json"
    --features_cache_file "$HARNESS_DIR/eval_features.pickle")
  if [[ -n "$MAX_EXAMPLES" ]]; then
    run_args+=(--max_examples "$MAX_EXAMPLES")
    eval_args+=(--max_examples "$MAX_EXAMPLES")
  fi
  (cd "$HARNESS_DIR" && "$VENV_DIR/bin/python" run.py "${run_args[@]}") \
    >"$OUTPUT_DIR/run.log" 2>&1
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
result = None
for line in reversed([x.strip() for x in open(sys.argv[1], encoding="utf-8") if x.strip()]):
    try:
        candidate = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(candidate, dict) and "exact_match" in candidate and "f1" in candidate:
        result = candidate
        break
if result is None:
    raise SystemExit("evaluator stdout contains no metric JSON")
for key in ("exact_match", "f1"):
    if not math.isfinite(float(result[key])):
        raise SystemExit("non-finite metric: " + key)
print(result)
PY
  touch "$RUN_MARKER"
}

run_stage() {
  case "$1" in
    environment) stage_environment ;;
    prepare) stage_prepare ;;
    run) stage_run ;;
    *) usage ;;
  esac
}

if [[ -n "$ONLY_STAGE" ]]; then
  run_stage "$ONLY_STAGE"
elif [[ -n "$FROM_STAGE" ]]; then
  start=0
  for stage in environment prepare run; do
    [[ "$stage" == "$FROM_STAGE" ]] && start=1
    (( start == 0 )) || run_stage "$stage"
  done
else
  for stage in environment prepare run; do run_stage "$stage"; done
fi
