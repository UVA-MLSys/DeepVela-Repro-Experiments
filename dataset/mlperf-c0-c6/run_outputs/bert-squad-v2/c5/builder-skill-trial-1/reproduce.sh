#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

TARGET=""
ONLY_STAGE=""
FROM_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || { echo "--target requires a value" >&2; exit 2; }; TARGET=$2; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || { echo "--stage requires a value" >&2; exit 2; }; ONLY_STAGE=$2; shift 2 ;;
    --from) [[ $# -ge 2 ]] || { echo "--from requires a value" >&2; exit 2; }; FROM_STAGE=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$TARGET" == "bert-squad" ]] || { echo "supported target: bert-squad" >&2; exit 2; }
[[ -z "$ONLY_STAGE" || -z "$FROM_STAGE" ]] || { echo "--stage and --from are mutually exclusive" >&2; exit 2; }

STAGES=(env prepare preflight gpu evaluate)
valid_stage() {
  local candidate=$1 stage
  for stage in "${STAGES[@]}"; do [[ "$candidate" == "$stage" ]] && return 0; done
  return 1
}
[[ -z "$ONLY_STAGE" ]] || valid_stage "$ONLY_STAGE" || { echo "invalid stage: $ONLY_STAGE" >&2; exit 2; }
[[ -z "$FROM_STAGE" ]] || valid_stage "$FROM_STAGE" || { echo "invalid stage: $FROM_STAGE" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO="$ROOT/repo"
if [[ ! -d "$REPO/language/bert" && -d "$ROOT/language/bert" ]]; then REPO="$ROOT"; fi
[[ -f "$REPO/language/bert/run.py" ]] || {
  echo "fresh repository not found beside reproduce.sh (expected repo/language/bert/run.py or language/bert/run.py)" >&2
  exit 2
}

OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/$TARGET/out"}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
case "$OUTPUT_DIR/" in "$ROOT"/*) ;; *) echo "OUTPUT_DIR must be inside the workspace: $ROOT" >&2; exit 2 ;; esac

TARGET_DIR="$ROOT/$TARGET"
WORK_DIR="$TARGET_DIR/work"
SRC_DIR="$WORK_DIR/src"
APP_DIR="$SRC_DIR/language/bert"
VENV="$WORK_DIR/venv"
MARKER_DIR="$WORK_DIR/markers"
mkdir -p "$TARGET_DIR" "$MARKER_DIR"
marker() { printf '%s/%s.done' "$MARKER_DIR" "$1"; }

clear_from() {
  local requested=$1 seen=0 stage
  for stage in "${STAGES[@]}"; do
    [[ "$stage" == "$requested" ]] && seen=1
    [[ $seen -eq 0 ]] || rm -f -- "$(marker "$stage")"
  done
  case "$requested" in
    env)
      rm -rf -- "$WORK_DIR"
      mkdir -p "$MARKER_DIR"
      rm -f -- "$OUTPUT_DIR"/*.log "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" \
        "$OUTPUT_DIR/mlperf_log_trace.json" "$OUTPUT_DIR/accuracy.txt"
      ;;
    prepare)
      rm -f -- "$APP_DIR/eval_features.pickle"
      rm -rf -- "$WORK_DIR/preflight"
      rm -f -- "$OUTPUT_DIR"/prepare.log "$OUTPUT_DIR"/preflight-*.log \
        "$OUTPUT_DIR"/gpu-preflight.log "$OUTPUT_DIR"/run.log "$OUTPUT_DIR"/evaluator*.log \
        "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" "$OUTPUT_DIR/accuracy.txt"
      ;;
    preflight)
      rm -rf -- "$WORK_DIR/preflight"
      rm -f -- "$OUTPUT_DIR"/preflight-*.log "$OUTPUT_DIR"/gpu-preflight.log \
        "$OUTPUT_DIR"/run.log "$OUTPUT_DIR"/evaluator*.log "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" \
        "$OUTPUT_DIR/mlperf_log_trace.json" "$OUTPUT_DIR/accuracy.txt"
      ;;
    gpu)
      rm -f -- "$OUTPUT_DIR"/gpu-preflight.log "$OUTPUT_DIR"/run.log "$OUTPUT_DIR"/evaluator*.log \
        "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" "$OUTPUT_DIR/accuracy.txt"
      ;;
    evaluate) rm -f -- "$OUTPUT_DIR"/evaluator*.log "$OUTPUT_DIR/accuracy.txt" ;;
  esac
}
[[ -z "$FROM_STAGE" ]] || clear_from "$FROM_STAGE"

stage_env() {
  mkdir -p "$OUTPUT_DIR" "$SRC_DIR" "$MARKER_DIR"
  {
    echo "stage=env"
    date --iso-8601=seconds
    echo "source=$REPO"
    echo "source_commit_from_prompt=3fbc329939999c13d0a7b5e67fb2092287e06047"
    echo "dle_commit_from_prompt=729963dd47e7c8bd462ad10bfac7a7b0b604e6dd"
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    rm -rf -- "$SRC_DIR"
    mkdir -p "$SRC_DIR/language/bert"
    cp -a "$REPO/language/bert/." "$APP_DIR/"
    cp -a "$REPO/loadgen" "$SRC_DIR/loadgen"
    rm -rf -- "$APP_DIR/DeepLearningExamples"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$APP_DIR/DeepLearningExamples"
    # setup.py lists a dependency-only header absent from this pinned source snapshot.
    sed -i '/"trace_generator.h",/d' "$SRC_DIR/loadgen/setup.py"
    mkdir -p "$APP_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2" "$APP_DIR/build/result"
    ln -sf "$SQUAD_DIR/dev-v1.1.json" "$APP_DIR/build/data/dev-v1.1.json"
    ln -sf "$MODEL_DIR/model.pytorch" "$APP_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
    ln -sf "$MODEL_DIR/vocab.txt" "$APP_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"

    cat > "$APP_DIR/pytorch_SUT.py" <<'PY'
import array
import json
import os
import sys

_dle_bert = os.path.join(os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT")
sys.path.insert(0, _dle_bert)
sys.path.insert(0, os.path.dirname(__file__))

import mlperf_loadgen as lg
import numpy as np
import torch
from squad_QSL import get_squad_QSL
from transformers import BertConfig, BertForQuestionAnswering


class BERT_PyTorch_SUT:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this target")
        self.device = torch.device("cuda:0")
        print("pytorch device: {}".format(self.device), flush=True)
        config_path = os.environ.get("BERT_CONFIG_FILE", os.path.join(os.path.dirname(__file__), "bert_config.json"))
        model_path = os.environ.get("ML_MODEL_FILE_WITH_PATH", "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = BertConfig(**json.load(handle))
        self.model = BertForQuestionAnswering(config)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        incompat = self.model.load_state_dict(state, strict=False)
        allowed_unexpected = {"bert.pooler.dense.weight", "bert.pooler.dense.bias"}
        actual_unexpected = set(incompat.unexpected_keys) - allowed_unexpected
        if incompat.missing_keys or actual_unexpected:
            raise RuntimeError("checkpoint mismatch: missing={} unexpected={}".format(incompat.missing_keys, sorted(actual_unexpected)))
        if incompat.unexpected_keys:
            print("ignored unused checkpoint keys: {}".format(sorted(incompat.unexpected_keys)), flush=True)
        self.model.eval()
        self.model.to(self.device)
        self.batch_size = int(os.environ.get("BERT_BATCH_SIZE", "8"))
        print("pytorch batch size: {}".format(self.batch_size), flush=True)
        print("Constructing SUT...", flush=True)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        with torch.inference_mode():
            for offset in range(0, len(query_samples), self.batch_size):
                samples = query_samples[offset:offset + self.batch_size]
                features = [self.qsl.get_features(sample.index) for sample in samples]
                input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=self.device)
                attention_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long, device=self.device)
                token_type_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=self.device)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids, return_dict=True)
                logits = torch.stack((output.start_logits, output.end_logits), dim=-1)
                logits = logits.float().cpu().numpy().astype(np.float32, copy=False)
                buffers = []
                responses = []
                for sample, sample_logits in zip(samples, logits):
                    response_array = array.array("B", sample_logits.tobytes())
                    buffers.append(response_array)
                    address, length = response_array.buffer_info()
                    responses.append(lg.QuerySampleResponse(sample.id, address, length))
                lg.QuerySamplesComplete(responses)

    def flush_queries(self):
        pass


def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
PY

    rm -rf -- "$VENV"
    uv venv --managed-python --python 3.11 "$VENV"
    uv pip install --offline --python "$VENV/bin/python" \
      "torch==2.4.0" "transformers==4.31.0" "numpy==1.26.4" \
      "absl-py" boto3 "pybind11==2.11.1" "setuptools<81" wheel six
    uv pip install --offline --python "$VENV/bin/python" --no-build-isolation "$SRC_DIR/loadgen"
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen
import numpy
import torch
import transformers
print("python environment import check: ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
PY
  } 2>&1 | tee "$OUTPUT_DIR/build.log"
  touch "$(marker env)"
}

stage_prepare() {
  [[ -f "$(marker env)" ]] || { echo "env stage has not succeeded" >&2; exit 3; }
  {
    echo "stage=prepare"
    date --iso-8601=seconds
    cd "$APP_DIR"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}"
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen as lg
from squad_QSL import get_squad_QSL
qsl = get_squad_QSL()
print("feature_count={}".format(qsl.count))
lg.DestroyQSL(qsl.qsl)
PY
    test -s eval_features.pickle
  } 2>&1 | tee "$OUTPUT_DIR/prepare.log"
  touch "$(marker prepare)"
}

choose_batch() {
  local log_file=$1 free_mib
  {
    echo "gpu_preflight_time=$(date --iso-8601=seconds)"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader
  } >> "$log_file"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "cannot determine free GPU memory" >&2; return 1; }
  if (( free_mib >= 24576 )); then echo 16
  elif (( free_mib >= 12288 )); then echo 8
  elif (( free_mib >= 6144 )); then echo 1
  else echo "insufficient free GPU memory: ${free_mib} MiB" >&2; return 1
  fi
}

common_run_env() {
  export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
  export VOCAB_FILE="$MODEL_DIR/vocab.txt"
  export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
  export BERT_CONFIG_FILE="$APP_DIR/bert_config.json"
  export PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}"
  export SKIP_VERIFY_ACCURACY=1
}

stage_preflight() {
  [[ -f "$(marker prepare)" ]] || { echo "prepare stage has not succeeded" >&2; exit 3; }
  local preflight_dir="$WORK_DIR/preflight" batch
  rm -rf -- "$preflight_dir"
  mkdir -p "$preflight_dir/logs" "$preflight_dir/result"
  : > "$OUTPUT_DIR/preflight-gpu.log"
  batch=$(choose_batch "$OUTPUT_DIR/preflight-gpu.log")
  common_run_env
  export BERT_BATCH_SIZE=$batch
  export LOG_PATH="$preflight_dir/logs"
  {
    echo "stage=preflight"
    echo "bounded_examples=2"
    echo "selected_batch=$batch"
    cd "$APP_DIR"
    "$VENV/bin/python" run.py --backend=pytorch --accuracy --max_examples 2
  } 2>&1 | tee "$OUTPUT_DIR/preflight-run.log"
  {
    cd "$APP_DIR"
    "$VENV/bin/python" accuracy-squad.py --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" --log_file "$preflight_dir/logs/mlperf_log_accuracy.json" \
      --out_file "$preflight_dir/result/predictions.json" --features_cache_file "$APP_DIR/eval_features.pickle" \
      --max_examples 2
  } 2> "$OUTPUT_DIR/preflight-evaluator-stderr.log" | tee "$OUTPUT_DIR/preflight-evaluator.log"
  test -s "$preflight_dir/logs/mlperf_log_accuracy.json"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/preflight-run.log"
  grep -Eq '\{"exact_match": [0-9]' "$OUTPUT_DIR/preflight-evaluator.log"
  touch "$(marker preflight)"
}

stage_gpu() {
  [[ -f "$(marker preflight)" ]] || { echo "preflight stage has not succeeded" >&2; exit 3; }
  local batch
  : > "$OUTPUT_DIR/gpu-preflight.log"
  batch=$(choose_batch "$OUTPUT_DIR/gpu-preflight.log")
  common_run_env
  export BERT_BATCH_SIZE=$batch
  export LOG_PATH="$OUTPUT_DIR"
  {
    echo "stage=gpu"
    echo "full_validation=true"
    echo "selected_batch=$batch"
    cd "$APP_DIR"
    "$VENV/bin/python" run.py --backend=pytorch --accuracy
  } 2>&1 | tee "$OUTPUT_DIR/run.log"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  touch "$(marker gpu)"
}

stage_evaluate() {
  [[ -f "$(marker gpu)" ]] || { echo "gpu stage has not succeeded" >&2; exit 3; }
  common_run_env
  {
    cd "$APP_DIR"
    "$VENV/bin/python" accuracy-squad.py --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --out_file "$WORK_DIR/predictions.json" --features_cache_file "$APP_DIR/eval_features.pickle"
  } 2> "$OUTPUT_DIR/evaluator-stderr.log" | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  grep -Eq '\{"exact_match": [0-9]' "$OUTPUT_DIR/accuracy.txt"
  sha256sum "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
    "$OUTPUT_DIR/accuracy.txt" | tee "$OUTPUT_DIR/output-hashes.log"
  touch "$(marker evaluate)"
}

run_stage() {
  local stage=$1
  if [[ -f "$(marker "$stage")" ]]; then
    echo "stage $stage already complete; use --from $stage to rerun"
    return
  fi
  "stage_$stage"
}

if [[ -n "$ONLY_STAGE" ]]; then
  run_stage "$ONLY_STAGE"
elif [[ -n "$FROM_STAGE" ]]; then
  begin=0
  for stage in "${STAGES[@]}"; do
    [[ "$stage" == "$FROM_STAGE" ]] && begin=1
    [[ $begin -eq 0 ]] || run_stage "$stage"
  done
else
  for stage in "${STAGES[@]}"; do run_stage "$stage"; done
fi
