#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
UV_CACHE_DIR="$CACHE_ROOT/uv"
UV_EXCLUDE_NEWER=2026-11-10
export UV_CACHE_DIR UV_EXCLUDE_NEWER

TARGET=""
FROM_STAGE=""
ONLY_STAGE=""
MAX_EXAMPLES=""
while (($#)); do
  case "$1" in
    --target) TARGET=${2:?missing target id}; shift 2 ;;
    --from) FROM_STAGE=${2:?missing stage}; shift 2 ;;
    --stage) ONLY_STAGE=${2:?missing stage}; shift 2 ;;
    --max-examples) MAX_EXAMPLES=${2:?missing count}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "supported target: bert-squad" >&2
  exit 2
fi
if [[ ! -d repo/language/bert || ! -d repo/loadgen ]]; then
  echo "reproduce.sh must be run beside the fresh repository directory 'repo'" >&2
  exit 2
fi

OUTPUT_DIR=${OUTPUT_DIR:-"./$TARGET/out"}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
TARGET_ROOT=$(dirname "$OUTPUT_DIR")
WORK_DIR="$TARGET_ROOT/work"
VENV="$WORK_DIR/venv"
PYTHON="$VENV/bin/python"
ENV_MARKER="$WORK_DIR/.env.success"
PREPARE_MARKER="$WORK_DIR/.prepare.success"
RUN_MARKER="$OUTPUT_DIR/.run.success"

stages=(env prepare run)
stage_index() {
  case "$1" in env) echo 0;; prepare) echo 1;; run) echo 2;; *) return 1;; esac
}

if [[ -n "$FROM_STAGE" ]]; then
  from_index=$(stage_index "$FROM_STAGE") || { echo "unknown stage: $FROM_STAGE" >&2; exit 2; }
  if ((from_index <= 0)); then
    rm -rf "$WORK_DIR"
  elif ((from_index <= 1)); then
    rm -f "$PREPARE_MARKER"
    rm -rf "$WORK_DIR/bert/eval_features.pickle" "$WORK_DIR/bert/build"
  fi
  if ((from_index <= 2)); then
    rm -f "$RUN_MARKER" "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" \
      "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log"
  fi
fi

want_stage() {
  local name=$1 idx start=0
  idx=$(stage_index "$name")
  if [[ -n "$ONLY_STAGE" ]]; then
    [[ "$name" == "$ONLY_STAGE" ]]
    return
  fi
  if [[ -n "$FROM_STAGE" ]]; then start=$(stage_index "$FROM_STAGE"); fi
  ((idx >= start))
}

run_env() {
  [[ -f "$ENV_MARKER" ]] && return
  mkdir -p "$WORK_DIR"
  {
    echo "python: /home/mi3se/.local/bin/python3.11"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv --python /home/mi3se/.local/bin/python3.11 "$VENV"
    uv pip install --python "$PYTHON" \
      'torch==2.5.1' 'transformers==4.46.3' 'numpy<2' \
      absl-py six pybind11 'setuptools==80.9.0' boto3
    cp -a repo/loadgen "$WORK_DIR/loadgen-src"
    uv pip install --python "$PYTHON" "$WORK_DIR/loadgen-src"
    "$PYTHON" -c 'import torch, mlperf_loadgen; print("torch", torch.__version__, "cuda", torch.version.cuda)'
  } 2>&1 | tee "$OUTPUT_DIR/build.log"
  touch "$ENV_MARKER"
}

run_prepare() {
  [[ -f "$PREPARE_MARKER" ]] && return
  [[ -f "$ENV_MARKER" ]] || { echo "env stage has not succeeded" >&2; exit 1; }
  {
    rm -rf "$WORK_DIR/bert"
    cp -a repo/language/bert "$WORK_DIR/bert"
    rm -rf "$WORK_DIR/bert/DeepLearningExamples"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$WORK_DIR/bert/DeepLearningExamples"
    mkdir -p "$WORK_DIR/bert/build/result"
    cat > "$WORK_DIR/bert/pytorch_SUT.py" <<'PY'
import array
import os
import sys

import mlperf_loadgen as lg
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL


class BERT_PyTorch_SUT:
    def __init__(self, args):
        print("Constructing PyTorch SUT...")
        model_path = os.environ.get("ML_MODEL_FILE_WITH_PATH", "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        config = BertConfig.from_json_file(os.path.join(os.path.dirname(__file__), "bert_config.json"))
        self.model = BertForQuestionAnswering(config)
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        expected_unused = {"bert.pooler.dense.weight", "bert.pooler.dense.bias"}
        if incompatible.missing_keys or set(incompatible.unexpected_keys) != expected_unused:
            raise RuntimeError("checkpoint mismatch: {}".format(incompatible))
        self.model.eval()
        self.model.to(self.device)
        print("pytorch device: {}".format(next(self.model.parameters()).device))
        self.qsl = get_squad_QSL(total_count_override=args.max_examples)
        self.batch_size = int(os.environ.get("BERT_BATCH_SIZE", "32"))
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing PyTorch SUT.")

    def issue_queries(self, query_samples):
        for offset in range(0, len(query_samples), self.batch_size):
            samples = query_samples[offset:offset + self.batch_size]
            features = [self.qsl.get_features(sample.index) for sample in samples]
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long, device=self.device)
            token_type_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=self.device)
            with torch.no_grad():
                output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            if hasattr(output, "start_logits"):
                output = torch.stack((output.start_logits, output.end_logits), dim=-1)
            elif isinstance(output, (tuple, list)):
                output = torch.stack((output[0], output[1]), dim=-1)
            output = output.detach().float().cpu().numpy()
            responses = []
            buffers = []
            for sample, logits in zip(samples, output):
                buf = array.array("B", np.ascontiguousarray(logits, dtype=np.float32).tobytes())
                buffers.append(buf)
                address, length = buf.buffer_info()
                responses.append(lg.QuerySampleResponse(sample.id, address, length))
            lg.QuerySamplesComplete(responses)

    def flush_queries(self):
        pass


def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
PY
    cd "$WORK_DIR/bert"
    DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" VOCAB_FILE="$MODEL_DIR/vocab.txt" \
      "$PYTHON" - <<'PY'
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL(total_count_override=1)
print("feature_count:", len(qsl.eval_features))
PY
  } 2>&1 | tee "$OUTPUT_DIR/prepare.log"
  touch "$PREPARE_MARKER"
}

run_benchmark() {
  [[ -f "$PREPARE_MARKER" ]] || { echo "prepare stage has not succeeded" >&2; exit 1; }
  echo "GPU state immediately before run:" | tee "$OUTPUT_DIR/gpu.log"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu \
    --format=csv,noheader | tee -a "$OUTPUT_DIR/gpu.log"
  local max_args=()
  if [[ -n "$MAX_EXAMPLES" ]]; then max_args=(--max_examples "$MAX_EXAMPLES"); fi
  (
    cd "$WORK_DIR/bert"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export LOG_PATH="$OUTPUT_DIR"
    export SKIP_VERIFY_ACCURACY=1
    "$PYTHON" run.py --backend=pytorch --accuracy "${max_args[@]}"
  ) 2>&1 | tee "$OUTPUT_DIR/run.log"
  (
    cd "$WORK_DIR/bert"
    "$PYTHON" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --out_file build/result/predictions.json \
      "${max_args[@]}"
  ) 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  touch "$RUN_MARKER"
}

for stage in "${stages[@]}"; do
  want_stage "$stage" || continue
  case "$stage" in
    env) run_env ;;
    prepare) run_prepare ;;
    run) run_benchmark ;;
  esac
done
