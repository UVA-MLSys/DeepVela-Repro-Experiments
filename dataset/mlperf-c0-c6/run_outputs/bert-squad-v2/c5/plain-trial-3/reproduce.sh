#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--target" || "${2:-}" != "bert-squad" || $# -ne 2 ]]; then
  echo "usage: $0 --target bert-squad" >&2
  exit 2
fi

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR="$ROOT_DIR/repo"
TARGET_DIR="$ROOT_DIR/bert-squad"
OUTPUT_DIR=${OUTPUT_DIR:-"$TARGET_DIR/out"}
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"
BUILD_LOG="$TARGET_DIR/build.log"
RUN_LOG="$TARGET_DIR/run.log"
EVAL_LOG="$TARGET_DIR/evaluator.log"
: >"$BUILD_LOG"
: >"$RUN_LOG"
: >"$EVAL_LOG"

for required in \
  "$REPO_DIR/language/bert/run.py" \
  "$REPO_DIR/loadgen/setup.py" \
  "$SQUAD_DIR/dev-v1.1.json" \
  "$MODEL_DIR/model.pytorch" \
  "$MODEL_DIR/vocab.txt" \
  "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/modeling.py"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required input: $required" | tee -a "$BUILD_LOG" >&2
    exit 1
  fi
done

PYTHON_DIR="$TARGET_DIR/python"
if ! UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" uv python find 3.12 --managed-python >/dev/null 2>&1; then
  UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" uv python install 3.12 --no-bin >>"$BUILD_LOG" 2>&1
fi
MANAGED_PY=$(UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" uv python find 3.12 --managed-python)
VENV="$TARGET_DIR/venv-managed"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python "$MANAGED_PY" "$VENV" >>"$BUILD_LOG" 2>&1
fi

uv pip install --python "$VENV/bin/python" \
  'torch==2.4.1' 'transformers==4.44.2' 'numpy<2' absl-py six \
  'setuptools==80.9.0' boto3 'pybind11==2.11.1' \
  >>"$BUILD_LOG" 2>&1
if ! "$VENV/bin/python" -c 'import mlperf_loadgen' >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" --no-build-isolation "$REPO_DIR/loadgen" \
    >>"$BUILD_LOG" 2>&1
fi

WORK_DIR="$TARGET_DIR/work"
BERT_DIR="$WORK_DIR/language/bert"
if [[ ! -f "$BERT_DIR/run.py" ]]; then
  mkdir -p "$WORK_DIR/language"
  cp -a "$REPO_DIR/language/bert" "$BERT_DIR"
fi
mkdir -p "$BERT_DIR/DeepLearningExamples"
ln -sfn "$DEEPLEARNING_EXAMPLES_DIR/PyTorch" "$BERT_DIR/DeepLearningExamples/PyTorch"
ln -sfn "$DEEPLEARNING_EXAMPLES_DIR/TensorFlow" "$BERT_DIR/DeepLearningExamples/TensorFlow"

cat >"$BERT_DIR/pytorch_SUT.py" <<'PY'
import array
import os
import sys

_bert_dir = os.path.dirname(os.path.abspath(__file__))
_dle_bert = os.path.join(_bert_dir, "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT")
sys.path.insert(0, _dle_bert)

import mlperf_loadgen as lg
import numpy as np
import torch

# The pinned NVIDIA model passes a historical boolean form of this argument.
_torch_gelu = torch.nn.functional.gelu
def _compatible_gelu(value, approximate="none"):
    if isinstance(approximate, bool):
        approximate = "tanh" if approximate else "none"
    return _torch_gelu(value, approximate=approximate)
torch.nn.functional.gelu = _compatible_gelu

from squad_QSL import get_squad_QSL
from modeling import BertConfig, BertForQuestionAnswering


class BERT_PyTorch_SUT:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this target")
        self.device = torch.device("cuda:0")
        config = BertConfig.from_json_file(os.path.join(_bert_dir, "bert_config.json"))
        self.model = BertForQuestionAnswering(config)
        state = torch.load(os.environ["ML_MODEL_FILE_WITH_PATH"], map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state and not any(k.startswith("bert.") for k in state):
            state = state["model"]
        renamed = {}
        for key, value in state.items():
            key = key.replace("intermediate.dense.", "intermediate.dense_act.")
            key = key.replace("pooler.dense.", "pooler.dense_act.")
            renamed[key] = value
        self.model.load_state_dict(renamed, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("pytorch device: {}".format(next(self.model.parameters()).device), flush=True)
        self.qsl = get_squad_QSL(args.max_examples)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        batch_size = int(os.environ.get("BERT_BATCH_SIZE", "8"))
        with torch.inference_mode():
            for start in range(0, len(query_samples), batch_size):
                samples = query_samples[start:start + batch_size]
                features = [self.qsl.get_features(sample.index) for sample in samples]
                input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=self.device)
                segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=self.device)
                input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long, device=self.device)
                logits = self.model(input_ids, segment_ids, input_mask)
                if isinstance(logits, (tuple, list)):
                    logits = torch.stack((logits[0], logits[1]), dim=-1)
                outputs = logits.detach().float().cpu().numpy()
                buffers = []
                responses = []
                for sample, output in zip(samples, outputs):
                    buf = array.array("B", np.ascontiguousarray(output, dtype=np.float32).tobytes())
                    buffers.append(buf)
                    addr, size = buf.buffer_info()
                    responses.append(lg.QuerySampleResponse(sample.id, addr, size))
                lg.QuerySamplesComplete(responses)

    def flush_queries(self):
        pass


def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
PY

LOG_DIR="$TARGET_DIR/loadgen-logs"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/mlperf_log_*.json "$LOG_DIR"/mlperf_log_*.txt

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits \
  | tee "$TARGET_DIR/gpu-before.log" >>"$RUN_LOG"
if (( FREE_MIB < 18000 )); then
  echo "insufficient free GPU memory: ${FREE_MIB} MiB (need 18000 MiB)" | tee -a "$RUN_LOG" >&2
  exit 1
fi
if (( FREE_MIB >= 50000 )); then
  export BERT_BATCH_SIZE=16
else
  export BERT_BATCH_SIZE=8
fi

export LOG_PATH="$LOG_DIR"
export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export SKIP_VERIFY_ACCURACY=1

(
  cd "$BERT_DIR"
  "$VENV/bin/python" run.py --backend=pytorch --accuracy
) >>"$RUN_LOG" 2>&1

cp "$LOG_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
cp "$LOG_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"

EVAL_STDOUT="$TARGET_DIR/evaluator.stdout.tmp"
(
  cd "$BERT_DIR"
  "$VENV/bin/python" accuracy-squad.py \
    --vocab_file "$MODEL_DIR/vocab.txt" \
    --val_data "$SQUAD_DIR/dev-v1.1.json" \
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --out_file "$TARGET_DIR/predictions.json" \
    --features_cache_file "$BERT_DIR/eval_features.pickle"
) >"$EVAL_STDOUT" 2>>"$EVAL_LOG"
tee "$OUTPUT_DIR/accuracy.txt" <"$EVAL_STDOUT" >>"$EVAL_LOG"
rm -f "$EVAL_STDOUT"
