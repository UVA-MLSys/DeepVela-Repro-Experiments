#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START_STAGE=""
ONE_STAGE=""
while (($#)); do
  case "$1" in
    --target) TARGET=${2:?missing target id}; shift 2 ;;
    --stage) ONE_STAGE=${2:?missing stage}; MODE=stage; shift 2 ;;
    --from) START_STAGE=${2:?missing stage}; MODE=from; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "supported target: bert-squad" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="$SCRIPT_DIR/repo"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DLE_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

VARIANT=${RUN_VARIANT:-full}
if [[ "$VARIANT" == "full" ]]; then
  STATE_DIR="$SCRIPT_DIR/$TARGET"
else
  STATE_DIR="$SCRIPT_DIR/$TARGET/$VARIANT"
fi
OUTPUT_DIR=${OUTPUT_DIR:-"$STATE_DIR/out"}
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SCRIPT_DIR/${OUTPUT_DIR#./}" ;;
esac
WORK_DIR="$STATE_DIR/work"
WORK_REPO="$WORK_DIR/repo"
VENV_DIR="$STATE_DIR/venv"
BERT_DIR="$WORK_REPO/language/bert"
PYTHON="$VENV_DIR/bin/python"
MAX_EXAMPLES=${MAX_EXAMPLES:-}

SETUP_MARKER="$STATE_DIR/.setup.success"
PREPROCESS_MARKER="$STATE_DIR/.preprocess.success"
RUN_MARKER="$OUTPUT_DIR/.run.success"
EVAL_MARKER="$OUTPUT_DIR/.evaluate.success"

stage_index() {
  case "$1" in setup) echo 0;; preprocess) echo 1;; run) echo 2;; evaluate) echo 3;; *) return 1;; esac
}

clear_from() {
  local stage=$1 idx
  idx=$(stage_index "$stage") || { echo "unknown stage: $stage" >&2; exit 2; }
  if ((idx <= 0)); then
    rm -rf -- "$WORK_DIR" "$VENV_DIR"
    rm -f -- "$SETUP_MARKER" "$PREPROCESS_MARKER"
  elif ((idx <= 1)); then
    rm -f -- "$PREPROCESS_MARKER" "$BERT_DIR/eval_features.pickle"
  fi
  if ((idx <= 2)); then
    rm -f -- "$RUN_MARKER" "$EVAL_MARKER"
    rm -f -- "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" \
      "$OUTPUT_DIR/mlperf_log_trace.json" "$OUTPUT_DIR/run.log" \
      "$OUTPUT_DIR/gpu-before.log" "$OUTPUT_DIR/accuracy.txt" \
      "$OUTPUT_DIR/evaluator.log"
  elif ((idx <= 3)); then
    rm -f -- "$EVAL_MARKER" "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  fi
}

write_repairs() {
  cat > "$BERT_DIR/file_utils.py" <<'PY'
import os

def cached_path(path, cache_dir=None, **kwargs):
    if os.path.exists(path):
        return path
    raise FileNotFoundError("only staged local inputs are allowed: {}".format(path))
PY

  cat > "$BERT_DIR/pytorch_SUT.py" <<'PY'
import array
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.path.dirname(__file__))

import mlperf_loadgen as lg
import numpy as np
import torch
from squad_QSL import get_squad_QSL
from transformers import BertConfig, BertForQuestionAnswering

class BERT_PyTorch_SUT:
    def __init__(self, args):
        print("Loading PyTorch model...")
        config_file = os.environ.get("BERT_CONFIG", "bert_config.json")
        model_file = os.environ.get(
            "ML_MODEL_FILE_WITH_PATH",
            "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch",
        )
        config = BertConfig.from_json_file(config_file)
        self.model = BertForQuestionAnswering(config)
        state = torch.load(model_file, map_location="cpu", weights_only=True)
        state.pop("bert.pooler.dense.weight")
        state.pop("bert.pooler.dense.bias")
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print("pytorch device: {}".format(next(self.model.parameters()).device))

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.qsl = get_squad_QSL(args.max_examples)
        print("Finished constructing SUT.")

    def issue_queries(self, query_samples):
        with torch.inference_mode():
            for sample in query_samples:
                feature = self.qsl.get_features(sample.index)
                input_ids = torch.tensor([feature.input_ids], dtype=torch.long, device=self.device)
                segment_ids = torch.tensor([feature.segment_ids], dtype=torch.long, device=self.device)
                input_mask = torch.tensor([feature.input_mask], dtype=torch.long, device=self.device)
                result = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
                start, end = result.start_logits, result.end_logits
                output = torch.stack((start, end), dim=-1).cpu().numpy().astype(np.float32, copy=False)[0]
                response_array = array.array("B", output.tobytes())
                address, length = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(sample.id, address, length)])

    def flush_queries(self):
        pass

def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
PY
}

setup_stage() {
  [[ -f "$SETUP_MARKER" ]] && return
  mkdir -p "$STATE_DIR" "$OUTPUT_DIR" "$WORK_DIR"
  exec > >(tee -a "$OUTPUT_DIR/setup.log") 2>&1
  echo "setup: copying pinned source and staged DeepLearningExamples"
  cp -a "$REPO_DIR" "$WORK_REPO"
  rm -rf -- "$BERT_DIR/DeepLearningExamples"
  mkdir -p "$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling"
  cp -a "$DLE_DIR/PyTorch/LanguageModeling/BERT" \
    "$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling/BERT"
  ln -sf "DeepLearningExamples/PyTorch/LanguageModeling/BERT/tokenization.py" \
    "$BERT_DIR/tokenization.py"
  mkdir -p "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2" "$BERT_DIR/build/result"
  ln -sf "$SQUAD_DIR/dev-v1.1.json" "$BERT_DIR/build/data/dev-v1.1.json"
  ln -sf "$MODEL_DIR/model.pytorch" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
  ln -sf "$MODEL_DIR/vocab.txt" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
  write_repairs

  UV_EXCLUDE_NEWER=2026-11-10 uv venv --python /apps/software/standard/core/ollama-python/0.6.1-py3.12/bin/python3.12 "$VENV_DIR"
  UV_EXCLUDE_NEWER=2026-11-10 uv pip install --offline --python "$PYTHON" \
    "torch==2.10.0" "numpy==1.26.4" "absl-py==2.5.1" "six==1.17.0" \
    "setuptools==69.5.1" "wheel==0.45.1" "pybind11==2.12.0" \
    "transformers==5.10.1" "boto3" "requests" "tqdm"
  UV_EXCLUDE_NEWER=2026-11-10 uv pip install --offline --python "$PYTHON" \
    --no-build-isolation "$WORK_REPO/loadgen"
  "$PYTHON" - <<'PY'
import mlperf_loadgen, numpy, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("loadgen import: ok")
PY
  touch "$SETUP_MARKER"
}

preprocess_stage() {
  [[ -f "$PREPROCESS_MARKER" ]] && return
  mkdir -p "$OUTPUT_DIR"
  exec > >(tee -a "$OUTPUT_DIR/preprocess.log") 2>&1
  write_repairs
  echo "preprocess: SQuAD v1.1 into max_seq_length=384 features"
  cd "$BERT_DIR"
  DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" VOCAB_FILE="$MODEL_DIR/vocab.txt" \
    "$PYTHON" - <<'PY'
from squad_QSL import get_squad_QSL
qsl = get_squad_QSL()
print("preprocessed feature count:", qsl.count)
PY
  touch "$PREPROCESS_MARKER"
}

run_stage() {
  [[ -f "$RUN_MARKER" ]] && return
  mkdir -p "$OUTPUT_DIR"
  write_repairs
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu \
    --format=csv,noheader | tee "$OUTPUT_DIR/gpu-before.log"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if ((free_mib < 6000)); then
    echo "run: only ${free_mib} MiB GPU memory free; need 6000 MiB" | tee "$OUTPUT_DIR/run.log" >&2
    exit 1
  fi
  echo "run: starting local A100 inference with ${free_mib} MiB free" | tee "$OUTPUT_DIR/run.log"
  cd "$BERT_DIR"
  local max_args=()
  [[ -n "$MAX_EXAMPLES" ]] && max_args=(--max_examples "$MAX_EXAMPLES")
  PATH="$VENV_DIR/bin:$PATH" LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 \
    DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" VOCAB_FILE="$MODEL_DIR/vocab.txt" \
    ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch" BERT_CONFIG="$BERT_DIR/bert_config.json" \
    "$PYTHON" run.py --backend=pytorch --accuracy "${max_args[@]}" 2>&1 | tee -a "$OUTPUT_DIR/run.log"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  touch "$RUN_MARKER"
}

evaluate_stage() {
  [[ -f "$EVAL_MARKER" ]] && return
  exec 3>>"$OUTPUT_DIR/evaluator.log"
  echo "evaluate: official accuracy-squad.py and evaluate_v1.1.py" | tee /dev/fd/3
  cd "$BERT_DIR"
  local max_args=()
  [[ -n "$MAX_EXAMPLES" ]] && max_args=(--max_examples "$MAX_EXAMPLES")
  PATH="$VENV_DIR/bin:$PATH" "$PYTHON" accuracy-squad.py \
    --vocab_file "$MODEL_DIR/vocab.txt" --val_data "$SQUAD_DIR/dev-v1.1.json" \
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --out_file "$BERT_DIR/build/result/predictions.json" \
    --features_cache_file "$BERT_DIR/eval_features.pickle" \
    "${max_args[@]}" 2>>"$OUTPUT_DIR/evaluator.log" | tee "$OUTPUT_DIR/accuracy.txt" >&3
  test -s "$OUTPUT_DIR/accuracy.txt"
  touch "$EVAL_MARKER"
}

stages=(setup preprocess run evaluate)
if [[ "$MODE" == "from" ]]; then
  clear_from "$START_STAGE"
fi
for stage in "${stages[@]}"; do
  if [[ "$MODE" == "stage" && "$stage" != "$ONE_STAGE" ]]; then continue; fi
  if [[ "$MODE" == "from" ]] && (( $(stage_index "$stage") < $(stage_index "$START_STAGE") )); then continue; fi
  "${stage}_stage"
done
