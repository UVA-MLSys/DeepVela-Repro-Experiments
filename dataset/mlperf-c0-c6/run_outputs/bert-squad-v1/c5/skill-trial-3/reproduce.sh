#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$TARGET" != "bert-squad" ]]; then
  echo "usage: $0 --target bert-squad [--stage build|preprocess|run] [--from build|preprocess|run]" >&2
  exit 2
fi
if [[ -n "$STAGE" && -n "$FROM" ]]; then
  echo "--stage and --from are mutually exclusive" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="$ROOT/repo"
TARGET_DIR="$ROOT/$TARGET"
WORK="$TARGET_DIR/work"
OUT=${OUTPUT_DIR:-$TARGET_DIR/out}
VENV="$WORK/venv"
SRC="$WORK/src"
BENCH="$SRC/bert"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DLE_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$TARGET_DIR" "$OUT"

clear_from() {
  case "$1" in
    build)
      rm -rf "$WORK"
      rm -f "$OUT"/*.log "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
        "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json" "$OUT/accuracy.txt"
      ;;
    preprocess)
      rm -f "$WORK/.preprocess.success" "$WORK/.run.success" "$WORK/eval_features.pickle" \
        "$WORK/predictions.json"
      rm -f "$OUT/preprocess.log" "$OUT/run.log" "$OUT/evaluator.log" \
        "$OUT/gpu-before-run.log" "$OUT/mlperf_log_accuracy.json" \
        "$OUT/mlperf_log_detail.txt" "$OUT/mlperf_log_summary.txt" \
        "$OUT/mlperf_log_trace.json" "$OUT/accuracy.txt"
      ;;
    run)
      rm -f "$WORK/.run.success" "$WORK/predictions.json"
      rm -f "$OUT/run.log" "$OUT/evaluator.log" "$OUT/gpu-before-run.log" \
        "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
        "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json" "$OUT/accuracy.txt"
      ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}

write_sut() {
  cat > "$BENCH/pytorch_SUT.py" <<'PY'
import array
import os

import mlperf_loadgen as lg
import numpy as np
import torch
from transformers import BertConfig, BertForQuestionAnswering

from squad_QSL import SQuAD_v1_QSL


class BERT_PyTorch_SUT:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this target")
        config = BertConfig.from_json_file(
            os.path.join(os.path.dirname(__file__), "bert_config.json")
        )
        self.model = BertForQuestionAnswering(config)
        model_path = os.environ["ML_MODEL_FILE_WITH_PATH"]
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location="cpu")
        incompatible = self.model.load_state_dict(state, strict=False)
        expected_unused = {"bert.pooler.dense.weight", "bert.pooler.dense.bias"}
        if incompatible.missing_keys or set(incompatible.unexpected_keys) != expected_unused:
            raise RuntimeError("checkpoint mismatch: {}".format(incompatible))
        self.device = torch.device("cuda:0")
        self.model.eval().to(self.device)
        print("pytorch device: {}".format(next(self.model.parameters()).device), flush=True)
        self.batch_size = int(os.environ.get("BERT_BATCH_SIZE", "32"))
        self.network = args.network
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.qsl = SQuAD_v1_QSL(
            total_count_override=args.max_examples,
            cache_path=os.environ["FEATURE_CACHE_FILE"],
        )

    def issue_queries(self, query_samples):
        for begin in range(0, len(query_samples), self.batch_size):
            samples = query_samples[begin:begin + self.batch_size]
            features = [self.qsl.get_features(sample.index) for sample in samples]
            input_ids = torch.tensor(
                [f.input_ids for f in features], dtype=torch.long, device=self.device
            )
            attention_mask = torch.tensor(
                [f.input_mask for f in features], dtype=torch.long, device=self.device
            )
            token_type_ids = torch.tensor(
                [f.segment_ids for f in features], dtype=torch.long, device=self.device
            )
            with torch.inference_mode():
                result = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict=True,
                )
            output = torch.stack(
                (result.start_logits, result.end_logits), dim=-1
            ).float().cpu().numpy()
            responses = []
            buffers = []
            for sample, logits in zip(samples, output):
                response_array = array.array("B", np.asarray(logits, dtype=np.float32).tobytes())
                buffers.append(response_array)
                address, size = response_array.buffer_info()
                responses.append(lg.QuerySampleResponse(sample.id, address, size))
            lg.QuerySamplesComplete(responses)

    def flush_queries(self):
        pass


def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
PY
}

build_stage() {
  [[ -f "$WORK/.build.success" ]] && return
  mkdir -p "$WORK" "$SRC"
  cp -a "$REPO/language/bert" "$BENCH"
  cp -a "$REPO/loadgen" "$SRC/loadgen"
  ln -s "$DLE_DIR" "$BENCH/DeepLearningExamples"
  write_sut
  {
    echo "python: /home/mi3se/.local/bin/python3.11"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv --python /home/mi3se/.local/bin/python3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'torch==2.7.1' 'transformers==4.52.4' 'numpy<2.3' \
      'absl-py' 'six' 'setuptools<81' 'pybind11' 'boto3'
    uv pip install --python "$VENV/bin/python" "$SRC/loadgen"
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("loadgen import ok")
PY
  } 2>&1 | tee "$OUT/build.log"
  touch "$WORK/.build.success"
}

preprocess_stage() {
  [[ -f "$WORK/.preprocess.success" ]] && return
  [[ -f "$WORK/.build.success" ]] || build_stage
  {
    cd "$BENCH"
    PYTHONPATH="$DLE_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
      VOCAB_FILE="$MODEL_DIR/vocab.txt" DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
      "$VENV/bin/python" - <<PY
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL(cache_path="$WORK/eval_features.pickle")
print("feature_count:", len(qsl.eval_features))
PY
  } 2>&1 | tee "$OUT/preprocess.log"
  touch "$WORK/.preprocess.success"
}

run_stage() {
  [[ -f "$WORK/.run.success" ]] && return
  [[ -f "$WORK/.preprocess.success" ]] || preprocess_stage
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader | tee "$OUT/gpu-before-run.log"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if (( free_mib >= 24000 )); then
    batch_size=32
  elif (( free_mib >= 12000 )); then
    batch_size=12
  elif (( free_mib >= 6000 )); then
    batch_size=4
  else
    echo "insufficient free GPU memory: ${free_mib} MiB" | tee -a "$OUT/gpu-before-run.log"
    exit 1
  fi
  max_args=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  {
    cd "$BENCH"
    PYTHONPATH="$DLE_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
      ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch" \
      VOCAB_FILE="$MODEL_DIR/vocab.txt" DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
      FEATURE_CACHE_FILE="$WORK/eval_features.pickle" \
      LOG_PATH="$OUT" SKIP_VERIFY_ACCURACY=1 BERT_BATCH_SIZE="$batch_size" \
      "$VENV/bin/python" run.py --backend=pytorch --accuracy \
        --user_conf "$BENCH/user.conf" "${max_args[@]}"
  } 2>&1 | tee "$OUT/run.log"
  {
    cd "$BENCH"
    PYTHONPATH="$DLE_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
      "$VENV/bin/python" accuracy-squad.py \
        --vocab_file "$MODEL_DIR/vocab.txt" \
        --val_data "$SQUAD_DIR/dev-v1.1.json" \
        --log_file "$OUT/mlperf_log_accuracy.json" \
        --out_file "$WORK/predictions.json" \
        --features_cache_file "$WORK/eval_features.pickle" \
        "${max_args[@]}"
  } 2>&1 | tee "$OUT/accuracy.txt" "$OUT/evaluator.log"
  touch "$WORK/.run.success"
}

if [[ -n "$FROM" ]]; then
  clear_from "$FROM"
fi
if [[ -n "$STAGE" ]]; then
  case "$STAGE" in
    build) build_stage ;;
    preprocess) preprocess_stage ;;
    run) run_stage ;;
    *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
  esac
else
  case "${FROM:-build}" in
    build) build_stage; preprocess_stage; run_stage ;;
    preprocess) preprocess_stage; run_stage ;;
    run) run_stage ;;
  esac
fi
