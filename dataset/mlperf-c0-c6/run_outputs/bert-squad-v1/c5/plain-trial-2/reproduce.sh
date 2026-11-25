#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 --target bert-squad" >&2
    exit 2
}

[[ ${1:-} == "--target" && ${2:-} == "bert-squad" && $# -eq 2 ]] || usage

ROOT_DIR=$(pwd -P)
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR=$CACHE_ROOT/bert-squad/data
MODEL_DIR=$CACHE_ROOT/bert-squad/model
DEEPLEARNING_EXAMPLES_DIR=$CACHE_ROOT/bert-squad/DeepLearningExamples
export UV_CACHE_DIR=$CACHE_ROOT/uv
export UV_EXCLUDE_NEWER=2026-11-10

TARGET_DIR=$ROOT_DIR/bert-squad
WORK_DIR=$TARGET_DIR/work
OUTPUT_DIR=${OUTPUT_DIR:-$TARGET_DIR/out}
mkdir -p "$TARGET_DIR" "$WORK_DIR" "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)

BUILD_LOG=$TARGET_DIR/build.log
RUN_LOG=$TARGET_DIR/run.log
EVALUATOR_LOG=$TARGET_DIR/evaluator.log
GPU_LOG=$TARGET_DIR/gpu_memory.log
: > "$BUILD_LOG"
: > "$RUN_LOG"
: > "$EVALUATOR_LOG"
: > "$GPU_LOG"

for path in \
    "$SQUAD_DIR/dev-v1.1.json" \
    "$MODEL_DIR/model.pytorch" \
    "$MODEL_DIR/vocab.txt" \
    "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/modeling.py"; do
    [[ -r "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done
[[ -d "$ROOT_DIR/repo/language/bert" && -d "$ROOT_DIR/repo/loadgen" ]] || {
    echo "reproduce.sh must be run beside the fresh repo directory" >&2
    exit 1
}

PYTHON_BIN=$(uv python find 3.10)
if [[ ! -x "$WORK_DIR/.venv/bin/python" ]]; then
    uv venv --python "$PYTHON_BIN" "$WORK_DIR/.venv" 2>&1 | tee -a "$BUILD_LOG"
fi
uv pip install --python "$WORK_DIR/.venv/bin/python" \
    'torch==2.5.1' 'transformers==4.46.3' 'numpy==1.26.4' \
    'absl-py==2.3.1' 'six==1.17.0' 'pybind11==2.13.6' \
    'boto3==1.40.76' 'setuptools<81' 2>&1 | tee -a "$BUILD_LOG"
if ! "$WORK_DIR/.venv/bin/python" -c 'import mlperf_loadgen' 2>/dev/null; then
    rm -rf "$WORK_DIR/loadgen-source"
    cp -a "$ROOT_DIR/repo/loadgen" "$WORK_DIR/loadgen-source"
    uv pip install --python "$WORK_DIR/.venv/bin/python" "$WORK_DIR/loadgen-source" \
        2>&1 | tee -a "$BUILD_LOG"
fi

rm -rf "$WORK_DIR/bert"
cp -a "$ROOT_DIR/repo/language/bert" "$WORK_DIR/bert"
mkdir -p "$WORK_DIR/bert/build/data/bert_tf_v1_1_large_fp32_384_v2" \
    "$WORK_DIR/bert/build/result"
ln -s "$SQUAD_DIR/dev-v1.1.json" "$WORK_DIR/bert/build/data/dev-v1.1.json"
ln -s "$MODEL_DIR/vocab.txt" \
    "$WORK_DIR/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
rm -rf "$WORK_DIR/bert/DeepLearningExamples"
ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$WORK_DIR/bert/DeepLearningExamples"
ln -s "$WORK_DIR/eval_features.pickle" "$WORK_DIR/bert/eval_features.pickle"

cat > "$WORK_DIR/bert/pytorch_SUT.py" <<'PY'
import array
import os
import sys

import mlperf_loadgen as lg
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.environ["DLE_BERT_DIR"])

import modeling
from squad_QSL import get_squad_QSL

# This pinned NVIDIA source used the former bool spelling for tanh GELU.
modeling.ACT2FN["gelu"] = lambda x: torch.nn.functional.gelu(
    x, approximate="tanh"
)
BertConfig = modeling.BertConfig
BertForQuestionAnswering = modeling.BertForQuestionAnswering


class BERT_PyTorch_SUT:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this target")
        self.device = torch.device("cuda:0")
        self.batch_size = int(os.environ.get("BERT_BATCH_SIZE", "32"))
        print("Loading PyTorch model...")
        config = BertConfig.from_json_file(os.environ["BERT_CONFIG"])
        self.model = BertForQuestionAnswering(config)
        state = torch.load(
            os.environ["ML_MODEL_FILE_WITH_PATH"],
            map_location="cpu",
            weights_only=False,
        )
        state = {
            key.replace(".intermediate.dense.", ".intermediate.dense_act.")
               .replace(".pooler.dense.", ".pooler.dense_act."): value
            for key, value in state.items()
        }
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.to(self.device)
        print("pytorch device: {}".format(next(self.model.parameters()).device))
        print("pytorch batch size: {}".format(self.batch_size))

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")
        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        with torch.inference_mode():
            for begin in range(0, len(query_samples), self.batch_size):
                samples = query_samples[begin:begin + self.batch_size]
                features = [self.qsl.get_features(s.index) for s in samples]
                input_ids = torch.as_tensor(
                    [f.input_ids for f in features], dtype=torch.long, device=self.device
                )
                segment_ids = torch.as_tensor(
                    [f.segment_ids for f in features], dtype=torch.long, device=self.device
                )
                input_mask = torch.as_tensor(
                    [f.input_mask for f in features], dtype=torch.long, device=self.device
                )
                start_logits, end_logits = self.model(
                    input_ids, segment_ids, input_mask
                )
                output = torch.stack((start_logits, end_logits), dim=-1)
                output = output.float().cpu().numpy()
                for sample, logits in zip(samples, output):
                    response_array = array.array(
                        "B", np.ascontiguousarray(logits, dtype=np.float32).tobytes()
                    )
                    address, size = response_array.buffer_info()
                    response = lg.QuerySampleResponse(sample.id, address, size)
                    lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
PY

rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" \
    "$OUTPUT_DIR/mlperf_log_detail.txt" \
    "$OUTPUT_DIR/mlperf_log_summary.txt" \
    "$OUTPUT_DIR/accuracy.txt"

{
    echo "GPU memory immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits
} | tee -a "$GPU_LOG"

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if (( FREE_MIB >= 50000 )); then
    BERT_BATCH_SIZE=64
elif (( FREE_MIB >= 30000 )); then
    BERT_BATCH_SIZE=32
elif (( FREE_MIB >= 16000 )); then
    BERT_BATCH_SIZE=16
elif (( FREE_MIB >= 8000 )); then
    BERT_BATCH_SIZE=4
else
    BERT_BATCH_SIZE=1
fi

export PATH="$WORK_DIR/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$WORK_DIR/huggingface"
export CUDA_VISIBLE_DEVICES=0
export DLE_BERT_DIR="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT"
export BERT_CONFIG="$WORK_DIR/bert/bert_config.json"
export BERT_BATCH_SIZE
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export LOG_PATH="$OUTPUT_DIR"
export SKIP_VERIFY_ACCURACY=1

cd "$WORK_DIR/bert"
set -o pipefail
python3 run.py --backend=pytorch --accuracy 2>&1 | tee "$RUN_LOG"

python3 accuracy-squad.py \
    --vocab_file "$MODEL_DIR/vocab.txt" \
    --val_data "$SQUAD_DIR/dev-v1.1.json" \
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --out_file "$WORK_DIR/bert/build/result/predictions.json" \
    --features_cache_file "$WORK_DIR/eval_features.pickle" \
    2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$EVALUATOR_LOG"

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
