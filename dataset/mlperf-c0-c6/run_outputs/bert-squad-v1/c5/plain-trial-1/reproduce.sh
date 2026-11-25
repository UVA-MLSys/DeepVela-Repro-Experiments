#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

usage() {
    echo "usage: $0 --target <target-id>" >&2
    exit 2
}

target=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) [[ $# -ge 2 ]] || usage; target="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ "$target" == "bert-squad" ]] || usage
[[ -d repo ]] || { echo "repo directory not found" >&2; exit 1; }

target_dir="$PWD/$target"
work_dir="$target_dir/work"
output_dir="${OUTPUT_DIR:-$target_dir/out}"
mkdir -p "$target_dir" "$output_dir"
rm -rf "$work_dir"
mkdir -p "$work_dir"

build_log="$target_dir/build.log"
run_log="$target_dir/run.log"
evaluator_log="$target_dir/evaluator.log"
: >"$build_log"
: >"$run_log"
: >"$evaluator_log"

{
    echo "Staging pinned source and inputs"
    cp -a repo/. "$work_dir/repo/"
    mkdir -p "$work_dir/repo/language/bert/DeepLearningExamples/PyTorch/LanguageModeling"
    cp -a "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT" \
        "$work_dir/repo/language/bert/DeepLearningExamples/PyTorch/LanguageModeling/"

    uv venv --python 3.11 "$work_dir/venv"
    uv pip install --offline --python "$work_dir/venv/bin/python" \
        'numpy==1.26.4' 'torch==2.4.1' 'transformers==4.45.2' \
        'absl-py' 'six' 'setuptools==80.9.0' 'pybind11' 'boto3'
    uv pip install --offline --python "$work_dir/venv/bin/python" \
        "$work_dir/repo/loadgen"
} 2>&1 | tee -a "$build_log"

bert_dir="$work_dir/repo/language/bert"

cat >"$bert_dir/pytorch_SUT.py" <<'PY'
import array
import os
import sys

import mlperf_loadgen as lg
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.path.dirname(__file__))

from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL


class BERT_PyTorch_SUT:
    def __init__(self, args):
        model_path = os.environ.get(
            "ML_MODEL_FILE_WITH_PATH",
            "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch",
        )
        print("Loading PyTorch model...")
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, torch.nn.Module):
            self.model = loaded
        else:
            state = loaded.get("model", loaded.get("state_dict", loaded)) if isinstance(loaded, dict) else loaded
            config = BertConfig.from_json_file(os.path.join(os.path.dirname(__file__), "bert_config.json"))
            self.model = BertForQuestionAnswering(config)
            incompatible = self.model.load_state_dict(state, strict=False)
            allowed_unexpected = {"bert.pooler.dense.weight", "bert.pooler.dense.bias"}
            if incompatible.missing_keys or set(incompatible.unexpected_keys) != allowed_unexpected:
                raise RuntimeError("Unexpected checkpoint mismatch: {}".format(incompatible))

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this target")
        self.device = torch.device("cuda:0")
        self.model.eval().to(self.device)
        print("pytorch device: {}".format(next(self.model.parameters()).device))
        self.batch_size = int(os.environ.get("PYTORCH_BATCH_SIZE", "16"))

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")
        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        for offset in range(0, len(query_samples), self.batch_size):
            samples = query_samples[offset:offset + self.batch_size]
            features = [self.qsl.get_features(sample.index) for sample in samples]
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long, device=self.device)
            segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long, device=self.device)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                logits = torch.stack((start_logits, end_logits), dim=-1).float().cpu().numpy()
            responses = []
            response_buffers = []
            for sample, output in zip(samples, logits):
                response_buffer = array.array("B", np.ascontiguousarray(output, dtype=np.float32).tobytes())
                response_buffers.append(response_buffer)
                address, length = response_buffer.buffer_info()
                responses.append(lg.QuerySampleResponse(sample.id, address, length))
            lg.QuerySamplesComplete(responses)

    def flush_queries(self):
        pass


def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
PY

export PATH="$work_dir/venv/bin:$PATH"
export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export LOG_PATH="$work_dir/loadgen-logs"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$bert_dir/DeepLearningExamples/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}"

{
    cd "$bert_dir"
    python3 - <<'PY'
from squad_QSL import get_squad_QSL
qsl = get_squad_QSL()
print("Preprocessed {} SQuAD features".format(qsl.count))
PY
} 2>&1 | tee -a "$build_log"

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
{
    echo "GPU memory before run: ${free_mib} MiB free"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
} 2>&1 | tee -a "$run_log"

if (( free_mib >= 40000 )); then
    export PYTORCH_BATCH_SIZE=32
elif (( free_mib >= 20000 )); then
    export PYTORCH_BATCH_SIZE=16
elif (( free_mib >= 10000 )); then
    export PYTORCH_BATCH_SIZE=8
elif (( free_mib >= 6000 )); then
    export PYTORCH_BATCH_SIZE=2
else
    echo "Insufficient free GPU memory: ${free_mib} MiB" | tee -a "$run_log" >&2
    exit 1
fi
echo "Using PyTorch batch size $PYTORCH_BATCH_SIZE" | tee -a "$run_log"

{
    cd "$bert_dir"
    SKIP_VERIFY_ACCURACY=1 python3 run.py --backend=pytorch --accuracy
} 2>&1 | tee -a "$run_log"

cp "$work_dir/loadgen-logs/mlperf_log_accuracy.json" "$output_dir/mlperf_log_accuracy.json"
cp "$work_dir/loadgen-logs/mlperf_log_detail.txt" "$output_dir/mlperf_log_detail.txt"

mkdir -p "$work_dir/result"
{
    cd "$bert_dir"
    python3 accuracy-squad.py \
        --vocab_file "$MODEL_DIR/vocab.txt" \
        --val_data "$SQUAD_DIR/dev-v1.1.json" \
        --log_file "$output_dir/mlperf_log_accuracy.json" \
        --out_file "$work_dir/result/predictions.json" \
        --features_cache_file "$bert_dir/eval_features.pickle"
} 2>&1 | tee "$output_dir/accuracy.txt" "$evaluator_log"

echo "Completed target $target"
