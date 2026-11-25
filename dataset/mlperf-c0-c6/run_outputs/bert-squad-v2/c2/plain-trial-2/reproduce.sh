#!/usr/bin/env bash
set -euo pipefail

target=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$target" != "bert-squad" ]]; then
  echo "unsupported target: $target" >&2
  exit 2
fi

: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples-c2"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repo_dir="$script_dir/repo"
mkdir -p "$(dirname -- "$OUTPUT_DIR")"
work_dir=$(cd -- "$(dirname -- "$OUTPUT_DIR")" && pwd)/work
mkdir -p "$OUTPUT_DIR" "$work_dir"

exec > >(tee -a "$OUTPUT_DIR/reproduce.log") 2>&1
echo "reproduction target: $target"

if [[ ! -x "$work_dir/.venv/bin/python" ]]; then
  uv venv --python 3.11 "$work_dir/.venv"
fi
uv pip install --python "$work_dir/.venv/bin/python" \
  'numpy<2' six absl-py boto3 'setuptools<81' 'torch==2.6.0' 'transformers==4.48.3' pybind11 \
  2>&1 | tee -a "$OUTPUT_DIR/build.log"
if ! "$work_dir/.venv/bin/python" -c 'import mlperf_loadgen' 2>/dev/null; then
  uv pip install --python "$work_dir/.venv/bin/python" --no-deps "$repo_dir/loadgen" \
    2>&1 | tee -a "$OUTPUT_DIR/build.log"
fi

rm -rf "$work_dir/bert"
cp -a "$repo_dir/language/bert" "$work_dir/bert"
rm -rf "$work_dir/bert/DeepLearningExamples"
ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$work_dir/bert/DeepLearningExamples"

cat > "$work_dir/feature_builder.py" <<'PY'
import collections
import os
import pickle
import sys

bert_dir, data_path, vocab_path, output_path = sys.argv[1:]
sys.path.insert(0, os.path.join(bert_dir, "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, bert_dir)
import tokenization
from create_squad_data import InputFeatures, read_squad_examples

MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
MAX_QUERY_LENGTH = 64

def is_max_context(spans, current_span, position):
    best_score = None
    best_span = None
    for i, span in enumerate(spans):
        end = span.start + span.length - 1
        if position < span.start or position > end:
            continue
        left = position - span.start
        right = end - position
        score = min(left, right) + 0.01 * span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span = i
    return current_span == best_span

Tokenizer = getattr(tokenization, "FullTokenizer", None) or tokenization.BertTokenizer
tokenizer = Tokenizer(vocab_file=vocab_path, do_lower_case=True)
examples = read_squad_examples(data_path, is_training=False, version_2_with_negative=False)
features = []
unique_id = 1000000000
DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

for example_index, example in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)[:MAX_QUERY_LENGTH]
    all_doc_tokens = []
    tok_to_orig_index = []
    for orig_index, token in enumerate(example.doc_tokens):
        for sub_token in tokenizer.tokenize(token):
            tok_to_orig_index.append(orig_index)
            all_doc_tokens.append(sub_token)

    max_doc_tokens = MAX_SEQ_LENGTH - len(query_tokens) - 3
    spans = []
    start = 0
    while start < len(all_doc_tokens):
        length = min(max_doc_tokens, len(all_doc_tokens) - start)
        spans.append(DocSpan(start, length))
        if start + length == len(all_doc_tokens):
            break
        start += min(length, DOC_STRIDE)

    for span_index, span in enumerate(spans):
        tokens = ["[CLS]"] + query_tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        token_to_orig_map = {}
        token_is_max_context = {}
        for i in range(span.length):
            doc_position = span.start + i
            feature_position = len(tokens)
            token_to_orig_map[feature_position] = tok_to_orig_index[doc_position]
            token_is_max_context[feature_position] = is_max_context(spans, span_index, doc_position)
            tokens.append(all_doc_tokens[doc_position])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = MAX_SEQ_LENGTH - len(input_ids)
        input_ids += [0] * padding
        input_mask += [0] * padding
        segment_ids += [0] * padding
        assert len(input_ids) == len(input_mask) == len(segment_ids) == MAX_SEQ_LENGTH
        features.append(InputFeatures(
            unique_id=unique_id,
            example_index=example_index,
            doc_span_index=span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
        ))
        unique_id += 1

with open(output_path, "wb") as f:
    pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"feature records: {len(features)} from {len(examples)} examples")
PY

cat > "$work_dir/bert/squad_QSL.py" <<'PY'
import os
import pickle
import mlperf_loadgen as lg

class SQuAD_v1_QSL:
    def __init__(self, total_count_override=None, perf_count_override=None, feature_records=None):
        print("Constructing QSL...")
        if feature_records is None:
            with open(os.environ["FEATURE_RECORDS"], "rb") as f:
                feature_records = pickle.load(f)
        self.eval_features = list(feature_records)
        self.count = total_count_override or len(self.eval_features)
        self.perf_count = perf_count_override or self.count
        self.qsl = lg.ConstructQSL(self.count, self.perf_count, self.load_query_samples, self.unload_query_samples)
        print("Finished constructing QSL.")
    def load_query_samples(self, sample_list):
        pass
    def unload_query_samples(self, sample_list):
        pass
    def get_features(self, sample_id):
        return self.eval_features[sample_id]
    def __del__(self):
        print("Finished destroying QSL.")

def get_squad_QSL(total_count_override=None, perf_count_override=None, feature_records=None):
    return SQuAD_v1_QSL(total_count_override, perf_count_override, feature_records)
PY

cd "$work_dir/bert"
ln -sf "$MODEL_DIR/vocab.txt" vocab.txt
FEATURE_RECORDS="$work_dir/features.pickle"
export FEATURE_RECORDS
if [[ ! -s "$FEATURE_RECORDS" ]]; then
  "$work_dir/.venv/bin/python" "$work_dir/feature_builder.py" \
    "$work_dir/bert" "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/vocab.txt" "$FEATURE_RECORDS" \
    2>&1 | tee "$OUTPUT_DIR/preprocess.log"
fi

export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export LOG_PATH="$OUTPUT_DIR"
export SKIP_VERIFY_ACCURACY=1
if ! "$work_dir/.venv/bin/python" -c 'import json,sys; json.load(open(sys.argv[1]))' \
     "$OUTPUT_DIR/mlperf_log_accuracy.json" 2>/dev/null; then
  echo "GPU memory immediately before run:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits \
    | tee "$OUTPUT_DIR/gpu-memory.log"
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -cd '0-9')
  if [[ -z "$free_mb" || "$free_mb" -lt 12000 ]]; then
    echo "insufficient free GPU memory: ${free_mb:-unknown} MiB" >&2
    exit 1
  fi
  echo "pytorch device: cuda:0" | tee "$OUTPUT_DIR/run.log"
  "$work_dir/.venv/bin/python" run.py --backend=pytorch --accuracy \
    2>&1 | tee -a "$OUTPUT_DIR/run.log"
else
  echo "reusing complete LoadGen accuracy log"
fi

mkdir -p "$work_dir/result"
"$work_dir/.venv/bin/python" accuracy-squad.py \
  --val_data "$SQUAD_DIR/dev-v1.1.json" \
  --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
  --out_file "$work_dir/result/predictions.json" \
  --feature_records "$FEATURE_RECORDS" \
  2> >(tee "$OUTPUT_DIR/evaluator.log" >&2) | tee "$OUTPUT_DIR/accuracy.txt"

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
echo "reproduction complete"
