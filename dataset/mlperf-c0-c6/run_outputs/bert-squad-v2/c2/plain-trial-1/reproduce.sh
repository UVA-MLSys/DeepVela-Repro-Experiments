#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--target" || "${2:-}" != "bert-squad" || $# -ne 2 ]]; then
  echo "usage: $0 --target bert-squad" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "$0")" && pwd)
TARGET="$ROOT/bert-squad"
WORK="$TARGET/work"
OUTPUT_DIR=${OUTPUT_DIR:-"$TARGET/out"}
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples-c2"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$TARGET" "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
rm -rf "$WORK" "$TARGET/venv"
rm -f "$OUTPUT_DIR"/*.log "$OUTPUT_DIR/mlperf_log_accuracy.json" \
  "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" \
  "$OUTPUT_DIR/mlperf_log_trace.json" "$OUTPUT_DIR/accuracy.txt"
cp -a "$ROOT/repo/language/bert" "$WORK"
rm -rf "$WORK/DeepLearningExamples"
ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$WORK/DeepLearningExamples"

{
  echo "Creating Python 3.11 environment"
  uv venv --python /home/mi3se/.local/bin/python3.11 "$TARGET/venv"
  uv pip install --python "$TARGET/venv/bin/python" \
    'torch==2.5.1' 'transformers==4.46.3' 'numpy<2' \
    absl-py six pybind11 boto3 'setuptools==80.9.0'
  uv pip install --python "$TARGET/venv/bin/python" "$ROOT/repo/loadgen"
} >"$OUTPUT_DIR/build.log" 2>&1

cat >"$WORK/build_features.py" <<'PY'
import collections
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
from create_squad_data import InputFeatures, read_squad_examples
import tokenization


def is_max_context(spans, current, position):
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
    return current == best_span


def convert(examples, tokenizer, max_seq_length=384, doc_stride=128,
            max_query_length=64):
    Span = collections.namedtuple("Span", ["start", "length"])
    features = []
    unique_id = 1000000000
    for example_index, example in enumerate(examples):
        query = tokenizer.tokenize(example.question_text)[:max_query_length]
        tok_to_orig = []
        all_doc_tokens = []
        for orig_index, token in enumerate(example.doc_tokens):
            for sub_token in tokenizer.tokenize(token):
                tok_to_orig.append(orig_index)
                all_doc_tokens.append(sub_token)

        max_doc = max_seq_length - len(query) - 3
        spans = []
        start = 0
        while start < len(all_doc_tokens):
            length = min(max_doc, len(all_doc_tokens) - start)
            spans.append(Span(start, length))
            if start + length == len(all_doc_tokens):
                break
            start += min(length, doc_stride)

        for span_index, span in enumerate(spans):
            tokens = ["[CLS]"] + query + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            token_to_orig_map = {}
            token_is_max_context = {}
            for i in range(span.length):
                split_index = span.start + i
                token_index = len(tokens)
                token_to_orig_map[token_index] = tok_to_orig[split_index]
                token_is_max_context[token_index] = is_max_context(
                    spans, span_index, split_index)
                tokens.append(all_doc_tokens[split_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = max_seq_length - len(input_ids)
            input_ids += [0] * padding
            input_mask += [0] * padding
            segment_ids += [0] * padding
            assert len(input_ids) == max_seq_length
            features.append(InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids))
            unique_id += 1
    return features


examples = read_squad_examples(os.environ["SQUAD_FILE"], False)
tok = tokenization.BertTokenizer(os.environ["VOCAB_FILE"], do_lower_case=True)
records = convert(examples, tok)
with open(os.environ["FEATURE_FILE"], "wb") as f:
    pickle.dump(records, f, protocol=4)
print("examples:", len(examples))
print("features:", len(records))
PY

# The stripped QSL requires externally supplied feature records. Wire the records
# into the otherwise unchanged reference SUT and add the requested device line.
sed -i '/^import sys$/a import pickle' "$WORK/pytorch_SUT.py"
sed -i '/        self.version = transformers.__version__/i\        print("pytorch device: {}".format(self.dev))' "$WORK/pytorch_SUT.py"
sed -i 's/self.qsl = get_squad_QSL(args.max_examples)/with open(os.environ["FEATURE_RECORDS"], "rb") as feature_file:\n            feature_records = pickle.load(feature_file)\n        self.qsl = get_squad_QSL(args.max_examples, feature_records=feature_records)/' "$WORK/pytorch_SUT.py"

export SQUAD_FILE="$SQUAD_DIR/dev-v1.1.json"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export FEATURE_FILE="$TARGET/features.pickle"
(
  cd "$WORK"
  "$TARGET/venv/bin/python" build_features.py
) >"$OUTPUT_DIR/preprocess.log" 2>&1

export FEATURE_RECORDS="$FEATURE_FILE"
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export LOG_PATH="$OUTPUT_DIR"
export SKIP_VERIFY_ACCURACY=1

# Keep the required immediately-before-run availability evidence in the run log.
{
  echo "GPU memory immediately before run (MiB):"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits
  cd "$WORK"
  "$TARGET/venv/bin/python" run.py --backend=pytorch --accuracy
} >"$OUTPUT_DIR/run.log" 2>&1

(
  cd "$WORK"
  "$TARGET/venv/bin/python" accuracy-squad.py \
    --val_data "$SQUAD_DIR/dev-v1.1.json" \
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --out_file "$TARGET/predictions.json" \
    --feature_records "$FEATURE_FILE"
) >"$OUTPUT_DIR/accuracy.txt" 2>"$OUTPUT_DIR/evaluator-stderr.log"
cp "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"

"$TARGET/venv/bin/python" - "$OUTPUT_DIR/accuracy.txt" "$ROOT/report.json" <<'PY'
import json
import sys

metric = None
for line in open(sys.argv[1]):
    try:
        value = json.loads(line)
    except Exception:
        continue
    if isinstance(value, dict) and "exact_match" in value and "f1" in value:
        metric = value
if metric is None:
    raise RuntimeError("evaluator metrics were not printed")
with open(sys.argv[2], "w") as f:
    json.dump({"bert-squad": {
        "exact_match": metric["exact_match"], "f1": metric["f1"]}}, f)
    f.write("\n")
PY

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
