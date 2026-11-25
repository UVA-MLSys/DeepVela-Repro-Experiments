#!/usr/bin/env bash
set -euo pipefail

# This runner is intentionally rooted at its invocation directory: it is
# designed to be copied beside a fresh `repo` directory.
TARGET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$TARGET" != "bert-squad" ]]; then
  echo "unsupported target: $TARGET" >&2
  exit 2
fi

ROOT=$(pwd -P)
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples-c2"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
WORK_DIR="$ROOT/$TARGET"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/out}"

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
if [[ ! -d "$WORK_DIR/repo" ]]; then
  cp -a "$ROOT/repo" "$WORK_DIR/repo"
fi
BERT_DIR="$WORK_DIR/repo/language/bert"
if [[ -d "$BERT_DIR/DeepLearningExamples" && ! -L "$BERT_DIR/DeepLearningExamples" ]]; then
  rmdir "$BERT_DIR/DeepLearningExamples"
fi
ln -sfn "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"
mkdir -p "$BERT_DIR/build/data/bert_model" "$BERT_DIR/build/result"
ln -sfn "$SQUAD_DIR/dev-v1.1.json" "$BERT_DIR/build/data/dev-v1.1.json"
ln -sfn "$MODEL_DIR/model.pytorch" "$BERT_DIR/build/data/bert_model/model.pytorch"
ln -sfn "$MODEL_DIR/vocab.txt" "$BERT_DIR/build/data/bert_model/vocab.txt"

python3 - "$BERT_DIR" <<'PY'
from pathlib import Path
import sys

bert = Path(sys.argv[1])
path = bert / "create_squad_data.py"
src = path.read_text()
start = src.index("def _missing_input_presentation")
implementation = r'''def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start or position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)
    return (input_start, input_end)


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn, verbose_logging=False):
    """Convert SQuAD examples to the fixed BERT input presentation."""
    import collections

    unique_id = 1000000000
    for example_index, example in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for i, token in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            tok_start_position, tok_end_position = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position,
                tokenizer, example.orig_answer_text)

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = min(len(all_doc_tokens) - start_offset, max_tokens_for_doc)
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for doc_span_index, doc_span in enumerate(doc_spans):
            tokens = ["[CLS]"]
            segment_ids = [0]
            token_to_orig_map = {}
            token_is_max_context = {}
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                token_is_max_context[len(tokens)] = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if (example.is_impossible or tok_start_position < doc_start or
                        tok_end_position > doc_end):
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible)
            output_fn(feature)
            unique_id += 1
'''
path.write_text(src[:start] + implementation)

path = bert / "pytorch_SUT.py"
src = path.read_text()
src = src.replace("import sys\n", "import sys\nimport pickle\n")
src = src.replace(
    "from squad_QSL import get_squad_QSL\n",
    "from squad_QSL import get_squad_QSL\n    from transformers import BertTokenizer\n"
    "    from create_squad_data import read_squad_examples, convert_examples_to_features\n")
src = src.replace(
    '        self.version = transformers.__version__\n',
    '        self.version = transformers.__version__\n'
    '        print("pytorch device: {}".format(self.dev))\n')
src = src.replace(
    "        self.qsl = get_squad_QSL(args.max_examples)\n",
    '''        val_file = os.environ.get("SQUAD_VAL_FILE", "build/data/dev-v1.1.json")
        vocab_file = os.environ.get(
            "VOCAB_FILE", "build/data/bert_model/vocab.txt")
        feature_file = os.environ.get(
            "FEATURE_FILE", "build/data/eval_features.pickle")
        print("Reading and converting SQuAD examples...")
        eval_examples = read_squad_examples(
            input_file=val_file, is_training=False,
            version_2_with_negative=False)
        tokenizer = BertTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        eval_features = []
        convert_examples_to_features(
            examples=eval_examples, tokenizer=tokenizer,
            max_seq_length=384, doc_stride=128, max_query_length=64,
            is_training=False, output_fn=eval_features.append,
            verbose_logging=False)
        with open(feature_file, "wb") as feature_handle:
            pickle.dump(eval_features, feature_handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print("Created {} feature records".format(len(eval_features)))
        self.qsl = get_squad_QSL(
            args.max_examples, feature_records=eval_features)
''')
path.write_text(src)

path = bert / "accuracy-squad.py"
src = path.read_text()
src = src.replace(
    "    import tokenization\n    from create_squad_data import read_squad_examples\n",
    "    from transformers.models.bert import tokenization_bert as tokenization\n"
    "    from create_squad_data import read_squad_examples\n")
path.write_text(src)
PY

VENV="$WORK_DIR/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python /usr/bin/python3.11 "$VENV" >"$OUTPUT_DIR/build.log" 2>&1
  uv pip install --python "$VENV/bin/python" \
    'torch==2.5.1' 'transformers==4.31.0' 'numpy<2' \
    absl-py six boto3 requests regex tqdm 'setuptools<81' \
    >>"$OUTPUT_DIR/build.log" 2>&1
  uv pip install --python "$VENV/bin/python" 'mlcommons-loadgen==6.0.17' \
    >>"$OUTPUT_DIR/build.log" 2>&1
else
  echo "Reusing $VENV" >"$OUTPUT_DIR/build.log"
fi

cd "$BERT_DIR"
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export SQUAD_VAL_FILE="$SQUAD_DIR/dev-v1.1.json"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export FEATURE_FILE="$BERT_DIR/build/data/eval_features.pickle"
export LOG_PATH="$OUTPUT_DIR"
export SKIP_VERIFY_ACCURACY=1

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits | tee "$OUTPUT_DIR/gpu-memory-before.log"
"$VENV/bin/python" run.py --backend=pytorch --accuracy \
  2>&1 | tee "$OUTPUT_DIR/run.log"

"$VENV/bin/python" accuracy-squad.py \
  --val_data "$SQUAD_DIR/dev-v1.1.json" \
  --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
  --out_file "$BERT_DIR/build/result/predictions.json" \
  --feature_records "$FEATURE_FILE" \
  2>&1 | tee "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
