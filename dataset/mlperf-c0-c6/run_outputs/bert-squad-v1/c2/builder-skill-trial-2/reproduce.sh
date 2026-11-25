#!/usr/bin/env bash
set -euo pipefail

TARGET=""
MODE="all"
START_STAGE=""
MAX_EXAMPLES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --stage) MODE="$2"; shift 2 ;;
    --from) START_STAGE="$2"; shift 2 ;;
    --max-examples) MAX_EXAMPLES="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$TARGET" == "bert-squad" ]] || { echo "target must be bert-squad" >&2; exit 2; }
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
: "${CACHE_ROOT:=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
: "${SQUAD_DIR:=$CACHE_ROOT/bert-squad/data}"
: "${MODEL_DIR:=$CACHE_ROOT/bert-squad/model}"
: "${DEEPLEARNING_EXAMPLES_DIR:=$CACHE_ROOT/bert-squad/DeepLearningExamples-c2}"
: "${UV_CACHE_DIR:=$CACHE_ROOT/uv}"
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

ROOT=$(pwd -P)
SRC="$ROOT/repo"
TARGET_DIR="$ROOT/$TARGET"
WORK="$TARGET_DIR/work"
RUN_SRC="$WORK/repo"
VENV="$WORK/venv"
MARKERS="$WORK/markers"
mkdir -p "$OUTPUT_DIR" "$WORK" "$MARKERS"

stages=(environment prepare gpu evaluate)

should_run() {
  local wanted="$1" active=0 s
  if [[ "$MODE" != "all" ]]; then [[ "$MODE" == "$wanted" ]]; return; fi
  if [[ -z "$START_STAGE" ]]; then return 0; fi
  for s in "${stages[@]}"; do
    [[ "$s" == "$START_STAGE" ]] && active=1
    [[ "$s" == "$wanted" && "$active" == 1 ]] && return 0
  done
  return 1
}

invalidate_from() {
  local start="$1" active=0 s
  for s in "${stages[@]}"; do
    [[ "$s" == "$start" ]] && active=1
    if [[ "$active" == 1 ]]; then
      rm -f "$MARKERS/$s.ok"
      case "$s" in
        environment) rm -f "$OUTPUT_DIR/environment.log" ;;
        prepare) rm -f "$OUTPUT_DIR/prepare.log" ;;
        gpu) rm -f "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/mlperf_log_accuracy.json" \
          "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_summary.txt" \
          "$OUTPUT_DIR/mlperf_log_trace.json" ;;
        evaluate) rm -f "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt" ;;
      esac
    fi
  done
}

if [[ -n "$START_STAGE" ]]; then invalidate_from "$START_STAGE"; fi

stage_environment() {
  local log="$OUTPUT_DIR/environment.log"
  {
    command -v uv
    uv --version
    uv venv --clear --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'torch==2.7.1' 'transformers==4.57.1' 'numpy<2' six absl-py pybind11 boto3
    uv pip install --python "$VENV/bin/python" "$SRC/loadgen"
    "$VENV/bin/python" -c 'import torch, transformers, mlperf_loadgen; print("torch", torch.__version__); print("transformers", transformers.__version__)'
  } 2>&1 | tee "$log"
  touch "$MARKERS/environment.ok"
}

write_repair_files() {
  "$VENV/bin/python" - "$RUN_SRC/language/bert" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
root.joinpath('tokenization.py').write_text(r'''import six
from transformers.models.bert.tokenization_bert import BasicTokenizer, BertTokenizer


def printable_text(text):
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    return str(text)


def whitespace_tokenize(text):
    text = text.strip()
    return text.split() if text else []


class FullTokenizer:
    def __init__(self, vocab_file, do_lower_case=True):
        self._tokenizer = BertTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)

    def tokenize(self, text):
        return self._tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        return self._tokenizer.convert_tokens_to_ids(tokens)
''')
root.joinpath('feature_contract.py').write_text(r'''import collections


def _imports():
    import tokenization
    from create_squad_data import InputFeatures, read_squad_examples
    return tokenization, InputFeatures, read_squad_examples


def _max_context(spans, current, position):
    best_score = None
    best_index = None
    for i, span in enumerate(spans):
        end = span.start + span.length - 1
        if position < span.start or position > end:
            continue
        left = position - span.start
        right = end - position
        score = min(left, right) + 0.01 * span.length
        if best_score is None or score > best_score:
            best_score = score
            best_index = i
    return current == best_index


def make_features(input_file, vocab_file, max_seq_length=384,
                  doc_stride=128, max_query_length=64):
    tokenization, InputFeatures, read_squad_examples = _imports()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=True)
    examples = read_squad_examples(input_file=input_file, is_training=False)
    features = []
    unique_id = 1000000000
    Span = collections.namedtuple('Span', ['start', 'length'])
    for example_index, example in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)[:max_query_length]
        tok_to_orig_index = []
        all_doc_tokens = []
        for i, token in enumerate(example.doc_tokens):
            for sub_token in tokenizer.tokenize(token):
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        max_doc = max_seq_length - len(query_tokens) - 3
        spans = []
        start = 0
        while start < len(all_doc_tokens):
            length = min(max_doc, len(all_doc_tokens) - start)
            spans.append(Span(start, length))
            if start + length == len(all_doc_tokens):
                break
            start += min(length, doc_stride)
        for span_index, span in enumerate(spans):
            tokens = ['[CLS]'] + query_tokens + ['[SEP]']
            segment_ids = [0] * len(tokens)
            token_to_orig_map = {}
            token_is_max_context = {}
            for i in range(span.length):
                split_index = span.start + i
                token_index = len(tokens)
                token_to_orig_map[token_index] = tok_to_orig_index[split_index]
                token_is_max_context[token_index] = _max_context(
                    spans, span_index, split_index)
                tokens.append(all_doc_tokens[split_index])
                segment_ids.append(1)
            tokens.append('[SEP]')
            segment_ids.append(1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == max_seq_length
            features.append(InputFeatures(
                unique_id=unique_id, example_index=example_index,
                doc_span_index=span_index, tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids, input_mask=input_mask,
                segment_ids=segment_ids))
            unique_id += 1
    return examples, features
''')

qsl = root / 'squad_QSL.py'
text = qsl.read_text()
old = '''def get_squad_QSL(total_count_override=None, perf_count_override=None,
                  feature_records=None):
    return SQuAD_v1_QSL(total_count_override, perf_count_override,
                        feature_records)
'''
new = '''def get_squad_QSL(total_count_override=None, perf_count_override=None,
                  feature_records=None):
    if feature_records is None:
        from feature_contract import make_features
        _, feature_records = make_features(
            "build/data/dev-v1.1.json",
            "build/data/bert_model/vocab.txt")
    return SQuAD_v1_QSL(total_count_override, perf_count_override,
                        feature_records)
'''
if old not in text:
    raise SystemExit('unexpected squad_QSL.py integration point')
qsl.write_text(text.replace(old, new))

sut = root / 'pytorch_SUT.py'
text = sut.read_text()
old = '        self.model.load_state_dict(torch.load(model_file), strict=False)\n'
new = '''        state = torch.load(model_file, map_location=self.dev, weights_only=True)
        self.model.load_state_dict(state, strict=False)
        print("pytorch device: {}".format(next(self.model.parameters()).device))
'''
if old not in text:
    raise SystemExit('unexpected pytorch_SUT.py integration point')
sut.write_text(text.replace(old, new))

accuracy = root / 'accuracy-squad.py'
text = accuracy.read_text()
text = text.replace('import pkg_resources\n', 'import importlib.metadata\n')
text = text.replace(
    'installed = {pkg.key for pkg in pkg_resources.working_set}',
    'installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}')
needle = '''try:
    import tokenization
'''
replacement = '''sys.path.insert(0, os.path.dirname(__file__))
try:
    import tokenization
'''
if needle not in text:
    raise SystemExit('unexpected accuracy-squad.py import point')
accuracy.write_text(text.replace(needle, replacement))
PY
}

stage_prepare() {
  local log="$OUTPUT_DIR/prepare.log"
  {
    [[ -f "$SQUAD_DIR/dev-v1.1.json" ]]
    [[ -f "$MODEL_DIR/model.pytorch" ]]
    [[ -f "$MODEL_DIR/vocab.txt" ]]
    [[ -d "$DEEPLEARNING_EXAMPLES_DIR" ]]
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    rm -rf "$RUN_SRC"
    cp -a "$SRC" "$RUN_SRC"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$RUN_SRC/language/bert/DeepLearningExamples"
    mkdir -p "$RUN_SRC/language/bert/build/data/bert_model" "$RUN_SRC/language/bert/build/logs"
    ln -s "$SQUAD_DIR/dev-v1.1.json" "$RUN_SRC/language/bert/build/data/dev-v1.1.json"
    ln -s "$MODEL_DIR/model.pytorch" "$RUN_SRC/language/bert/build/data/bert_model/model.pytorch"
    ln -s "$MODEL_DIR/vocab.txt" "$RUN_SRC/language/bert/build/data/bert_model/vocab.txt"
    write_repair_files
    cd "$RUN_SRC/language/bert"
    PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT" \
      "$VENV/bin/python" - <<'PY'
import json
import pickle
from feature_contract import make_features
examples, features = make_features('build/data/dev-v1.1.json', 'build/data/bert_model/vocab.txt')
print('examples', len(examples))
print('features', len(features))
print('first_feature_length', len(features[0].input_ids))
assert len(examples) == 10570 and len(features) > len(examples)
assert len(features[0].input_ids) == 384
with open('build/data/eval_features.pickle', 'wb') as f:
    pickle.dump(features, f, protocol=4)
PY
  } 2>&1 | tee "$log"
  touch "$MARKERS/prepare.ok"
}

stage_gpu() {
  local log="$OUTPUT_DIR/run.log" extra=()
  [[ -n "$MAX_EXAMPLES" ]] && extra=(--max_examples "$MAX_EXAMPLES")
  {
    echo "GPU memory immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    cd "$RUN_SRC/language/bert"
    LOG_PATH="$OUTPUT_DIR" SKIP_VERIFY_ACCURACY=1 \
      PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT" \
      "$VENV/bin/python" run.py --backend=pytorch --accuracy "${extra[@]}"
  } 2>&1 | tee "$log"
  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" ]]
  [[ -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
  grep -q '^pytorch device: ' "$log"
  touch "$MARKERS/gpu.ok"
}

stage_evaluate() {
  local log="$OUTPUT_DIR/evaluator.log" extra=()
  [[ -n "$MAX_EXAMPLES" ]] && extra=(--max_examples "$MAX_EXAMPLES")
  {
    cd "$RUN_SRC/language/bert"
    LOG_PATH="$OUTPUT_DIR" \
      PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT" \
      "$VENV/bin/python" accuracy-squad.py \
        --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
        --feature_records build/data/eval_features.pickle \
        --out_file "$WORK/predictions.json" "${extra[@]}"
  } 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" | tee "$log"
  grep -q '"exact_match"' "$OUTPUT_DIR/accuracy.txt"
  grep -q '"f1"' "$OUTPUT_DIR/accuracy.txt"
  touch "$MARKERS/evaluate.ok"
}

for stage in "${stages[@]}"; do
  if should_run "$stage"; then
    if [[ "$MODE" == "all" && -f "$MARKERS/$stage.ok" && -z "$START_STAGE" ]]; then
      echo "Skipping completed stage: $stage"
    else
      "stage_$stage"
    fi
  fi
done
