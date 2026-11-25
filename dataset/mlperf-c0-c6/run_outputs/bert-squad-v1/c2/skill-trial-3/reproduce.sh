#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET=${2:?missing target id}; shift 2 ;;
    --stage) STAGE=${2:?missing stage}; shift 2 ;;
    --from) FROM=${2:?missing stage}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "bert-squad" ]] || { echo "unsupported target: $TARGET" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")" && pwd)
REPO="$ROOT/repo"
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/$TARGET/out"}
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$ROOT/${OUTPUT_DIR#./}" ;; esac
TARGET_DIR=$(dirname "$OUTPUT_DIR")
WORK="$TARGET_DIR/work"
VENV="$WORK/.venv"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
mkdir -p "$OUTPUT_DIR" "$WORK"

stage_setup() {
  if [[ -f "$WORK/setup.success" ]]; then return; fi
  {
    echo "stage: setup"
    echo "python: $(python3.11 --version 2>&1)"
    uv venv --clear --python python3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'torch==2.4.1' 'transformers==4.45.2' 'numpy==1.26.4' \
      'absl-py==2.5.1' 'six==1.17.0' 'setuptools<81'
    rm -rf "$WORK/bert" "$WORK/loadgen"
    cp -a "$REPO/language/bert" "$WORK/bert"
    cp -a "$REPO/loadgen" "$WORK/loadgen"
    uv pip install --python "$VENV/bin/python" "$WORK/loadgen"
    "$VENV/bin/python" - <<'PY'
import torch, transformers, numpy, mlperf_loadgen
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
  } 2>&1 | tee "$OUTPUT_DIR/setup.log"
  touch "$WORK/setup.success"
}

write_helpers() {
  cat > "$WORK/bert/tokenization.py" <<'PY'
from __future__ import absolute_import, division, print_function
import collections, re, unicodedata

def convert_to_unicode(text):
    if isinstance(text, str): return text
    if isinstance(text, bytes): return text.decode("utf-8", "ignore")
    raise ValueError("Unsupported string type: %s" % type(text))

def printable_text(text):
    return convert_to_unicode(text)

def whitespace_tokenize(text):
    text = text.strip()
    return text.split() if text else []

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, encoding="utf-8") as reader:
        for index, token in enumerate(reader):
            vocab[token.rstrip("\n")] = index
    return vocab

def convert_by_vocab(vocab, items):
    return [vocab[item] for item in items]

class FullTokenizer:
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
    def tokenize(self, text):
        output = []
        for token in self.basic_tokenizer.tokenize(text):
            output.extend(self.wordpiece_tokenizer.tokenize(token))
        return output
    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

class BasicTokenizer:
    def __init__(self, do_lower_case=True): self.do_lower_case = do_lower_case
    def tokenize(self, text):
        text = self._clean_text(convert_to_unicode(text))
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = self._run_strip_accents(token.lower())
            split_tokens.extend(self._run_split_on_punc(token))
        return whitespace_tokenize(" ".join(split_tokens))
    def _run_strip_accents(self, text):
        return "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")
    def _run_split_on_punc(self, text):
        chars, output, start_new = list(text), [], True
        for char in chars:
            if _is_punctuation(char):
                output.append([char]); start_new = True
            else:
                if start_new: output.append([])
                start_new = False; output[-1].append(char)
        return ["".join(x) for x in output]
    def _tokenize_chinese_chars(self, text):
        out = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp): out.extend((" ", char, " "))
            else: out.append(char)
        return "".join(out)
    def _clean_text(self, text):
        out = []
        for char in text:
            cp = ord(char)
            if cp in (0, 0xfffd) or _is_control(char): continue
            out.append(" " if _is_whitespace(char) else char)
        return "".join(out)

class WordpieceTokenizer:
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab, self.unk_token = vocab, unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
    def tokenize(self, text):
        output = []
        for token in whitespace_tokenize(convert_to_unicode(text)):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output.append(self.unk_token); continue
            start, sub_tokens, bad = 0, [], False
            while start < len(chars):
                end, cur = len(chars), None
                while start < end:
                    piece = "".join(chars[start:end])
                    if start > 0: piece = "##" + piece
                    if piece in self.vocab: cur = piece; break
                    end -= 1
                if cur is None: bad = True; break
                sub_tokens.append(cur); start = end
            output.extend([self.unk_token] if bad else sub_tokens)
        return output

def _is_whitespace(char):
    return char in " \t\n\r" or unicodedata.category(char) == "Zs"
def _is_control(char):
    if char in "\t\n\r": return False
    return unicodedata.category(char) in ("Cc", "Cf")
def _is_punctuation(char):
    cp = ord(char)
    return (33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or
            123 <= cp <= 126 or unicodedata.category(char).startswith("P"))
def _is_chinese_char(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F or
            0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or
            0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)
PY

  cat > "$WORK/bert/make_features.py" <<'PY'
import argparse, collections, pickle
from create_squad_data import InputFeatures, read_squad_examples
import tokenization

def max_context(spans, cur, pos):
    best_score, best = None, None
    for i, span in enumerate(spans):
        end = span.start + span.length - 1
        if pos < span.start or pos > end: continue
        score = min(pos - span.start, end - pos) + 0.01 * span.length
        if best_score is None or score > best_score: best_score, best = score, i
    return cur == best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--vocab', required=True)
    ap.add_argument('--output', required=True); ap.add_argument('--max_examples', type=int)
    a = ap.parse_args()
    tokenizer = tokenization.FullTokenizer(a.vocab, do_lower_case=True)
    examples = read_squad_examples(a.data, False, False)
    if a.max_examples: examples = examples[:a.max_examples]
    Span = collections.namedtuple('DocSpan', ['start', 'length'])
    features, unique_id = [], 1000000000
    for example_index, example in enumerate(examples):
        query = tokenizer.tokenize(example.question_text)[:64]
        tok_to_orig, all_doc = [], []
        for i, token in enumerate(example.doc_tokens):
            for sub in tokenizer.tokenize(token):
                tok_to_orig.append(i); all_doc.append(sub)
        max_doc = 384 - len(query) - 3
        spans, start = [], 0
        while start < len(all_doc):
            length = min(max_doc, len(all_doc) - start)
            spans.append(Span(start, length))
            if start + length == len(all_doc): break
            start += min(length, 128)
        for span_index, span in enumerate(spans):
            tokens, segments, to_orig, is_max = ['[CLS]'], [0], {}, {}
            tokens.extend(query); segments.extend([0] * len(query))
            tokens.append('[SEP]'); segments.append(0)
            for i in range(span.length):
                split_index = span.start + i
                pos = len(tokens)
                to_orig[pos] = tok_to_orig[split_index]
                is_max[pos] = max_context(spans, span_index, split_index)
                tokens.append(all_doc[split_index]); segments.append(1)
            tokens.append('[SEP]'); segments.append(1)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(ids)
            pad = 384 - len(ids)
            ids.extend([0] * pad); mask.extend([0] * pad); segments.extend([0] * pad)
            assert len(ids) == len(mask) == len(segments) == 384
            features.append(InputFeatures(unique_id, example_index, span_index,
                tokens, to_orig, is_max, ids, mask, segments))
            unique_id += 1
    with open(a.output, 'wb') as f: pickle.dump(features, f, protocol=4)
    print('examples:', len(examples))
    print('features:', len(features))

if __name__ == '__main__': main()
PY

  BERT_WORK="$WORK/bert" "$VENV/bin/python" - <<'PY'
import os
from pathlib import Path
p = Path(os.environ["BERT_WORK"]) / "pytorch_SUT.py"
s = p.read_text()
if 'import pickle\n' not in s:
    s = s.replace('import json\n', 'import json\nimport pickle\n')
device_line = '        print("pytorch device: {}".format(self.dev))\n'
if device_line not in s:
    s = s.replace('        self.version = transformers.__version__\n',
                  '        self.version = transformers.__version__\n' + device_line)
old_qsl = '        self.qsl = get_squad_QSL(args.max_examples)\n'
if old_qsl in s:
    s = s.replace(old_qsl,
                  '        feature_path = os.environ["FEATURE_RECORDS"]\n        with open(feature_path, "rb") as feature_file:\n            feature_records = pickle.load(feature_file)\n        self.qsl = get_squad_QSL(args.max_examples, feature_records=feature_records)\n')
p.write_text(s)
PY
}

stage_preprocess() {
  if [[ -f "$WORK/preprocess.success" ]]; then return; fi
  write_helpers
  {
    echo "stage: preprocess"
    feature_args=()
    if [[ -n "${REPRO_MAX_EXAMPLES:-}" ]]; then
      feature_args+=(--max_examples "$REPRO_MAX_EXAMPLES")
    fi
    "$VENV/bin/python" "$WORK/bert/make_features.py" \
      --data "$SQUAD_DIR/dev-v1.1.json" --vocab "$MODEL_DIR/vocab.txt" \
      --output "$WORK/features.pkl" "${feature_args[@]}"
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/vocab.txt" "$WORK/features.pkl"
  } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  touch "$WORK/preprocess.success"
}

stage_run() {
  write_helpers
  rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
        "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" \
        "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/run.log" "$OUTPUT_DIR/evaluator.log" \
        "$WORK/run.success"
  {
    echo "stage: run"
    echo "gpu memory immediately before run:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    cd "$WORK/bert"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export FEATURE_RECORDS="$WORK/features.pkl"
    export LOG_PATH="$OUTPUT_DIR"
    export SKIP_VERIFY_ACCURACY=1
    run_args=()
    if [[ -n "${REPRO_MAX_EXAMPLES:-}" ]]; then
      run_args+=(--max_examples "$REPRO_MAX_EXAMPLES")
    fi
    "$VENV/bin/python" run.py --backend=pytorch --accuracy "${run_args[@]}"
  } 2>&1 | tee "$OUTPUT_DIR/run.log"
  {
    cd "$WORK/bert"
    eval_args=()
    if [[ -n "${REPRO_MAX_EXAMPLES:-}" ]]; then
      eval_args+=(--max_examples "$REPRO_MAX_EXAMPLES")
    fi
    "$VENV/bin/python" accuracy-squad.py \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --out_file "$WORK/predictions.json" \
      --feature_records "$WORK/features.pkl" "${eval_args[@]}" \
      > "$OUTPUT_DIR/accuracy.txt" 2> "$OUTPUT_DIR/evaluator.stderr.log"
  }
  tee "$OUTPUT_DIR/evaluator.log" < "$OUTPUT_DIR/accuracy.txt"
  cat "$OUTPUT_DIR/evaluator.stderr.log" >&2
  touch "$WORK/run.success"
}

run_stage() {
  case "$1" in
    setup) stage_setup ;;
    preprocess) stage_setup; stage_preprocess ;;
    run) stage_setup; stage_preprocess; stage_run ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}

if [[ -n "$STAGE" ]]; then
  case "$STAGE" in
    setup) stage_setup ;;
    preprocess) stage_preprocess ;;
    run) stage_run ;;
    *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
  esac
elif [[ -n "$FROM" ]]; then
  case "$FROM" in
    setup) rm -f "$WORK/setup.success" "$WORK/preprocess.success" "$WORK/run.success"; stage_setup; stage_preprocess; stage_run ;;
    preprocess) rm -f "$WORK/preprocess.success" "$WORK/run.success"; stage_preprocess; stage_run ;;
    run) rm -f "$WORK/run.success"; stage_run ;;
    *) echo "unknown stage: $FROM" >&2; exit 2 ;;
  esac
else
  stage_setup
  stage_preprocess
  stage_run
fi
