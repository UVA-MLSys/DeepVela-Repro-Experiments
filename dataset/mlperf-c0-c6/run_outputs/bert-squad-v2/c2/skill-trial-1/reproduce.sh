#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad [--stage build|preprocess|run|evaluate] [--from build|preprocess|run|evaluate]" >&2
  exit 2
}

target=""
stage=""
from=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="${2:-}"; shift 2 ;;
    --stage) stage="${2:-}"; shift 2 ;;
    --from) from="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$target" == "bert-squad" ]] || usage
[[ -z "$stage" || -z "$from" ]] || usage

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO="$ROOT/repo"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
TARGET_DIR=$(dirname "$OUTPUT_DIR")
WORK="$TARGET_DIR/work"
VENV="$WORK/.venv"
BERT="$WORK/bert"
STATE="$WORK/state"

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DLE_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples-c2"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$WORK" "$STATE"

clear_from() {
  case "$1" in
    build)
      rm -f "$STATE"/{build,preprocess,run,evaluate}.done
      rm -rf "$VENV" "$WORK/loadgen_src" "$BERT"
      rm -f "$OUTPUT_DIR"/*
      ;;
    preprocess)
      rm -f "$STATE"/{preprocess,run,evaluate}.done
      rm -rf "$BERT"
      rm -f "$OUTPUT_DIR"/{preprocess.log,run.log,evaluator.log,accuracy.txt,mlperf_log_accuracy.json,mlperf_log_detail.txt,mlperf_log_summary.txt,mlperf_log_trace.json}
      ;;
    run)
      rm -f "$STATE"/{run,evaluate}.done
      rm -f "$OUTPUT_DIR"/{run.log,evaluator.log,accuracy.txt,mlperf_log_accuracy.json,mlperf_log_detail.txt,mlperf_log_summary.txt,mlperf_log_trace.json}
      ;;
    evaluate)
      rm -f "$STATE/evaluate.done"
      rm -f "$OUTPUT_DIR"/{evaluator.log,accuracy.txt}
      ;;
    *) usage ;;
  esac
}

do_build() {
  [[ ! -f "$STATE/build.done" ]] || return 0
  {
    echo "python: 3.11"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    uv venv --python /home/mi3se/.local/bin/python3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'setuptools<81' wheel 'pybind11==2.11.1' \
      'numpy==1.26.4' 'absl-py==2.3.1' 'six==1.17.0' \
      'boto3==1.40.61' \
      'torch==2.6.0' 'transformers==4.46.3'
    rm -rf "$WORK/loadgen_src"
    cp -a "$REPO/loadgen" "$WORK/loadgen_src"
    uv pip install --python "$VENV/bin/python" --no-build-isolation "$WORK/loadgen_src"
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("mlperf_loadgen: imported")
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
PY
  } 2>&1 | tee "$OUTPUT_DIR/build.log"
  touch "$STATE/build.done"
}

write_feature_builder() {
  cat > "$BERT/build_features.py" <<'PY'
import argparse
import collections
import pickle

import tokenization
from create_squad_data import InputFeatures, read_squad_examples


def is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start or position > end:
            continue
        left = position - doc_span.start
        right = end - position
        score = min(left, right) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index


def convert(examples, tokenizer, max_seq_length=384, doc_stride=128,
            max_query_length=64):
    unique_id = 1000000000
    features = []
    DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    for example_index, example in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[:max_query_length]

        tok_to_orig_index = []
        all_doc_tokens = []
        for orig_index, token in enumerate(example.doc_tokens):
            for sub_token in tokenizer.tokenize(token):
                tok_to_orig_index.append(orig_index)
                all_doc_tokens.append(sub_token)

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = min(len(all_doc_tokens) - start_offset, max_tokens_for_doc)
            doc_spans.append(DocSpan(start=start_offset, length=length))
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
                token_index = len(tokens)
                token_to_orig_map[token_index] = tok_to_orig_index[split_token_index]
                token_is_max_context[token_index] = is_max_context(
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
            features.append(InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids))
            unique_id += 1
    return features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    examples = read_squad_examples(args.data, is_training=False,
                                   version_2_with_negative=False)
    tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab,
                                           do_lower_case=True)
    features = convert(examples, tokenizer)
    with open(args.output, "wb") as f:
        pickle.dump(features, f, protocol=4)
    print("examples:", len(examples))
    print("features:", len(features))
    print("sequence_length:", len(features[0].input_ids))


if __name__ == "__main__":
    main()
PY
}

do_preprocess() {
  [[ ! -f "$STATE/preprocess.done" ]] || return 0
  {
    rm -rf "$BERT"
    mkdir -p "$BERT/DeepLearningExamples/PyTorch/LanguageModeling/BERT" "$BERT/build/result"
    cp -a "$REPO/language/bert/." "$BERT/"
    cp -a "$DLE_DIR/PyTorch/LanguageModeling/BERT/." \
      "$BERT/DeepLearningExamples/PyTorch/LanguageModeling/BERT/"
    write_feature_builder
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    cd "$BERT"
    PYTHONPATH="$BERT/DeepLearningExamples/PyTorch/LanguageModeling/BERT:$BERT" \
      "$VENV/bin/python" build_features.py \
      --data "$SQUAD_DIR/dev-v1.1.json" \
      --vocab "$MODEL_DIR/vocab.txt" \
      --output "$BERT/eval_features.pickle"
    "$VENV/bin/python" - <<'PY'
from pathlib import Path
p = Path("squad_QSL.py")
s = p.read_text()
s = s.replace("import mlperf_loadgen as lg\n", "import mlperf_loadgen as lg\nimport os\nimport pickle\n")
old = '''        if feature_records is None:
            raise RuntimeError(
                "The packaged evidence does not specify the reference input presentation; "
                "supply feature records."
            )
        eval_features = list(feature_records)'''
new = '''        if feature_records is None:
            feature_path = os.environ.get("MLPERF_FEATURE_RECORDS")
            if not feature_path:
                raise RuntimeError("supply feature records")
            with open(feature_path, "rb") as feature_file:
                feature_records = pickle.load(feature_file)
        eval_features = list(feature_records)'''
if old not in s:
    raise RuntimeError("unexpected squad_QSL.py")
p.write_text(s.replace(old, new))

p = Path("pytorch_SUT.py")
s = p.read_text()
needle = '''        self.version = transformers.__version__

        print("Loading PyTorch model...")'''
replacement = '''        self.version = transformers.__version__
        print("pytorch device: {}".format(self.dev))

        print("Loading PyTorch model...")'''
if needle not in s:
    raise RuntimeError("unexpected pytorch_SUT.py")
p.write_text(s.replace(needle, replacement))
PY
    test -s "$BERT/eval_features.pickle"
  } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
  touch "$STATE/preprocess.done"
}

gpu_check() {
  local free
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if (( free < 12000 )); then
    echo "insufficient free GPU memory: ${free} MiB (need at least 12000 MiB)" >&2
    exit 1
  fi
}

do_run() {
  [[ ! -f "$STATE/run.done" ]] || return 0
  {
    gpu_check
    cd "$BERT"
    export LOG_PATH="$OUTPUT_DIR"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export MLPERF_FEATURE_RECORDS="$BERT/eval_features.pickle"
    export SKIP_VERIFY_ACCURACY=1
    export PYTHONPATH="$BERT/DeepLearningExamples/PyTorch/LanguageModeling/BERT:$BERT"
    max_args=()
    if [[ -n "${REPRO_MAX_EXAMPLES:-}" ]]; then
      max_args=(--max_examples "$REPRO_MAX_EXAMPLES")
    fi
    "$VENV/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
  } 2>&1 | tee "$OUTPUT_DIR/run.log"
  test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
  grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
  touch "$STATE/run.done"
}

do_evaluate() {
  [[ ! -f "$STATE/evaluate.done" ]] || return 0
  cd "$BERT"
  export PYTHONPATH="$BERT/DeepLearningExamples/PyTorch/LanguageModeling/BERT:$BERT"
  max_args=()
  if [[ -n "${REPRO_MAX_EXAMPLES:-}" ]]; then
    max_args=(--max_examples "$REPRO_MAX_EXAMPLES")
  fi
  "$VENV/bin/python" accuracy-squad.py \
    --val_data "$SQUAD_DIR/dev-v1.1.json" \
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
    --out_file "$BERT/build/result/predictions.json" \
    --feature_records "$BERT/eval_features.pickle" \
    "${max_args[@]}" \
    2>&1 | tee "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  grep -q '"exact_match"' "$OUTPUT_DIR/accuracy.txt"
  grep -q '"f1"' "$OUTPUT_DIR/accuracy.txt"
  touch "$STATE/evaluate.done"
}

run_sequence() {
  local begin="$1"
  case "$begin" in
    build) do_build; do_preprocess; do_run; do_evaluate ;;
    preprocess) do_preprocess; do_run; do_evaluate ;;
    run) do_run; do_evaluate ;;
    evaluate) do_evaluate ;;
    *) usage ;;
  esac
}

if [[ -n "$from" ]]; then
  clear_from "$from"
  run_sequence "$from"
elif [[ -n "$stage" ]]; then
  case "$stage" in
    build) do_build ;;
    preprocess) do_preprocess ;;
    run) do_run ;;
    evaluate) do_evaluate ;;
    *) usage ;;
  esac
else
  run_sequence build
fi
