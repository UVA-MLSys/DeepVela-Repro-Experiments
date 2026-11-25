#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples-c2"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

target=
stage=
from_stage=
max_examples=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --stage) stage="$2"; shift 2 ;;
    --from) from_stage="$2"; shift 2 ;;
    --max-examples) max_examples="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$target" == "bert-squad" ]] || { echo "expected --target bert-squad" >&2; exit 2; }
[[ -d repo/language/bert && -d repo/loadgen ]] || { echo "run beside the fresh repo directory" >&2; exit 2; }

ROOT=$(pwd -P)
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/$target/out"}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd -P)
WORK="$ROOT/$target/work"
VENV="$ROOT/.venv-bert-squad"

run_env() {
  if [[ -f "$ROOT/.bert-squad-env.success" ]]; then return; fi
  {
    uv venv --python 3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" \
      'setuptools<81' 'numpy==1.26.4' 'six==1.17.0' 'absl-py==2.3.1' \
      'pybind11==3.0.1' 'torch==2.5.1' 'transformers==4.46.2'
    uv pip install --python "$VENV/bin/python" ./repo/loadgen
    "$VENV/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("loadgen:", mlperf_loadgen.__file__)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
PY
  } 2>&1 | tee "$OUTPUT_DIR/build.log"
  touch "$ROOT/.bert-squad-env.success"
}

run_prepare() {
  if [[ -f "$WORK/prepare.success" ]]; then return; fi
  mkdir -p "$WORK/repo"
  cp -a repo/. "$WORK/repo/"
  BERT="$WORK/repo/language/bert"
  mkdir -p "$BERT/build/data/bert_model" "$BERT/build/result"
  ln -sfn "$SQUAD_DIR/dev-v1.1.json" "$BERT/build/data/dev-v1.1.json"
  ln -sfn "$MODEL_DIR/model.pytorch" "$BERT/build/data/bert_model/model.pytorch"
  ln -sfn "$MODEL_DIR/vocab.txt" "$BERT/build/data/bert_model/vocab.txt"
  mkdir -p "$BERT/DeepLearningExamples/PyTorch/LanguageModeling/BERT"
  cp "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py" \
     "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/file_utils.py" \
    "$BERT/DeepLearningExamples/PyTorch/LanguageModeling/BERT/"
  "$VENV/bin/python" - "$BERT/DeepLearningExamples/PyTorch/LanguageModeling/BERT/tokenization.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text().replace(
    'from file_utils import cached_path',
    'def cached_path(path, cache_dir=None):\n'
    '    if not os.path.exists(path):\n'
    '        raise FileNotFoundError(path)\n'
    '    return path')
p.write_text(s)
PY

  "$VENV/bin/python" - "$BERT" <<'PY' 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
import os, pickle, sys
bert = sys.argv[1]
sys.path[:0] = [bert, os.path.join(bert, "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT")]
import tokenization
from create_squad_data import InputFeatures, read_squad_examples

max_seq_length, doc_stride, max_query_length = 384, 128, 64
tokenizer = tokenization.BertTokenizer(
    vocab_file=os.path.join(bert, "build/data/bert_model/vocab.txt"),
    do_lower_case=True)
examples = read_squad_examples(
    os.path.join(bert, "build/data/dev-v1.1.json"), is_training=False)

def best_context(spans, current, position):
    best_score, best_index = None, None
    for i, span in enumerate(spans):
        end = span[0] + span[1] - 1
        if position < span[0] or position > end:
            continue
        score = min(position - span[0], end - position) + 0.01 * span[1]
        if best_score is None or score > best_score:
            best_score, best_index = score, i
    return current == best_index

features = []
unique_id = 1000000000
for example_index, example in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)[:max_query_length]
    tok_to_orig_index, all_doc_tokens = [], []
    for orig_index, token in enumerate(example.doc_tokens):
        for sub_token in tokenizer.tokenize(token):
            tok_to_orig_index.append(orig_index)
            all_doc_tokens.append(sub_token)
    max_doc = max_seq_length - len(query_tokens) - 3
    spans, start = [], 0
    while start < len(all_doc_tokens):
        length = min(max_doc, len(all_doc_tokens) - start)
        spans.append((start, length))
        if start + length == len(all_doc_tokens):
            break
        start += min(length, doc_stride)
    for doc_span_index, (doc_start, doc_length) in enumerate(spans):
        tokens = ["[CLS]"] + query_tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        token_to_orig_map, token_is_max_context = {}, {}
        for i in range(doc_length):
            split_index = doc_start + i
            token_index = len(tokens)
            token_to_orig_map[token_index] = tok_to_orig_index[split_index]
            token_is_max_context[token_index] = best_context(spans, doc_span_index, split_index)
            tokens.append(all_doc_tokens[split_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        pad = max_seq_length - len(input_ids)
        input_ids += [0] * pad
        input_mask += [0] * pad
        segment_ids += [0] * pad
        assert len(input_ids) == len(input_mask) == len(segment_ids) == max_seq_length
        features.append(InputFeatures(
            unique_id=unique_id, example_index=example_index,
            doc_span_index=doc_span_index, tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids))
        unique_id += 1

out = os.path.join(bert, "build/data/eval_features.pickle")
with open(out, "wb") as f:
    pickle.dump(features, f, protocol=4)
print("examples:", len(examples))
print("feature records:", len(features))
print("feature records path:", out)
PY

  "$VENV/bin/python" - "$BERT/pytorch_SUT.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
s = s.replace('import json\n', 'import json\nimport pickle\n')
s = s.replace(
    '        self.model.load_state_dict(torch.load(model_file), strict=False)\n',
    '        self.model.load_state_dict(torch.load(model_file, map_location=self.dev, weights_only=True), strict=False)\n'
    '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n')
s = s.replace(
    '        self.qsl = get_squad_QSL(args.max_examples)\n',
    '        with open(os.environ["FEATURE_RECORDS"], "rb") as feature_file:\n'
    '            feature_records = pickle.load(feature_file)\n'
    '        if args.max_examples:\n'
    '            feature_records = feature_records[:args.max_examples]\n'
    '        self.qsl = get_squad_QSL(feature_records=feature_records)\n')
p.write_text(s)
PY
  touch "$WORK/prepare.success"
}

run_gpu() {
  BERT="$WORK/repo/language/bert"
  rm -f "$OUTPUT_DIR"/mlperf_log_*.json "$OUTPUT_DIR"/mlperf_log_*.txt \
        "$OUTPUT_DIR"/accuracy.txt "$OUTPUT_DIR"/run.log "$OUTPUT_DIR"/evaluator.log
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader | tee "$OUTPUT_DIR/gpu-memory-before.log"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  if (( free_mib < 12000 )); then
    echo "insufficient free GPU memory: ${free_mib} MiB (need 12000 MiB)" | tee -a "$OUTPUT_DIR/gpu-memory-before.log"
    exit 1
  fi
  args=(--backend=pytorch --accuracy)
  if [[ -n "$max_examples" ]]; then args+=(--max_examples "$max_examples"); fi
  (
    cd "$BERT"
    export LOG_PATH="$OUTPUT_DIR"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export FEATURE_RECORDS="$BERT/build/data/eval_features.pickle"
    export SKIP_VERIFY_ACCURACY=1
    "$VENV/bin/python" run.py "${args[@]}"
  ) 2>&1 | tee "$OUTPUT_DIR/run.log"

  eval_args=(--val_data "$SQUAD_DIR/dev-v1.1.json"
             --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json"
             --out_file "$BERT/build/result/predictions.json"
             --feature_records "$BERT/build/data/eval_features.pickle")
  if [[ -n "$max_examples" ]]; then eval_args+=(--max_examples "$max_examples"); fi
  (
    cd "$BERT"
    "$VENV/bin/python" accuracy-squad.py "${eval_args[@]}"
  ) 2>&1 | tee "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"
  touch "$OUTPUT_DIR/run.success"
}

if [[ -n "$from_stage" ]]; then
  case "$from_stage" in
    env)
      rm -rf "$VENV" "$WORK"
      rm -f "$ROOT/.bert-squad-env.success" "$OUTPUT_DIR/run.success"
      ;;
    prepare) rm -f "$WORK/prepare.success" "$OUTPUT_DIR/run.success" ;;
    gpu) rm -f "$OUTPUT_DIR/run.success" ;;
    *) echo "unknown stage: $from_stage" >&2; exit 2 ;;
  esac
fi

if [[ -n "$stage" ]]; then
  case "$stage" in
    env) run_env ;;
    prepare) run_env; run_prepare ;;
    gpu) run_env; run_prepare; run_gpu ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
else
  run_env
  run_prepare
  run_gpu
fi
