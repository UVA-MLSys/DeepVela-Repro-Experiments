#!/usr/bin/env bash
set -Eeuo pipefail

TARGET=""
ONLY_STAGE=""
FROM_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET=${2:?missing target id}; shift 2 ;;
    --stage) ONLY_STAGE=${2:?missing stage}; shift 2 ;;
    --from) FROM_STAGE=${2:?missing stage}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "supported target: bert-squad" >&2
  exit 2
fi
if [[ -n "$ONLY_STAGE" && -n "$FROM_STAGE" ]]; then
  echo "--stage and --from are mutually exclusive" >&2
  exit 2
fi

ROOT=$(pwd -P)
REPO="$ROOT/repo"
WORK="$ROOT/$TARGET"
OUT=${OUTPUT_DIR:-"$WORK/out"}
STATE="$WORK/state"
VENV="$WORK/venv"
PY="$VENV/bin/python"
BERT="$REPO/language/bert"
CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
SQUAD_DIR=${SQUAD_DIR:-"$CACHE_ROOT/bert-squad/data"}
MODEL_DIR=${MODEL_DIR:-"$CACHE_ROOT/bert-squad/model"}
DEEPLEARNING_EXAMPLES_DIR=${DEEPLEARNING_EXAMPLES_DIR:-"$CACHE_ROOT/bert-squad/DeepLearningExamples-c2"}
export UV_CACHE_DIR=${UV_CACHE_DIR:-"$CACHE_ROOT/uv"}
export UV_EXCLUDE_NEWER=2026-11-10

case "$OUT" in
  /|""|"$ROOT") echo "unsafe OUTPUT_DIR: $OUT" >&2; exit 2 ;;
esac
[[ -d "$REPO" ]] || { echo "missing repository: $REPO" >&2; exit 2; }
mkdir -p "$OUT" "$STATE"

stages=(environment preprocess run)
valid_stage() {
  local candidate=$1
  local item
  for item in "${stages[@]}"; do [[ "$candidate" == "$item" ]] && return 0; done
  return 1
}
if [[ -n "$ONLY_STAGE" ]]; then valid_stage "$ONLY_STAGE" || { echo "bad stage: $ONLY_STAGE" >&2; exit 2; }; fi
if [[ -n "$FROM_STAGE" ]]; then valid_stage "$FROM_STAGE" || { echo "bad stage: $FROM_STAGE" >&2; exit 2; }; fi

if [[ -n "$FROM_STAGE" ]]; then
  clearing=0
  for item in "${stages[@]}"; do
    [[ "$item" == "$FROM_STAGE" ]] && clearing=1
    if [[ $clearing -eq 1 ]]; then rm -f "$STATE/$item.ok"; fi
  done
  if [[ "$FROM_STAGE" == "environment" || "$FROM_STAGE" == "preprocess" ]]; then
    rm -f "$WORK/features.pickle"
  fi
  rm -f "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" \
    "$OUT/mlperf_log_summary.txt" "$OUT/mlperf_log_trace.json" \
    "$OUT/accuracy.txt" "$OUT/predictions.json" "$OUT/run.log" \
    "$OUT/evaluator.log" "$OUT/gpu-memory.log"
fi

run_logged() {
  local logfile=$1
  shift
  "$@" 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  return "$rc"
}

stage_environment() {
  echo "stage=environment"
  [[ -f "$SQUAD_DIR/dev-v1.1.json" ]]
  [[ -f "$MODEL_DIR/model.pytorch" ]]
  [[ -f "$MODEL_DIR/vocab.txt" ]]
  [[ -f "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py" ]]
  echo "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9  $SQUAD_DIR/dev-v1.1.json" | sha256sum -c -
  echo "71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f  $MODEL_DIR/model.pytorch" | sha256sum -c -
  echo "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3  $MODEL_DIR/vocab.txt" | sha256sum -c -
  if [[ ! -x "$PY" ]]; then
    uv venv --python 3.11 "$VENV"
  fi
  uv pip install --python "$PY" \
    'setuptools==75.6.0' 'wheel==0.45.1' 'pybind11==2.13.6' \
    'numpy==1.26.4' 'six==1.16.0' 'absl-py==2.1.0' \
    'boto3==1.35.76' 'torch==2.5.1' 'transformers==4.37.2'
  uv pip install --python "$PY" --no-build-isolation "$REPO/loadgen"
  "$PY" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
}

stage_preprocess() {
  echo "stage=preprocess"
  if [[ -d "$BERT/DeepLearningExamples" && ! -L "$BERT/DeepLearningExamples" ]]; then
    rm -f "$BERT/DeepLearningExamples/DeepLearningExamples-c2"
    rmdir "$BERT/DeepLearningExamples"
  fi
  ln -sfn "$DEEPLEARNING_EXAMPLES_DIR" "$BERT/DeepLearningExamples"
  "$PY" - "$BERT/squad_QSL.py" "$BERT/pytorch_SUT.py" <<'PY'
from pathlib import Path
import sys
qsl = Path(sys.argv[1])
sut = Path(sys.argv[2])
q = qsl.read_text()
old = '''        if feature_records is None:
            raise RuntimeError(
                "The packaged evidence does not specify the reference input presentation; "
                "supply feature records."
            )
        eval_features = list(feature_records)'''
new = '''        if feature_records is None:
            import os
            import pickle
            feature_path = os.environ.get("FEATURE_RECORDS")
            if not feature_path:
                raise RuntimeError("FEATURE_RECORDS is required")
            with open(feature_path, "rb") as feature_file:
                feature_records = pickle.load(feature_file)
        eval_features = list(feature_records)'''
if old in q:
    qsl.write_text(q.replace(old, new))
elif new not in q:
    raise SystemExit("unexpected squad_QSL.py content")
s = sut.read_text()
needle = '''        self.model.to(self.dev)
        self.model.eval()'''
device_line = '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n'
s = s.replace(device_line, "")
if needle in s:
    sut.write_text(s.replace(needle, needle + "\n" + device_line.rstrip("\n")))
else:
    raise SystemExit("unexpected pytorch_SUT.py content")
PY
  if [[ ! -f "$WORK/features.pickle" ]]; then
    (
      cd "$BERT"
      "$PY" - "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/vocab.txt" "$WORK/features.pickle" <<'PY'
import collections
import pickle
import sys

sys.path.insert(0, "DeepLearningExamples/PyTorch/LanguageModeling/BERT")
import tokenization
from create_squad_data import InputFeatures, read_squad_examples

data_path, vocab_path, output_path = sys.argv[1:]
max_seq_length = 384
doc_stride = 128
max_query_length = 64
tokenizer = tokenization.BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
examples = read_squad_examples(data_path, is_training=False)
features = []
unique_id = 1000000000

def is_max_context(spans, current_span, position):
    best_score = None
    best_index = None
    for i, span in enumerate(spans):
        start = span["start"]
        end = start + span["length"] - 1
        if position < start or position > end:
            continue
        left = position - start
        right = end - position
        score = min(left, right) + 0.01 * span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_index = i
    return current_span == best_index

for example_index, example in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)
    query_tokens = query_tokens[:max_query_length]
    orig_to_tok_index = []
    tok_to_orig_index = []
    all_doc_tokens = []
    for orig_index, token in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(orig_index)
            all_doc_tokens.append(sub_token)
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    spans = []
    start = 0
    while start < len(all_doc_tokens):
        length = min(max_tokens_for_doc, len(all_doc_tokens) - start)
        spans.append({"start": start, "length": length})
        if start + length == len(all_doc_tokens):
            break
        start += min(length, doc_stride)
    for span_index, span in enumerate(spans):
        tokens = ["[CLS]"] + query_tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        token_to_orig_map = {}
        token_is_max_context = {}
        for i in range(span["length"]):
            split_index = span["start"] + i
            position = len(tokens)
            token_to_orig_map[position] = tok_to_orig_index[split_index]
            token_is_max_context[position] = is_max_context(spans, span_index, split_index)
            tokens.append(all_doc_tokens[split_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        if not (len(input_ids) == len(input_mask) == len(segment_ids) == max_seq_length):
            raise RuntimeError("invalid feature length")
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

with open(output_path, "wb") as output:
    pickle.dump(features, output, protocol=4)
print("examples", len(examples))
print("features", len(features))
print("sequence_length", max_seq_length)
if len(examples) != 10570 or len(features) < len(examples):
    raise RuntimeError("unexpected input coverage")
PY
    )
  fi
  "$PY" - "$WORK/features.pickle" <<'PY'
import pickle, sys
sys.path.insert(0, "repo/language/bert")
sys.path.insert(0, "repo/language/bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT")
with open(sys.argv[1], "rb") as f:
    features = pickle.load(f)
print("validated_features", len(features))
assert features and all(len(x.input_ids) == 384 for x in features)
PY
}

stage_run() {
  echo "stage=run"
  local max_args=()
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    max_args=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader | tee "$OUT/gpu-memory.log"
  local free_mb
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || (( free_mb < 8000 )); then
    echo "insufficient free GPU memory: ${free_mb:-unknown} MiB" >&2
    return 1
  fi
  (
    cd "$BERT"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export FEATURE_RECORDS="$WORK/features.pickle"
    export LOG_PATH="$OUT"
    export SKIP_VERIFY_ACCURACY=1
    "$PY" run.py --backend=pytorch --accuracy "${max_args[@]}"
  )
  grep -q '^pytorch device: ' "$OUT/run.log" || {
    echo "run log lacks required device prefix" >&2
    return 1
  }
  (
    cd "$BERT"
    "$PY" accuracy-squad.py \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUT/mlperf_log_accuracy.json" \
      --out_file "$OUT/predictions.json" \
      --feature_records "$WORK/features.pickle" \
      "${max_args[@]}"
  ) 2>&1 | tee "$OUT/evaluator.log" "$OUT/accuracy.txt"
  [[ -s "$OUT/mlperf_log_accuracy.json" ]]
  [[ -s "$OUT/mlperf_log_detail.txt" ]]
  [[ -s "$OUT/accuracy.txt" ]]
  "$PY" - "$OUT/accuracy.txt" <<'PY'
import json, sys
lines = open(sys.argv[1]).read().splitlines()
objects = []
for line in lines:
    try:
        value = json.loads(line)
    except Exception:
        continue
    if isinstance(value, dict) and "exact_match" in value and "f1" in value:
        objects.append(value)
if not objects:
    raise SystemExit("evaluator metrics not found")
print("validated_metrics", json.dumps(objects[-1], sort_keys=True))
PY
}

execute_stage() {
  local stage=$1
  [[ -f "$STATE/$stage.ok" ]] && { echo "skip completed stage=$stage"; return 0; }
  case "$stage" in
    environment) run_logged "$OUT/build.log" stage_environment ;;
    preprocess) run_logged "$OUT/preprocess.log" stage_preprocess ;;
    run) run_logged "$OUT/run.log" stage_run ;;
  esac
  touch "$STATE/$stage.ok"
}

if [[ -n "$ONLY_STAGE" ]]; then
  execute_stage "$ONLY_STAGE"
else
  started=1
  if [[ -n "$FROM_STAGE" ]]; then started=0; fi
  for item in "${stages[@]}"; do
    if [[ -n "$FROM_STAGE" && "$item" == "$FROM_STAGE" ]]; then started=1; fi
    [[ $started -eq 1 ]] && execute_stage "$item"
  done
fi
