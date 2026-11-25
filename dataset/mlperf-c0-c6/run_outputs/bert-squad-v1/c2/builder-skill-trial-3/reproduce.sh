#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target bert-squad [--stage environment|prepare|run] [--from environment|prepare|run] [--max-examples N]" >&2
  exit 2
}

TARGET=""
ONLY_STAGE=""
FROM_STAGE=""
MAX_EXAMPLES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET=${2:-}; shift 2 ;;
    --stage) ONLY_STAGE=${2:-}; shift 2 ;;
    --from) FROM_STAGE=${2:-}; shift 2 ;;
    --max-examples) MAX_EXAMPLES=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$TARGET" == "bert-squad" ]] || usage
[[ -z "$ONLY_STAGE" || -z "$FROM_STAGE" ]] || usage
if [[ -n "$ONLY_STAGE" && ! "$ONLY_STAGE" =~ ^(environment|prepare|run)$ ]]; then usage; fi
if [[ -n "$FROM_STAGE" && ! "$FROM_STAGE" =~ ^(environment|prepare|run)$ ]]; then usage; fi
if [[ -n "$MAX_EXAMPLES" && ! "$MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then usage; fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET_DIR="$ROOT/$TARGET"
OUT_DIR=${OUTPUT_DIR:-"$TARGET_DIR/out"}
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT/${OUT_DIR#./}"
fi
WORK_DIR="$TARGET_DIR/work"
SOURCE_DIR="$WORK_DIR/repo"
VENV_DIR="$WORK_DIR/venv"
MARKER_DIR="$TARGET_DIR/markers"

CACHE_ROOT=${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}
SQUAD_DIR=${SQUAD_DIR:-$CACHE_ROOT/bert-squad/data}
MODEL_DIR=${MODEL_DIR:-$CACHE_ROOT/bert-squad/model}
DEEPLEARNING_EXAMPLES_DIR=${DEEPLEARNING_EXAMPLES_DIR:-$CACHE_ROOT/bert-squad/DeepLearningExamples-c2}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE_ROOT/uv}
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$TARGET_DIR" "$OUT_DIR" "$MARKER_DIR"

clear_from() {
  case "$1" in
    environment)
      rm -rf "$WORK_DIR"
      rm -f "$MARKER_DIR/environment.success" "$MARKER_DIR/prepare.success" "$MARKER_DIR/run.success"
      rm -f "$OUT_DIR/build.log" "$OUT_DIR/prepare.log" "$OUT_DIR/gpu_check.log" "$OUT_DIR/run.log" "$OUT_DIR/evaluator.log" "$OUT_DIR/accuracy.txt" "$OUT_DIR/mlperf_log_accuracy.json" "$OUT_DIR/mlperf_log_detail.txt" "$OUT_DIR/mlperf_log_summary.txt" "$OUT_DIR/mlperf_log_trace.json" "$OUT_DIR/predictions.json"
      ;;
    prepare)
      rm -f "$MARKER_DIR/prepare.success" "$MARKER_DIR/run.success"
      rm -f "$OUT_DIR/prepare.log" "$OUT_DIR/gpu_check.log" "$OUT_DIR/run.log" "$OUT_DIR/evaluator.log" "$OUT_DIR/accuracy.txt" "$OUT_DIR/mlperf_log_accuracy.json" "$OUT_DIR/mlperf_log_detail.txt" "$OUT_DIR/mlperf_log_summary.txt" "$OUT_DIR/mlperf_log_trace.json" "$OUT_DIR/predictions.json"
      rm -f "$WORK_DIR/features.pickle"
      ;;
    run)
      rm -f "$MARKER_DIR/run.success"
      rm -f "$OUT_DIR/gpu_check.log" "$OUT_DIR/run.log" "$OUT_DIR/evaluator.log" "$OUT_DIR/accuracy.txt" "$OUT_DIR/mlperf_log_accuracy.json" "$OUT_DIR/mlperf_log_detail.txt" "$OUT_DIR/mlperf_log_summary.txt" "$OUT_DIR/mlperf_log_trace.json" "$OUT_DIR/predictions.json"
      ;;
  esac
}

stage_environment() {
  if [[ -f "$MARKER_DIR/environment.success" ]]; then return; fi
  for path in "$ROOT/repo" "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt" "$DEEPLEARNING_EXAMPLES_DIR" "$UV_CACHE_DIR"; do
    [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
  done
  mkdir -p "$WORK_DIR"
  {
    echo "stage: environment"
    echo "python: 3.11"
    echo "UV_CACHE_DIR=$UV_CACHE_DIR"
    echo "UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER"
    cp -a "$ROOT/repo" "$SOURCE_DIR"
    uv venv --python 3.11 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" \
      'torch==2.4.1' 'transformers==4.44.2' 'numpy==1.26.4' \
      'absl-py==2.3.1' 'six==1.17.0' 'boto3==1.40.76' 'pybind11==2.11.1' \
      'setuptools==80.9.0' 'wheel==0.45.1'
    uv pip install --python "$VENV_DIR/bin/python" "$SOURCE_DIR/loadgen"
    "$VENV_DIR/bin/python" - <<'PY'
import torch, transformers, mlperf_loadgen
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_runtime", torch.version.cuda)
print("loadgen", mlperf_loadgen.__file__)
PY
  } 2>&1 | tee "$OUT_DIR/build.log"
  touch "$MARKER_DIR/environment.success"
}

write_repair_sources() {
  local bert_dir="$SOURCE_DIR/language/bert"
  rm -rf "$bert_dir/DeepLearningExamples"
  ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$bert_dir/DeepLearningExamples"
  mkdir -p "$bert_dir/build/data" "$bert_dir/build/result"
  ln -sfn "$SQUAD_DIR/dev-v1.1.json" "$bert_dir/build/data/dev-v1.1.json"
  ln -sfn "$MODEL_DIR" "$bert_dir/build/data/bert_model"

  cat > "$bert_dir/build_features.py" <<'PY'
#!/usr/bin/env python3
import argparse
import collections
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.path.dirname(__file__))
import tokenization
from create_squad_data import InputFeatures, read_squad_examples
from transformers import BertTokenizer

def is_max_context(doc_spans, current_span_index, position):
    best_score = None
    best_span_index = None
    for span_index, span in enumerate(doc_spans):
        end = span.start + span.length - 1
        if position < span.start or position > end:
            continue
        left = position - span.start
        right = end - position
        score = min(left, right) + 0.01 * span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return current_span_index == best_span_index

def convert(examples, tokenizer, max_seq_length=384, doc_stride=128, max_query_length=64):
    DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    features = []
    unique_id = 1000000000
    for example_index, example in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[:max_query_length]
        all_doc_tokens = []
        tok_to_orig_index = []
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
            tokens = ["[CLS]"] + query_tokens + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            token_to_orig_map = {}
            token_is_max_context = {}
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_index = len(tokens)
                token_to_orig_map[token_index] = tok_to_orig_index[split_token_index]
                token_is_max_context[token_index] = is_max_context(doc_spans, doc_span_index, split_token_index)
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = max_seq_length - len(input_ids)
            input_ids += [0] * padding
            input_mask += [0] * padding
            segment_ids += [0] * padding
            assert len(input_ids) == len(input_mask) == len(segment_ids) == max_seq_length
            features.append(InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
            ))
            unique_id += 1
    return features

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    examples = read_squad_examples(args.input, is_training=False, version_2_with_negative=False)
    tokenizer = BertTokenizer(vocab_file=args.vocab, do_lower_case=True)
    features = convert(examples, tokenizer)
    with open(args.output, "wb") as stream:
        pickle.dump(features, stream, protocol=4)
    print("examples", len(examples))
    print("features", len(features))
    print("max_seq_length 384")
    print("doc_stride 128")
    print("max_query_length 64")

if __name__ == "__main__":
    main()
PY

  "$VENV_DIR/bin/python" - "$bert_dir/pytorch_SUT.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
s = s.replace("import os\nimport sys\n", "import os\nimport pickle\nimport sys\n")
if 'print("pytorch device: {}".format(self.dev))' not in s:
    s = s.replace(
        "        self.version = transformers.__version__\n",
        "        self.version = transformers.__version__\n        print(\"pytorch device: {}\".format(self.dev))\n",
    )
s = s.replace(
    "        self.model.load_state_dict(torch.load(model_file), strict=False)\n",
    "        state = torch.load(model_file, map_location=self.dev, weights_only=True)\n        incompatible = self.model.load_state_dict(state, strict=False)\n        print(\"missing model keys: {}\".format(len(incompatible.missing_keys)))\n        print(\"unexpected model keys: {}\".format(len(incompatible.unexpected_keys)))\n",
)
s = s.replace(
    "        self.qsl = get_squad_QSL(args.max_examples)\n",
    "        feature_path = os.environ[\"FEATURE_RECORDS\"]\n        with open(feature_path, \"rb\") as feature_file:\n            feature_records = pickle.load(feature_file)\n        self.qsl = get_squad_QSL(args.max_examples, feature_records=feature_records)\n",
)
p.write_text(s)
PY
}

stage_prepare() {
  if [[ -f "$MARKER_DIR/prepare.success" ]]; then return; fi
  [[ -f "$MARKER_DIR/environment.success" ]] || { echo "environment stage has not succeeded" >&2; exit 1; }
  {
    echo "stage: prepare"
    write_repair_sources
    cd "$SOURCE_DIR/language/bert"
    "$VENV_DIR/bin/python" build_features.py \
      --input "$SQUAD_DIR/dev-v1.1.json" \
      --vocab "$MODEL_DIR/vocab.txt" \
      --output "$WORK_DIR/features.pickle"
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt" "$WORK_DIR/features.pickle"
  } 2>&1 | tee "$OUT_DIR/prepare.log"
  touch "$MARKER_DIR/prepare.success"
}

stage_run() {
  if [[ -f "$MARKER_DIR/run.success" ]]; then return; fi
  [[ -f "$MARKER_DIR/prepare.success" ]] || { echo "prepare stage has not succeeded" >&2; exit 1; }
  rm -f "$OUT_DIR/mlperf_log_accuracy.json" "$OUT_DIR/mlperf_log_detail.txt" "$OUT_DIR/mlperf_log_summary.txt" "$OUT_DIR/mlperf_log_trace.json" "$OUT_DIR/predictions.json" "$OUT_DIR/accuracy.txt" "$OUT_DIR/evaluator.log" "$OUT_DIR/run.log"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free --format=csv,noheader,nounits | tee "$OUT_DIR/gpu_check.log"
  local max_args=()
  if [[ -n "$MAX_EXAMPLES" ]]; then max_args=(--max_examples "$MAX_EXAMPLES"); fi
  (
    cd "$SOURCE_DIR/language/bert"
    export LOG_PATH="$OUT_DIR"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export FEATURE_RECORDS="$WORK_DIR/features.pickle"
    export SKIP_VERIFY_ACCURACY=1
    export CUDA_VISIBLE_DEVICES=0
    "$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
  ) 2>&1 | tee "$OUT_DIR/run.log"
  (
    cd "$SOURCE_DIR/language/bert"
    export PYTHONUNBUFFERED=1
    "$VENV_DIR/bin/python" accuracy-squad.py \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUT_DIR/mlperf_log_accuracy.json" \
      --out_file "$OUT_DIR/predictions.json" \
      --feature_records "$WORK_DIR/features.pickle" \
      "${max_args[@]}"
  ) 2>&1 | tee "$OUT_DIR/evaluator.log" "$OUT_DIR/accuracy.txt"
  test -s "$OUT_DIR/mlperf_log_accuracy.json"
  test -s "$OUT_DIR/mlperf_log_detail.txt"
  test -s "$OUT_DIR/accuracy.txt"
  grep -q '^pytorch device: ' "$OUT_DIR/run.log"
  tail -n 1 "$OUT_DIR/accuracy.txt" | "$VENV_DIR/bin/python" -c 'import json,sys; d=json.load(sys.stdin); assert set(d)=={"exact_match","f1"}'
  touch "$MARKER_DIR/run.success"
}

run_stage() {
  case "$1" in
    environment) stage_environment ;;
    prepare) stage_prepare ;;
    run) stage_run ;;
  esac
}

if [[ -n "$FROM_STAGE" ]]; then
  clear_from "$FROM_STAGE"
  case "$FROM_STAGE" in
    environment) run_stage environment; run_stage prepare; run_stage run ;;
    prepare) run_stage prepare; run_stage run ;;
    run) run_stage run ;;
  esac
elif [[ -n "$ONLY_STAGE" ]]; then
  run_stage "$ONLY_STAGE"
else
  run_stage environment
  run_stage prepare
  run_stage run
fi
