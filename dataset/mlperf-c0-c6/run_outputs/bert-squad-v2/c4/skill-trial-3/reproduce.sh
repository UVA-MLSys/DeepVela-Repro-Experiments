#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_CACHE_DIR
export UV_EXCLUDE_NEWER=2026-11-10

TARGET=""
MODE="all"
START_STAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET=${2:?missing target id}; shift 2 ;;
    --stage) MODE=stage; START_STAGE=${2:?missing stage}; shift 2 ;;
    --from) MODE=from; START_STAGE=${2:?missing stage}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "bert-squad" ]] || { echo "unsupported target: $TARGET" >&2; exit 2; }

ROOT=$(cd "$(dirname "$0")" && pwd)
SOURCE_REPO="$ROOT/repo"
TARGET_DIR="$ROOT/$TARGET"
WORK_DIR="$TARGET_DIR/work"
RUN_REPO="$WORK_DIR/repo"
VENV="$WORK_DIR/.venv"
MARKERS="$WORK_DIR/markers"
OUTPUT_DIR=${OUTPUT_DIR:-"$TARGET_DIR/out"}
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT/${OUTPUT_DIR#./}"
fi
mkdir -p "$TARGET_DIR" "$OUTPUT_DIR" "$MARKERS"

stages=(setup preprocess run evaluate)
stage_index() {
  local wanted=$1 i
  for i in "${!stages[@]}"; do
    [[ "${stages[$i]}" == "$wanted" ]] && { echo "$i"; return 0; }
  done
  return 1
}

if [[ "$MODE" != all ]]; then
  START_INDEX=$(stage_index "$START_STAGE") || { echo "unknown stage: $START_STAGE" >&2; exit 2; }
else
  START_INDEX=0
fi

if [[ "$MODE" == from ]]; then
  for ((i=START_INDEX; i<${#stages[@]}; i++)); do
    rm -f "$MARKERS/${stages[$i]}.ok"
  done
  if (( START_INDEX <= 2 )); then
    rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
      "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" \
      "$OUTPUT_DIR/run.log"
  fi
  if (( START_INDEX <= 3 )); then
    rm -f "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  fi
  if (( START_INDEX == 0 )); then
    case "$WORK_DIR" in
      "$TARGET_DIR"/work) rm -rf "$WORK_DIR" ;;
      *) echo "refusing to clear unexpected work directory: $WORK_DIR" >&2; exit 2 ;;
    esac
    mkdir -p "$WORK_DIR" "$MARKERS"
  fi
fi

write_bindings() {
  cat > "$RUN_REPO/language/bert/tokenization.py" <<'PY'
import unicodedata

def printable_text(text):
    return text.decode("utf-8", "ignore") if isinstance(text, bytes) else text

def whitespace_tokenize(text):
    text = text.strip()
    return text.split() if text else []
PY

  cat > "$RUN_REPO/language/bert/evaluate_v1.1.py" <<'PY'
"""SQuAD v1.1 evaluator interface used by the local BERT benchmark."""
import argparse
import collections
import json
import re
import string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth):
    pred = normalize_answer(prediction).split()
    truth = normalize_answer(ground_truth).split()
    common = collections.Counter(pred) & collections.Counter(truth)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(truth)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, truth) for truth in ground_truths)

def evaluate_dataset(dataset, predictions, max_examples=None):
    exact = 0.0
    f1 = 0.0
    total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if max_examples is not None and total >= max_examples:
                    return {"exact_match": 100.0 * exact / total,
                            "f1": 100.0 * f1 / total}
                total += 1
                prediction = predictions.get(qa["id"], "")
                truths = [answer["text"] for answer in qa["answers"]]
                exact += metric_max_over_ground_truths(
                    exact_match_score, prediction, truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, truths)
    return {"exact_match": 100.0 * exact / total, "f1": 100.0 * f1 / total}

def evaluate(raw_start_end_logits, feature_records, question_ids, contexts,
             annotations):
    raise RuntimeError("Use the benchmark span decoder before dataset evaluation")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("prediction_file")
    parser.add_argument("--max_examples", type=int)
    args = parser.parse_args()
    with open(args.dataset_file, encoding="utf-8") as handle:
        dataset = json.load(handle)["data"]
    with open(args.prediction_file, encoding="utf-8") as handle:
        predictions = json.load(handle)
    print(json.dumps(evaluate_dataset(dataset, predictions, args.max_examples)))

if __name__ == "__main__":
    main()
PY

  cat > "$RUN_REPO/language/bert/accuracy-squad.py" <<'PY'
"""Decode BERT span logits and invoke the local SQuAD v1.1 evaluator."""
import argparse
import collections
import json
import os
import pickle
import subprocess
import sys

import numpy as np
from transformers.models.bert.tokenization_bert import BasicTokenizer
from create_squad_data import read_squad_examples

max_seq_length = 384
RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits"])
dtype_map = {name: getattr(np, name) for name in
             ("int8", "int16", "int32", "int64", "float16", "float32", "float64")}

def load_loadgen_log(path, features, dtype=np.float32, output_transposed=False):
    with open(path) as handle:
        records = json.load(handle)
    results = []
    for record in records:
        logits = np.frombuffer(bytes.fromhex(record["data"]), dtype)
        logits = logits.reshape(2, -1).T if output_transposed else logits.reshape(-1, 2)
        start = np.full(max_seq_length, -10000.0)
        end = np.full(max_seq_length, -10000.0)
        start[:len(logits)] = logits[:, 0]
        end[:len(logits)] = logits[:, 1]
        results.append(RawResult(features[record["qsl_idx"]].unique_id,
                                 start.tolist(), end.tolist()))
    return results

def best_indexes(logits, count):
    return sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:count]

def final_text(pred_text, orig_text):
    tokenizer = BasicTokenizer(do_lower_case=True)
    tokenized = " ".join(tokenizer.tokenize(orig_text))
    start = tokenized.find(pred_text)
    if start < 0:
        return orig_text
    end = start + len(pred_text) - 1
    def strip_spaces(text):
        compact = []
        mapping = {}
        for i, char in enumerate(text):
            if char == " ":
                continue
            mapping[len(compact)] = i
            compact.append(char)
        return "".join(compact), mapping
    orig_ns, orig_map = strip_spaces(orig_text)
    tok_ns, tok_map = strip_spaces(tokenized)
    if len(orig_ns) != len(tok_ns):
        return orig_text
    tok_to_ns = {value: key for key, value in tok_map.items()}
    if start not in tok_to_ns or end not in tok_to_ns:
        return orig_text
    ns_start, ns_end = tok_to_ns[start], tok_to_ns[end]
    if ns_start not in orig_map or ns_end not in orig_map:
        return orig_text
    return orig_text[orig_map[ns_start]:orig_map[ns_end] + 1]

def decode(examples, features, results):
    by_example = collections.defaultdict(list)
    for feature in features:
        by_example[feature.example_index].append(feature)
    result_by_id = {result.unique_id: result for result in results}
    predictions = collections.OrderedDict()
    for example_index, example in enumerate(examples):
        candidates = []
        for feature in by_example[example_index]:
            result = result_by_id.get(feature.unique_id)
            if result is None:
                continue
            for start in best_indexes(result.start_logits, 20):
                for end in best_indexes(result.end_logits, 20):
                    if start not in feature.token_to_orig_map:
                        continue
                    if end not in feature.token_to_orig_map or end < start:
                        continue
                    if end - start + 1 > 30:
                        continue
                    if not feature.token_is_max_context.get(start, False):
                        continue
                    candidates.append((result.start_logits[start] + result.end_logits[end],
                                       feature, start, end))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            predictions[example.qas_id] = ""
            continue
        _, feature, start, end = candidates[0]
        tok_text = " ".join(feature.tokens[start:end + 1])
        tok_text = tok_text.replace(" ##", "").replace("##", "").strip()
        orig_start = feature.token_to_orig_map[start]
        orig_end = feature.token_to_orig_map[end]
        orig_text = " ".join(example.doc_tokens[orig_start:orig_end + 1])
        predictions[example.qas_id] = final_text(tok_text, orig_text)
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--features_cache_file", required=True)
    parser.add_argument("--output_transposed", action="store_true")
    parser.add_argument("--output_dtype", default="float32", choices=dtype_map)
    parser.add_argument("--max_examples", type=int)
    args = parser.parse_args()
    examples = read_squad_examples(args.val_data, False, False)
    with open(args.features_cache_file, "rb") as handle:
        features = pickle.load(handle)
    results = load_loadgen_log(args.log_file, features,
                               dtype_map[args.output_dtype], args.output_transposed)
    predictions = decode(examples, features, results)
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as handle:
        json.dump(predictions, handle)
    command = [sys.executable, os.path.join(os.path.dirname(__file__), "evaluate_v1.1.py"),
               args.val_data, args.out_file]
    if args.max_examples is not None:
        command += ["--max_examples", str(args.max_examples)]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
PY

  python3 - "$RUN_REPO/language/bert/pytorch_SUT.py" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
needle = '        self.version = transformers.__version__\n'
addition = needle + '        print("pytorch device: {}".format(self.dev))\n'
if 'pytorch device: ' not in text:
    if needle not in text:
        raise SystemExit("could not locate device initialization")
    path.write_text(text.replace(needle, addition, 1))
PY
}

stage_setup() {
  mkdir -p "$WORK_DIR"
  if [[ ! -d "$RUN_REPO" ]]; then
    cp -a "$SOURCE_REPO" "$RUN_REPO"
  fi
  write_bindings
  if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --python 3.11 "$VENV"
  fi
  uv pip install --python "$VENV/bin/python" \
    'torch==2.5.1' 'transformers==4.46.3' 'numpy==1.26.4' \
    'absl-py==2.1.0' 'six==1.16.0' 'pybind11==2.11.1' \
    'setuptools==75.6.0' 'wheel==0.45.1'
  uv pip install --python "$VENV/bin/python" --no-build-isolation "$RUN_REPO/loadgen"
}

stage_preprocess() {
  cd "$RUN_REPO/language/bert"
  export VOCAB_FILE="$MODEL_DIR/vocab.txt"
  export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
  export PYTHONPATH="$RUN_REPO/language/bert"
  "$VENV/bin/python" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL()
print("preprocessed features:", len(qsl.eval_features))
PY
}

stage_run() {
  cd "$RUN_REPO/language/bert"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "GPU free memory before run: ${free_mib} MiB"
  if (( free_mib < 6000 )); then
    echo "insufficient free GPU memory: ${free_mib} MiB" >&2
    return 1
  fi
  export CUDA_VISIBLE_DEVICES=0
  export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
  export VOCAB_FILE="$MODEL_DIR/vocab.txt"
  export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
  export LOG_PATH="$OUTPUT_DIR"
  export SKIP_VERIFY_ACCURACY=1
  export PYTHONPATH="$RUN_REPO/language/bert"
  args=(--backend=pytorch --accuracy)
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    args+=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  "$VENV/bin/python" run.py "${args[@]}"
}

stage_evaluate() {
  cd "$RUN_REPO/language/bert"
  export PYTHONPATH="$RUN_REPO/language/bert"
  args=(--vocab_file "$MODEL_DIR/vocab.txt"
        --val_data "$SQUAD_DIR/dev-v1.1.json"
        --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json"
        --out_file "$WORK_DIR/predictions.json"
        --features_cache_file "$RUN_REPO/language/bert/eval_features.pickle")
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    args+=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  "$VENV/bin/python" accuracy-squad.py "${args[@]}"
}

run_stage() {
  local stage=$1 log=$2
  if [[ -f "$MARKERS/$stage.ok" ]]; then
    echo "Skipping completed stage: $stage"
    return
  fi
  echo "Starting stage: $stage"
  set +e
  "stage_$stage" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  (( status == 0 )) || return "$status"
  touch "$MARKERS/$stage.ok"
}

if [[ "$MODE" == stage ]]; then
  case "$START_STAGE" in
    setup) run_stage setup "$OUTPUT_DIR/build.log" ;;
    preprocess) run_stage preprocess "$OUTPUT_DIR/preprocess.log" ;;
    run) run_stage run "$OUTPUT_DIR/run.log" ;;
    evaluate)
      set +e
      stage_evaluate 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
      status=${PIPESTATUS[0]}
      set -e
      (( status == 0 )) && touch "$MARKERS/evaluate.ok"
      exit "$status"
      ;;
  esac
  exit 0
fi

for ((i=START_INDEX; i<${#stages[@]}; i++)); do
  stage=${stages[$i]}
  case "$stage" in
    setup) run_stage setup "$OUTPUT_DIR/build.log" ;;
    preprocess) run_stage preprocess "$OUTPUT_DIR/preprocess.log" ;;
    run) run_stage run "$OUTPUT_DIR/run.log" ;;
    evaluate)
      if [[ ! -f "$MARKERS/evaluate.ok" ]]; then
        set +e
        stage_evaluate 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
        status=${PIPESTATUS[0]}
        set -e
        (( status == 0 )) || exit "$status"
        touch "$MARKERS/evaluate.ok"
      fi
      ;;
  esac
done
