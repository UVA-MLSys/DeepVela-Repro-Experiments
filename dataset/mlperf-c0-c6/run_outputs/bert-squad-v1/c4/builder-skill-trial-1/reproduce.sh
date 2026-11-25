#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:?missing target id}"; shift 2 ;;
    --stage) STAGE="${2:?missing stage name}"; shift 2 ;;
    --from) FROM="${2:?missing stage name}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$TARGET" != "bert-squad" ]]; then
  echo "supported target: bert-squad" >&2
  exit 2
fi
if [[ -n "$STAGE" && -n "$FROM" ]]; then
  echo "--stage and --from are mutually exclusive" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$ROOT_DIR/$TARGET"
LOG_DIR="$WORK_DIR/logs"
SOURCE_DIR="$WORK_DIR/source"
BERT_DIR="$SOURCE_DIR/language/bert"
VENV_DIR="$WORK_DIR/venv"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/out}"
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT_DIR/${OUTPUT_DIR#./}"
fi
CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
SQUAD_DIR="${SQUAD_DIR:-$CACHE_ROOT/bert-squad/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/bert-squad/model}"
DEEPLEARNING_EXAMPLES_DIR="${DEEPLEARNING_EXAMPLES_DIR:-$CACHE_ROOT/bert-squad/DeepLearningExamples-c4}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$WORK_DIR" "$LOG_DIR" "$OUTPUT_DIR"

stages=(setup preprocess run evaluate)
stage_index() {
  case "$1" in
    setup) echo 0 ;; preprocess) echo 1 ;; run) echo 2 ;; evaluate) echo 3 ;;
    *) echo "unknown stage: $1" >&2; exit 2 ;;
  esac
}

clear_from() {
  local first
  first="$(stage_index "$1")"
  if (( first == 0 )); then
    rm -rf "$SOURCE_DIR" "$VENV_DIR"
  fi
  local i name
  for i in "${!stages[@]}"; do
    if (( i >= first )); then
      name="${stages[$i]}"
      rm -f "$WORK_DIR/.${name}.success"
    fi
  done
  if (( first <= 2 )); then
    rm -f "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" "$OUTPUT_DIR/accuracy.txt"
  elif (( first <= 3 )); then
    rm -f "$OUTPUT_DIR/accuracy.txt"
  fi
}

write_evaluator_repair() {
  cat > "$BERT_DIR/evaluate_v1.1.py" <<'PY'
"""Benchmark-owned SQuAD v1.1 evaluator restored from BENCHMARK_CARD.md."""
import argparse
import collections
import json
import re
import string


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)
    def white_space_fix(value):
        return " ".join(value.split())
    def remove_punc(value):
        return "".join(ch for ch in value if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, ground_truth) for ground_truth in ground_truths)


def evaluate(dataset, predictions, allowed_ids=None):
    exact_match = 0.0
    f1 = 0.0
    total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if allowed_ids is not None and qa["id"] not in allowed_ids:
                    continue
                total += 1
                prediction = predictions.get(qa["id"], "")
                answers = [answer["text"] for answer in qa["answers"]]
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, answers)
                f1 += metric_max_over_ground_truths(f1_score, prediction, answers)
    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0}
    return {"exact_match": 100.0 * exact_match / total, "f1": 100.0 * f1 / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("prediction_file")
    parser.add_argument("--max_examples", type=int)
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_handle:
        dataset = json.load(dataset_handle)["data"]
    with open(args.prediction_file) as prediction_handle:
        predictions = json.load(prediction_handle)
    allowed = set(list(predictions)[:args.max_examples]) if args.max_examples else None
    print(json.dumps(evaluate(dataset, predictions, allowed)))


if __name__ == "__main__":
    main()
PY

  cat > "$BERT_DIR/accuracy-squad.py" <<'PY'
"""Convert LoadGen BERT logits to text and score with evaluate_v1.1.py."""
import argparse
import collections
import json
import os
import pickle
import sys

import numpy as np
from transformers import BertTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepLearningExamples",
                                "PyTorch", "LanguageModeling", "BERT"))
from create_squad_data import convert_examples_to_features, read_squad_examples
import evaluate_v1_1_compat

max_seq_length = 384
max_query_length = 64
doc_stride = 128
RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
PrelimPrediction = collections.namedtuple(
    "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
dtype_map = {"int8": np.int8, "int16": np.int16, "int32": np.int32,
             "int64": np.int64, "float16": np.float16, "float32": np.float32,
             "float64": np.float64}


def get_final_text(pred_text, orig_text, do_lower_case=True):
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    def strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for index, char in enumerate(text):
            if char == " ":
                continue
            ns_to_s_map[len(ns_chars)] = index
            ns_chars.append(char)
        return "".join(ns_chars), ns_to_s_map

    orig_ns_text, orig_ns_to_s_map = strip_spaces(orig_text)
    tok_ns_text, tok_ns_to_s_map = strip_spaces(tok_text)
    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text
    tok_s_to_ns_map = {s: ns for ns, s in tok_ns_to_s_map.items()}
    if start_position not in tok_s_to_ns_map or end_position not in tok_s_to_ns_map:
        return orig_text
    ns_start_position = tok_s_to_ns_map[start_position]
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_start_position not in orig_ns_to_s_map or ns_end_position not in orig_ns_to_s_map:
        return orig_text
    return orig_text[orig_ns_to_s_map[ns_start_position]:orig_ns_to_s_map[ns_end_position] + 1]


def get_best_indexes(logits, n_best_size):
    indexed = sorted(enumerate(logits), key=lambda item: item[1], reverse=True)
    return [index for index, _ in indexed[:n_best_size]]


def load_loadgen_log(log_path, eval_features, dtype=np.float32, output_transposed=False):
    with open(log_path) as log_handle:
        predictions = json.load(log_handle)
    results = []
    for prediction in predictions:
        qsl_idx = prediction["qsl_idx"]
        if output_transposed:
            logits = np.frombuffer(bytes.fromhex(prediction["data"]), dtype).reshape(2, -1).T
        else:
            logits = np.frombuffer(bytes.fromhex(prediction["data"]), dtype).reshape(-1, 2)
        start_logits = np.full(max_seq_length, -10000.0)
        end_logits = np.full(max_seq_length, -10000.0)
        start_logits[:logits.shape[0]] = logits[:, 0]
        end_logits[:logits.shape[0]] = logits[:, 1]
        results.append(RawResult(eval_features[qsl_idx].unique_id,
                                 start_logits.tolist(), end_logits.tolist()))
    return results


def write_predictions(eval_examples, eval_features, all_results):
    example_to_features = collections.defaultdict(list)
    for feature_index, feature in enumerate(eval_features):
        example_to_features[feature.example_index].append((feature_index, feature))
    unique_id_to_result = {result.unique_id: result for result in all_results}
    all_predictions = collections.OrderedDict()
    predicted_ids = set()
    for example_index, example in enumerate(eval_examples):
        prelim_predictions = []
        for feature_index, feature in example_to_features[example_index]:
            result = unique_id_to_result.get(feature.unique_id)
            if result is None:
                continue
            for start_index in get_best_indexes(result.start_logits, 20):
                for end_index in get_best_indexes(result.end_logits, 20):
                    if start_index >= len(feature.tokens) or end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map or end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > 30:
                        continue
                    prelim_predictions.append(PrelimPrediction(
                        feature_index, start_index, end_index,
                        result.start_logits[start_index], result.end_logits[end_index]))
        prelim_predictions.sort(key=lambda item: item.start_logit + item.end_logit, reverse=True)
        seen = set()
        texts = []
        for prediction in prelim_predictions:
            if len(texts) >= 20:
                break
            feature = eval_features[prediction.feature_index]
            tok_text = " ".join(feature.tokens[prediction.start_index:prediction.end_index + 1])
            tok_text = tok_text.replace(" ##", "").replace("##", "")
            tok_text = " ".join(tok_text.strip().split())
            orig_start = feature.token_to_orig_map[prediction.start_index]
            orig_end = feature.token_to_orig_map[prediction.end_index]
            orig_text = " ".join(example.doc_tokens[orig_start:orig_end + 1])
            final_text = get_final_text(tok_text, orig_text)
            if final_text in seen:
                continue
            seen.add(final_text)
            texts.append(final_text)
        all_predictions[example.qas_id] = texts[0] if texts else ""
        if prelim_predictions:
            predicted_ids.add(example.qas_id)
    return all_predictions, predicted_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", default="build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt")
    parser.add_argument("--val_data", default="build/data/dev-v1.1.json")
    parser.add_argument("--log_file", default="build/logs/mlperf_log_accuracy.json")
    parser.add_argument("--out_file", default="build/result/predictions.json")
    parser.add_argument("--features_cache_file", default="eval_features.pickle")
    parser.add_argument("--output_transposed", action="store_true")
    parser.add_argument("--output_dtype", default="float32", choices=dtype_map.keys())
    parser.add_argument("--max_examples", type=int)
    args = parser.parse_args()
    print("Reading examples...")
    eval_examples = read_squad_examples(args.val_data, is_training=False, version_2_with_negative=False)
    if os.path.exists(args.features_cache_file):
        print("Loading cached features from '%s'..." % args.features_cache_file)
        with open(args.features_cache_file, "rb") as cache_handle:
            eval_features = pickle.load(cache_handle)
    else:
        print("Creating tokenizer and features...")
        tokenizer = BertTokenizer(args.vocab_file)
        eval_features = []
        convert_examples_to_features(eval_examples, tokenizer, max_seq_length, doc_stride,
                                     max_query_length, False, eval_features.append, False)
        with open(args.features_cache_file, "wb") as cache_handle:
            pickle.dump(eval_features, cache_handle)
    print("Loading LoadGen logs...")
    results = load_loadgen_log(args.log_file, eval_features, dtype_map[args.output_dtype],
                               args.output_transposed)
    predictions, predicted_ids = write_predictions(eval_examples, eval_features, results)
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w") as prediction_handle:
        json.dump(predictions, prediction_handle, indent=2)
    with open(args.val_data) as dataset_handle:
        dataset = json.load(dataset_handle)["data"]
    allowed_ids = predicted_ids if args.max_examples else None
    scores = evaluate_v1_1_compat.evaluate(dataset, predictions, allowed_ids)
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
PY

  cp "$BERT_DIR/evaluate_v1.1.py" "$BERT_DIR/evaluate_v1_1_compat.py"
}

run_setup() {
  mkdir -p "$SOURCE_DIR"
  if [[ ! -f "$SOURCE_DIR/loadgen/VERSION.txt" ]]; then
    cp -a "$ROOT_DIR/repo/." "$SOURCE_DIR/"
  fi
  rm -rf "$BERT_DIR/DeepLearningExamples"
  ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$BERT_DIR/DeepLearningExamples"
  write_evaluator_repair
  python3 - "$BERT_DIR/pytorch_SUT.py" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
needle = '        self.version = transformers.__version__\n'
replacement = needle + '        print("pytorch device: {}".format(self.dev))\n'
if 'pytorch device: ' not in text:
    if needle not in text:
        raise SystemExit('device insertion point not found')
    path.write_text(text.replace(needle, replacement, 1))
PY
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    uv venv --python 3.11 --seed "$VENV_DIR"
  fi
  uv pip install --python "$VENV_DIR/bin/python" \
    torch==2.4.1 transformers==4.44.2 numpy==1.26.4 absl-py==2.1.0 \
    six==1.16.0 boto3==1.35.54 pybind11==2.11.1 wheel setuptools
  uv pip install --python "$VENV_DIR/bin/python" --no-build-isolation "$SOURCE_DIR/loadgen"
  "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("loadgen import", mlperf_loadgen.__file__)
PY
  touch "$WORK_DIR/.setup.success"
}

run_preprocess() {
  mkdir -p "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2" "$BERT_DIR/build/result" "$BERT_DIR/build/logs"
  ln -sfn "$SQUAD_DIR/dev-v1.1.json" "$BERT_DIR/build/data/dev-v1.1.json"
  ln -sfn "$MODEL_DIR/model.pytorch" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
  ln -sfn "$MODEL_DIR/vocab.txt" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
  (
    cd "$BERT_DIR"
    PYTHONPATH="$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}" \
      DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" VOCAB_FILE="$MODEL_DIR/vocab.txt" \
      "$VENV_DIR/bin/python" - <<'PY'
from squad_QSL import get_squad_QSL
qsl = get_squad_QSL()
print("preprocessed feature count:", qsl.count)
PY
  )
  touch "$WORK_DIR/.preprocess.success"
}

run_gpu() {
  local free_mib
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 8192 )); then
    echo "insufficient free GPU memory: ${free_mib:-unknown} MiB; require 8192 MiB" >&2
    return 1
  fi
  local args=(--backend=pytorch --accuracy)
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    args+=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  (
    cd "$BERT_DIR"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export LOG_PATH="$BERT_DIR/build/logs"
    export SKIP_VERIFY_ACCURACY=1
    "$VENV_DIR/bin/python" run.py "${args[@]}"
  )
  cp "$BERT_DIR/build/logs/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
  cp "$BERT_DIR/build/logs/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"
  touch "$WORK_DIR/.run.success"
}

run_evaluate() {
  local args=(
    --vocab_file "$MODEL_DIR/vocab.txt"
    --val_data "$SQUAD_DIR/dev-v1.1.json"
    --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json"
    --out_file "$BERT_DIR/build/result/predictions.json"
    --features_cache_file "$BERT_DIR/eval_features.pickle"
  )
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
    args+=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
  fi
  (
    cd "$BERT_DIR"
    "$VENV_DIR/bin/python" accuracy-squad.py "${args[@]}"
  ) | tee "$OUTPUT_DIR/accuracy.txt"
  touch "$WORK_DIR/.evaluate.success"
}

run_one() {
  local name="$1"
  if [[ -f "$WORK_DIR/.${name}.success" ]]; then
    echo "stage $name already complete"
    return
  fi
  case "$name" in
    setup) run_setup 2>&1 | tee "$LOG_DIR/setup.log" ;;
    preprocess) run_preprocess 2>&1 | tee "$LOG_DIR/preprocess.log" ;;
    run) run_gpu 2>&1 | tee "$LOG_DIR/run.log" ;;
    evaluate) run_evaluate 2>&1 | tee "$LOG_DIR/evaluator.log" ;;
  esac
}

if [[ -n "$FROM" ]]; then
  clear_from "$FROM"
fi

if [[ -n "$STAGE" ]]; then
  index="$(stage_index "$STAGE")"
  if (( index > 0 )) && [[ ! -f "$WORK_DIR/.${stages[$((index - 1))]}.success" ]]; then
    echo "stage $STAGE requires completed ${stages[$((index - 1))]} stage" >&2
    exit 1
  fi
  run_one "$STAGE"
elif [[ -n "$FROM" ]]; then
  first="$(stage_index "$FROM")"
  for i in "${!stages[@]}"; do
    (( i < first )) && continue
    run_one "${stages[$i]}"
  done
else
  for name in "${stages[@]}"; do
    run_one "$name"
  done
fi
