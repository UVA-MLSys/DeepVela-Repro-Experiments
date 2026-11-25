#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target <target-id>" >&2
  exit 2
}

target=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; target="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$target" == "bert-squad" ]] || usage

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples-c4"
UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_CACHE_DIR UV_EXCLUDE_NEWER=2026-11-10

TARGET_DIR="$ROOT_DIR/$target"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  mkdir -p "$OUTPUT_DIR"
  OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
else
  OUTPUT_DIR="$TARGET_DIR/out"
  mkdir -p "$OUTPUT_DIR"
fi
mkdir -p "$TARGET_DIR"
WORK_DIR="$TARGET_DIR/work"
mkdir -p "$WORK_DIR"
if [[ -L "$WORK_DIR/language/bert/DeepLearningExamples" ]]; then
  unlink "$WORK_DIR/language/bert/DeepLearningExamples"
fi
cp -a "$ROOT_DIR/repo/." "$WORK_DIR/"
if [[ -d "$WORK_DIR/language/bert/DeepLearningExamples" && ! -L "$WORK_DIR/language/bert/DeepLearningExamples" ]]; then
  if [[ -L "$WORK_DIR/language/bert/DeepLearningExamples/DeepLearningExamples-c4" ]]; then
    unlink "$WORK_DIR/language/bert/DeepLearningExamples/DeepLearningExamples-c4"
  fi
  rmdir "$WORK_DIR/language/bert/DeepLearningExamples"
fi
ln -sfn "$DEEPLEARNING_EXAMPLES_DIR" "$WORK_DIR/language/bert/DeepLearningExamples"

cat > "$WORK_DIR/language/bert/evaluate_v1.1.py" <<'PY'
import argparse
import collections
import json
import re
import string

def normalize_answer(text):
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)
    def white_space_fix(s):
        return " ".join(s.split())
    def remove_punc(s):
        return "".join(ch for ch in s if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(text.lower())))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, truth) for truth in ground_truths)

def evaluate(dataset, predictions):
    exact_match = 0
    f1 = 0
    total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    continue
                ground_truths = [answer["text"] for answer in qa["answers"]]
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    return {"exact_match": 100.0 * exact_match / total, "f1": 100.0 * f1 / total}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("prediction_file")
    parser.add_argument("--max_examples", type=int)
    args = parser.parse_args()
    with open(args.dataset_file) as source:
        dataset = json.load(source)["data"]
    with open(args.prediction_file) as source:
        predictions = json.load(source)
    if args.max_examples is not None:
        kept = []
        remaining = args.max_examples
        for article in dataset:
            article = dict(article)
            article["paragraphs"] = []
            for paragraph in article.get("paragraphs", []):
                if remaining <= 0:
                    break
                paragraph = dict(paragraph)
                paragraph["qas"] = paragraph["qas"][:remaining]
                remaining -= len(paragraph["qas"])
                article["paragraphs"].append(paragraph)
            kept.append(article)
            if remaining <= 0:
                break
        dataset = kept
    print(json.dumps(evaluate(dataset, predictions)))

if __name__ == "__main__":
    main()
PY

cat > "$WORK_DIR/language/bert/accuracy-squad.py" <<'PY'
import argparse
import collections
import json
import os
import pickle
import subprocess
import sys

import numpy as np
from transformers import BasicTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.path.dirname(__file__))
from create_squad_data import read_squad_examples

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

def best_indexes(logits, n_best_size):
    return sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:n_best_size]

def strip_spaces(text):
    ns_chars = []
    ns_to_s_map = {}
    for index, char in enumerate(text):
        if char == " ":
            continue
        ns_to_s_map[len(ns_chars)] = index
        ns_chars.append(char)
    return "".join(ns_chars), ns_to_s_map

def final_text(pred_text, orig_text):
    tok_text = " ".join(BasicTokenizer(do_lower_case=True).tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1
    orig_ns_text, orig_ns_to_s_map = strip_spaces(orig_text)
    tok_ns_text, tok_ns_to_s_map = strip_spaces(tok_text)
    if len(orig_ns_text) != len(tok_ns_text):
        return orig_text
    tok_s_to_ns_map = {space_index: ns_index for ns_index, space_index in tok_ns_to_s_map.items()}
    if start_position not in tok_s_to_ns_map or end_position not in tok_s_to_ns_map:
        return orig_text
    orig_start = orig_ns_to_s_map[tok_s_to_ns_map[start_position]]
    orig_end = orig_ns_to_s_map[tok_s_to_ns_map[end_position]]
    return orig_text[orig_start:orig_end + 1]

def load_results(path, features):
    with open(path) as source:
        records = json.load(source)
    results = {}
    for record in records:
        feature = features[record["qsl_idx"]]
        logits = np.frombuffer(bytes.fromhex(record["data"]), dtype=np.float32).reshape(-1, 2)
        results[feature.unique_id] = RawResult(feature.unique_id, logits[:, 0], logits[:, 1])
    return results

def make_predictions(examples, features, results, n_best_size=20, max_answer_length=30):
    features_by_example = collections.defaultdict(list)
    for feature in features:
        features_by_example[feature.example_index].append(feature)
    predictions = collections.OrderedDict()
    for example_index, example in enumerate(examples):
        prelim = []
        for feature in features_by_example[example_index]:
            result = results.get(feature.unique_id)
            if result is None:
                continue
            for start_index in best_indexes(result.start_logits, n_best_size):
                for end_index in best_indexes(result.end_logits, n_best_size):
                    if start_index >= len(feature.tokens) or end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map or end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    prelim.append((float(result.start_logits[start_index] + result.end_logits[end_index]), feature, start_index, end_index))
        prelim.sort(key=lambda item: item[0], reverse=True)
        seen = set()
        best_answer = "empty"
        for _, feature, start_index, end_index in prelim:
            token_text = " ".join(feature.tokens[start_index:end_index + 1])
            token_text = token_text.replace(" ##", "").replace("##", "")
            token_text = " ".join(token_text.strip().split())
            orig_start = feature.token_to_orig_map[start_index]
            orig_end = feature.token_to_orig_map[end_index]
            orig_text = " ".join(example.doc_tokens[orig_start:orig_end + 1])
            answer = final_text(token_text, orig_text)
            if answer in seen:
                continue
            seen.add(answer)
            best_answer = answer
            break
        predictions[example.qas_id] = best_answer
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", default="build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt")
    parser.add_argument("--val_data", default="build/data/dev-v1.1.json")
    parser.add_argument("--log_file", default="build/logs/mlperf_log_accuracy.json")
    parser.add_argument("--out_file", default="build/result/predictions.json")
    parser.add_argument("--features_cache_file", default="eval_features.pickle")
    parser.add_argument("--max_examples", type=int)
    args = parser.parse_args()
    print("Reading examples...")
    examples = read_squad_examples(args.val_data, is_training=False, version_2_with_negative=False)
    print("Loading cached features from '%s'..." % args.features_cache_file)
    with open(args.features_cache_file, "rb") as source:
        features = pickle.load(source)
    print("Loading LoadGen logs...")
    results = load_results(args.log_file, features)
    predictions = make_predictions(examples, features, results)
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w") as destination:
        json.dump(predictions, destination, indent=2)
    command = [sys.executable, os.path.join(os.path.dirname(__file__), "evaluate_v1.1.py"), args.val_data, args.out_file]
    if args.max_examples is not None:
        command += ["--max_examples", str(args.max_examples)]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
PY

python3 - "$WORK_DIR/language/bert/pytorch_SUT.py" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
needle = "        self.model.to(self.dev)\n        self.model.eval()"
replacement = "        self.model.to(self.dev)\n        print(\"pytorch device: {}\".format(next(self.model.parameters()).device))\n        self.model.eval()"
if needle not in text:
    raise SystemExit("unable to patch device diagnostic")
path.write_text(text.replace(needle, replacement))
PY

VENV="$TARGET_DIR/.venv"
{
  uv venv --clear --python /home/mi3se/.local/bin/python3.11 "$VENV"
  uv pip install --python "$VENV/bin/python" \
    'torch==2.4.1' 'transformers==4.45.2' 'numpy==1.26.4' \
    'absl-py' 'six' 'boto3' 'setuptools' 'wheel' 'pybind11==2.11.1'
  uv pip install --python "$VENV/bin/python" "$WORK_DIR/loadgen"
} > "$TARGET_DIR/build.log" 2>&1

export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export LOG_PATH="$OUTPUT_DIR"
export SKIP_VERIFY_ACCURACY=1
export PYTHONUNBUFFERED=1

cd "$WORK_DIR/language/bert"
mkdir -p build/data/bert_tf_v1_1_large_fp32_384_v2 build/result
ln -sfn "$SQUAD_DIR/dev-v1.1.json" build/data/dev-v1.1.json
ln -sfn "$MODEL_DIR/vocab.txt" build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt

{
  date -Is
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
} > "$TARGET_DIR/gpu-memory.log"

"$VENV/bin/python" run.py --backend=pytorch --accuracy 2>&1 | tee "$TARGET_DIR/run.log"

"$VENV/bin/python" accuracy-squad.py \
  --vocab_file "$MODEL_DIR/vocab.txt" \
  --val_data "$SQUAD_DIR/dev-v1.1.json" \
  --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
  --out_file build/result/predictions.json \
  --features_cache_file eval_features.pickle \
  2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$TARGET_DIR/evaluator.log"

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
