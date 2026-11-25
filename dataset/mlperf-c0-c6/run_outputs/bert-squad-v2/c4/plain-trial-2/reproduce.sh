#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --target <target-id>" >&2
  exit 2
}

[[ $# -eq 2 && "$1" == "--target" ]] || usage
TARGET_ID=$2
[[ "$TARGET_ID" == "bert-squad" ]] || { echo "unknown target: $TARGET_ID" >&2; exit 2; }

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR="$ROOT_DIR/repo"
TARGET_DIR="$ROOT_DIR/$TARGET_ID"
OUTPUT_DIR=${OUTPUT_DIR:-"$TARGET_DIR/out"}
WORK_DIR="$TARGET_DIR/work"
VENV_DIR="$TARGET_DIR/venv"

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples-c4"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

mkdir -p "$OUTPUT_DIR"
: > "$OUTPUT_DIR/build.log"

for required in \
  "$REPO_DIR/language/bert/run.py" \
  "$REPO_DIR/loadgen/setup.py" \
  "$SQUAD_DIR/dev-v1.1.json" \
  "$MODEL_DIR/model.pytorch" \
  "$MODEL_DIR/vocab.txt" \
  "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT/tokenization.py"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" | tee -a "$OUTPUT_DIR/build.log"; exit 1; }
done

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cp -a "$REPO_DIR/language/bert" "$WORK_DIR/bert"
cp -a "$REPO_DIR/loadgen" "$WORK_DIR/loadgen"
rm -rf "$WORK_DIR/bert/DeepLearningExamples"
ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$WORK_DIR/bert/DeepLearningExamples"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  PYTHON311=$(command -v python3.11 || true)
  [[ -n "$PYTHON311" ]] || { echo "python3.11 is required" | tee -a "$OUTPUT_DIR/build.log"; exit 1; }
  uv venv --python "$PYTHON311" "$VENV_DIR" 2>&1 | tee -a "$OUTPUT_DIR/build.log"
fi

uv pip install --python "$VENV_DIR/bin/python" \
  'torch==2.6.0' 'transformers==4.46.3' 'numpy<2' 'absl-py' \
  'pybind11' 'setuptools<81' 'six' 'boto3' 2>&1 | tee -a "$OUTPUT_DIR/build.log"
uv pip install --python "$VENV_DIR/bin/python" "$WORK_DIR/loadgen" \
  --no-build-isolation 2>&1 | tee -a "$OUTPUT_DIR/build.log"

# Supply the independently reconstructed logits-to-answer binding in the work copy.
cat > "$WORK_DIR/bert/evaluate_v1.1.py" <<'PY'
import argparse
import json
import re
import string
from collections import Counter


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
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, truth) for truth in ground_truths)


def evaluate(dataset, predictions):
    exact_match = 0.0
    f1 = 0.0
    total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    continue
                ground_truths = [answer["text"] for answer in qa["answers"]]
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    return {"exact_match": 100.0 * exact_match / total,
            "f1": 100.0 * f1 / total}


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
        remaining = args.max_examples
        limited = []
        for article in dataset:
            new_article = dict(article)
            new_paragraphs = []
            for paragraph in article["paragraphs"]:
                if remaining <= 0:
                    break
                new_paragraph = dict(paragraph)
                new_paragraph["qas"] = paragraph["qas"][:remaining]
                remaining -= len(new_paragraph["qas"])
                new_paragraphs.append(new_paragraph)
            new_article["paragraphs"] = new_paragraphs
            limited.append(new_article)
            if remaining <= 0:
                break
        dataset = limited
    print(json.dumps(evaluate(dataset, predictions)))


if __name__ == "__main__":
    main()
PY

cat > "$WORK_DIR/bert/accuracy-squad.py" <<'PY'
import argparse
import collections
import json
import os
import pickle
import subprocess
import sys

import numpy as np
from transformers import BasicTokenizer, BertTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepLearningExamples",
                                "PyTorch", "LanguageModeling", "BERT"))
from create_squad_data import convert_examples_to_features, read_squad_examples

max_seq_length = 384
max_query_length = 64
doc_stride = 128
RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
dtype_map = {"int8": np.int8, "int16": np.int16, "int32": np.int32,
             "int64": np.int64, "float16": np.float16, "float32": np.float32,
             "float64": np.float64}


def load_loadgen_log(log_path, eval_features, dtype=np.float32, output_transposed=False):
    with open(log_path) as source:
        records = json.load(source)
    results = []
    for record in records:
        logits = np.frombuffer(bytes.fromhex(record["data"]), dtype)
        logits = logits.reshape(2, -1).T if output_transposed else logits.reshape(-1, 2)
        start_logits = np.full(max_seq_length, -10000.0)
        end_logits = np.full(max_seq_length, -10000.0)
        start_logits[:len(logits)] = logits[:, 0]
        end_logits[:len(logits)] = logits[:, 1]
        results.append(RawResult(eval_features[record["qsl_idx"]].unique_id,
                                 start_logits.tolist(), end_logits.tolist()))
    return results


def best_indexes(logits, count):
    return sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:count]


def strip_spaces(text):
    compact = []
    index_map = {}
    for index, char in enumerate(text):
        if char == " ":
            continue
        index_map[len(compact)] = index
        compact.append(char)
    return "".join(compact), index_map


def get_final_text(pred_text, orig_text):
    tok_text = " ".join(BasicTokenizer(do_lower_case=True).tokenize(orig_text))
    start = tok_text.find(pred_text)
    if start == -1:
        return orig_text
    end = start + len(pred_text) - 1
    orig_ns, orig_map = strip_spaces(orig_text)
    tok_ns, tok_map = strip_spaces(tok_text)
    if len(orig_ns) != len(tok_ns):
        return orig_text
    tok_s_to_ns = {space_index: ns_index for ns_index, space_index in tok_map.items()}
    if start not in tok_s_to_ns or end not in tok_s_to_ns:
        return orig_text
    orig_start = orig_map[tok_s_to_ns[start]]
    orig_end = orig_map[tok_s_to_ns[end]]
    return orig_text[orig_start:orig_end + 1]


def write_predictions(examples, features, results, output_file):
    by_example = collections.defaultdict(list)
    for feature in features:
        by_example[feature.example_index].append(feature)
    by_id = {result.unique_id: result for result in results}
    predictions = collections.OrderedDict()
    for example_index, example in enumerate(examples):
        preliminary = []
        for feature in by_example[example_index]:
            result = by_id.get(feature.unique_id)
            if result is None:
                continue
            for start in best_indexes(result.start_logits, 20):
                for end in best_indexes(result.end_logits, 20):
                    if start >= len(feature.tokens) or end >= len(feature.tokens):
                        continue
                    if start not in feature.token_to_orig_map or end not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start, False):
                        continue
                    if end < start or end - start + 1 > 30:
                        continue
                    preliminary.append((result.start_logits[start] + result.end_logits[end],
                                        feature, start, end))
        preliminary.sort(key=lambda item: item[0], reverse=True)
        seen = set()
        answers = []
        for _, feature, start, end in preliminary:
            tok_text = " ".join(feature.tokens[start:end + 1])
            tok_text = tok_text.replace(" ##", "").replace("##", "")
            tok_text = " ".join(tok_text.strip().split())
            orig_start = feature.token_to_orig_map[start]
            orig_end = feature.token_to_orig_map[end]
            orig_text = " ".join(example.doc_tokens[orig_start:orig_end + 1])
            final_text = get_final_text(tok_text, orig_text)
            if final_text in seen:
                continue
            seen.add(final_text)
            answers.append(final_text)
            if len(answers) >= 20:
                break
        predictions[example.qas_id] = answers[0] if answers else "empty"
    with open(output_file, "w") as destination:
        json.dump(predictions, destination, indent=2)


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
    examples = read_squad_examples(args.val_data, False, False)
    if args.max_examples:
        examples = examples[:args.max_examples]
    if os.path.exists(args.features_cache_file):
        print("Loading cached features from '%s'..." % args.features_cache_file)
        with open(args.features_cache_file, "rb") as source:
            features = pickle.load(source)
    else:
        print("Converting examples to features...")
        features = []
        tokenizer = BertTokenizer(args.vocab_file)
        convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride,
                                     max_query_length, False, features.append, False)
    print("Loading LoadGen logs...")
    results = load_loadgen_log(args.log_file, features, dtype_map[args.output_dtype],
                               args.output_transposed)
    print("Post-processing predictions...")
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    write_predictions(examples, features, results, args.out_file)
    command = [sys.executable, os.path.join(os.path.dirname(__file__), "evaluate_v1.1.py"),
               args.val_data, args.out_file]
    if args.max_examples:
        command += ["--max_examples", str(args.max_examples)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
PY

# Emit the required device evidence from the same model object used for inference.
"$VENV_DIR/bin/python" - "$WORK_DIR/bert/pytorch_SUT.py" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
needle = "        self.version = transformers.__version__\n"
replacement = needle + "        print(\"pytorch device: {}\".format(self.dev))\n"
if needle not in text:
    raise SystemExit("unable to patch PyTorch device log")
path.write_text(text.replace(needle, replacement, 1))
PY

cd "$WORK_DIR/bert"
mkdir -p build/logs build/result
export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export LOG_PATH="$WORK_DIR/bert/build/logs"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="$VENV_DIR/bin:$PATH"

# Check immediately before the sole GPU run and wait for enough room for fp32 BERT-Large.
: > "$OUTPUT_DIR/gpu-memory.log"
for attempt in $(seq 1 20); do
  gpu_line=$(nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used \
    --format=csv,noheader,nounits | head -n 1)
  printf '%s %s\n' "$(date -Iseconds)" "$gpu_line" | tee -a "$OUTPUT_DIR/gpu-memory.log"
  free_mib=$(printf '%s\n' "$gpu_line" | awk -F, '{gsub(/ /,"",$4); print $4}')
  [[ "$free_mib" -ge 8192 ]] && break
  if [[ "$attempt" -eq 20 ]]; then
    echo "insufficient free GPU memory after 20 checks" | tee -a "$OUTPUT_DIR/run.log"
    exit 1
  fi
  sleep 30
done

export SKIP_VERIFY_ACCURACY=1
"$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy 2>&1 | tee "$OUTPUT_DIR/run.log"

"$VENV_DIR/bin/python" accuracy-squad.py \
  --vocab_file "$MODEL_DIR/vocab.txt" \
  --val_data "$SQUAD_DIR/dev-v1.1.json" \
  --log_file "$LOG_PATH/mlperf_log_accuracy.json" \
  --out_file "$WORK_DIR/bert/build/result/predictions.json" \
  --features_cache_file "$WORK_DIR/bert/eval_features.pickle" \
  2>&1 | tee "$OUTPUT_DIR/evaluator.log"

cp "$LOG_PATH/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_accuracy.json"
cp "$LOG_PATH/mlperf_log_detail.txt" "$OUTPUT_DIR/mlperf_log_detail.txt"
cp "$OUTPUT_DIR/evaluator.log" "$OUTPUT_DIR/accuracy.txt"

grep -q '^pytorch device: ' "$OUTPUT_DIR/run.log"
test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
