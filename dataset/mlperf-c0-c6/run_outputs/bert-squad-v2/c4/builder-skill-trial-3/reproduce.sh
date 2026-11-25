#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE=""
FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --from) FROM="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "bert-squad" ]] || { echo "--target bert-squad is required" >&2; exit 2; }
[[ -z "$STAGE" || -z "$FROM" ]] || { echo "use only one of --stage and --from" >&2; exit 2; }

ROOT=$(pwd -P)
REPO="$ROOT/repo"
TARGET_DIR="$ROOT/bert-squad"
WORK="$TARGET_DIR/work"
SRC="$WORK/src"
VENV="$WORK/venv"
OUT="${OUTPUT_DIR:-$TARGET_DIR/out}"
CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
SQUAD_DIR="${SQUAD_DIR:-$CACHE_ROOT/bert-squad/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/bert-squad/model}"
DEEPLEARNING_EXAMPLES_DIR="${DEEPLEARNING_EXAMPLES_DIR:-$CACHE_ROOT/bert-squad/DeepLearningExamples-c4}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_EXCLUDE_NEWER=2026-11-10

case "$OUT" in "$ROOT"/*) ;; *) echo "OUTPUT_DIR must be inside $ROOT" >&2; exit 2 ;; esac
mkdir -p "$TARGET_DIR" "$OUT"

stages=(build prepare run)
start=0
if [[ -n "$FROM" ]]; then
  case "$FROM" in build) start=0 ;; prepare) start=1 ;; run) start=2 ;; *) echo "bad --from stage" >&2; exit 2 ;; esac
  if [[ $start -le 0 ]]; then rm -rf "$WORK"; rm -f "$TARGET_DIR/.build.ok"; fi
  if [[ $start -le 1 ]]; then rm -f "$TARGET_DIR/.prepare.ok" "$WORK/eval_features.pickle"; fi
  if [[ $start -le 2 ]]; then
    rm -f "$TARGET_DIR/.run.ok" "$OUT/mlperf_log_accuracy.json" "$OUT/mlperf_log_detail.txt" "$OUT/accuracy.txt"
  fi
elif [[ -n "$STAGE" ]]; then
  case "$STAGE" in build) start=0 ;; prepare) start=1 ;; run) start=2 ;; *) echo "bad --stage stage" >&2; exit 2 ;; esac
fi

want_stage() {
  local name="$1" idx="$2"
  if [[ -n "$STAGE" ]]; then [[ "$STAGE" == "$name" ]]; else [[ $idx -ge $start ]]; fi
}

stage_build() {
  [[ -f "$TARGET_DIR/.build.ok" ]] && return
  mkdir -p "$WORK"
  {
    echo "building target-local environment"
    test -f "$REPO/language/bert/run.py"
    test -f "$SQUAD_DIR/dev-v1.1.json"
    test -f "$MODEL_DIR/model.pytorch"
    test -f "$MODEL_DIR/vocab.txt"
    test -d "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT"
    cp -a "$REPO" "$SRC"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR/PyTorch" "$SRC/language/bert/DeepLearningExamples/PyTorch"
    uv venv --python /home/mi3se/.local/bin/python3.11 "$VENV"
    uv pip install --python "$VENV/bin/python" --offline \
      'setuptools==84.2.1' wheel pybind11 'numpy==2.4.6' six absl-py \
      'torch==2.15.0' 'transformers==5.21.0' 'boto3==1.44.0' 'requests==2.34.2'
    uv pip install --python "$VENV/bin/python" --offline --no-build-isolation "$SRC/loadgen"
    "$VENV/bin/python" -m pip freeze 2>/dev/null || uv pip freeze --python "$VENV/bin/python"
  } 2>&1 | tee "$TARGET_DIR/build.log"
  touch "$TARGET_DIR/.build.ok"
}

stage_prepare() {
  [[ -f "$TARGET_DIR/.prepare.ok" ]] && return
  test -f "$TARGET_DIR/.build.ok"
  {
    echo "preparing fixed inputs and benchmark-owned evaluator"
    cd "$SRC/language/bert"
    ln -sfn "$DEEPLEARNING_EXAMPLES_DIR/PyTorch" DeepLearningExamples/PyTorch
    ln -sf "$SQUAD_DIR/dev-v1.1.json" dev-v1.1.json
    ln -sf "$MODEL_DIR/model.pytorch" model.pytorch
    ln -sf "$MODEL_DIR/vocab.txt" vocab.txt
    awk '/^: <<'"'"'__BENCHMARK_EVALUATOR__'"'"'$/ { found=1; next } found && /^__BENCHMARK_EVALUATOR__$/ { exit } found { print }' "$ROOT/reproduce.sh" > benchmark_evaluator.py
    "$VENV/bin/python" - <<'PY'
from pathlib import Path
p = Path('pytorch_SUT.py')
s = p.read_text()
needle = '        self.model.to(self.dev)\n'
assert s.count(needle) == 1
device_line = '        print("pytorch device: {}".format(next(self.model.parameters()).device))\n'
if device_line not in s:
    s = s.replace(needle, needle + device_line)
p.write_text(s)
PY
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    "$VENV/bin/python" - <<'PY'
import json
d=json.load(open('dev-v1.1.json'))
ids=[q['id'] for a in d['data'] for p in a['paragraphs'] for q in p['qas']]
assert len(ids)==10570 and len(set(ids))==10570
print('validated SQuAD v1.1 questions:', len(ids))
PY
  } 2>&1 | tee "$TARGET_DIR/prepare.log"
  touch "$TARGET_DIR/.prepare.ok"
}

stage_run() {
  [[ -f "$TARGET_DIR/.run.ok" ]] && return
  test -f "$TARGET_DIR/.prepare.ok"
  local max_args=()
  local eval_args=()
  if [[ -n "${REPRO_MAX_EXAMPLES:-}" ]]; then
    max_args=(--max_examples "$REPRO_MAX_EXAMPLES")
    eval_args=(--max_examples "$REPRO_MAX_EXAMPLES")
  fi
  mkdir -p "$OUT" "$WORK/logs"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader \
    2>&1 | tee "$TARGET_DIR/gpu-before-run.log"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 12000 )) || {
    echo "insufficient free GPU memory: ${free_mib:-unknown} MiB" | tee -a "$TARGET_DIR/gpu-before-run.log"
    exit 1
  }
  (
    cd "$SRC/language/bert"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export LOG_PATH="$WORK/logs"
    export SKIP_VERIFY_ACCURACY=1
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1
    "$VENV/bin/python" run.py --backend=pytorch --accuracy "${max_args[@]}"
  ) 2>&1 | tee "$TARGET_DIR/run.log"
  grep -q '^pytorch device: ' "$TARGET_DIR/run.log"
  (
    cd "$SRC/language/bert"
    "$VENV/bin/python" benchmark_evaluator.py \
      --dataset "$SQUAD_DIR/dev-v1.1.json" \
      --features eval_features.pickle \
      --log "$WORK/logs/mlperf_log_accuracy.json" "${eval_args[@]}"
  ) 2>&1 | tee "$TARGET_DIR/evaluator.log" "$OUT/accuracy.txt"
  cp "$WORK/logs/mlperf_log_accuracy.json" "$OUT/mlperf_log_accuracy.json"
  cp "$WORK/logs/mlperf_log_detail.txt" "$OUT/mlperf_log_detail.txt"
  test -s "$OUT/mlperf_log_accuracy.json"
  test -s "$OUT/mlperf_log_detail.txt"
  grep -q '"exact_match"' "$OUT/accuracy.txt"
  grep -q '"f1"' "$OUT/accuracy.txt"
  touch "$TARGET_DIR/.run.ok"
}

want_stage build 0 && stage_build
want_stage prepare 1 && stage_prepare
want_stage run 2 && stage_run
exit 0

: <<'__BENCHMARK_EVALUATOR__'
import argparse
import collections
import json
import math
import pickle
import re
import string
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
from tokenization import BasicTokenizer
import create_squad_data  # noqa: F401; required to resolve feature pickle classes


def load_examples(path):
    data = json.load(open(path, encoding="utf-8"))["data"]
    out = []
    for article in data:
        for paragraph in article["paragraphs"]:
            doc_tokens = []
            prev_ws = True
            for char in paragraph["context"]:
                ws = char.isspace() or ord(char) == 0x202F
                if ws:
                    prev_ws = True
                else:
                    if prev_ws:
                        doc_tokens.append(char)
                    else:
                        doc_tokens[-1] += char
                    prev_ws = False
            for qa in paragraph["qas"]:
                out.append({"id": qa["id"], "doc_tokens": doc_tokens,
                            "answers": [a["text"] for a in qa["answers"]]})
    return out


def load_results(path, features):
    rows = json.load(open(path, encoding="utf-8"))
    by_uid = {}
    import numpy as np
    for row in rows:
        idx = row["qsl_idx"]
        logits = np.frombuffer(bytes.fromhex(row["data"]), dtype=np.float32).reshape(-1, 2)
        if not np.isfinite(logits).all():
            raise ValueError("non-finite logits")
        by_uid[features[idx].unique_id] = (logits[:, 0].tolist(), logits[:, 1].tolist())
    return by_uid, sorted({features[row["qsl_idx"]].example_index for row in rows})


def best_indexes(values, n):
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:n]


def strip_spaces(text):
    ns = []
    mapping = {}
    for i, char in enumerate(text):
        if char == " ":
            continue
        mapping[len(ns)] = i
        ns.append(char)
    return "".join(ns), mapping


def final_text(pred_text, orig_text):
    tok_text = " ".join(BasicTokenizer(do_lower_case=True).tokenize(orig_text))
    start = tok_text.find(pred_text)
    if start < 0:
        return orig_text
    end = start + len(pred_text) - 1
    orig_ns, orig_map = strip_spaces(orig_text)
    tok_ns, tok_map = strip_spaces(tok_text)
    if len(orig_ns) != len(tok_ns):
        return orig_text
    tok_to_ns = {v: k for k, v in tok_map.items()}
    if start not in tok_to_ns or end not in tok_to_ns:
        return orig_text
    ns_start, ns_end = tok_to_ns[start], tok_to_ns[end]
    if ns_start not in orig_map or ns_end not in orig_map:
        return orig_text
    return orig_text[orig_map[ns_start]:orig_map[ns_end] + 1]


def make_predictions(examples, features, results, example_indexes):
    by_example = collections.defaultdict(list)
    for feature in features:
        if feature.unique_id in results:
            by_example[feature.example_index].append(feature)
    predictions = {}
    for example_index in example_indexes:
        prelim = []
        for feature in by_example[example_index]:
            starts, ends = results[feature.unique_id]
            for start in best_indexes(starts, 20):
                for end in best_indexes(ends, 20):
                    if start not in feature.token_to_orig_map or end not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start, False):
                        continue
                    if end < start or end - start + 1 > 30:
                        continue
                    prelim.append((starts[start] + ends[end], feature, start, end))
        prelim.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        texts = []
        for _, feature, start, end in prelim:
            token_text = " ".join(feature.tokens[start:end + 1])
            token_text = token_text.replace(" ##", "").replace("##", "").strip()
            orig_start = feature.token_to_orig_map[start]
            orig_end = feature.token_to_orig_map[end]
            orig_text = " ".join(examples[example_index]["doc_tokens"][orig_start:orig_end + 1])
            text = final_text(token_text, orig_text)
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)
            if len(texts) >= 20:
                break
        predictions[examples[example_index]["id"]] = texts[0] if texts else ""
    return predictions


def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact(prediction, truth):
    return int(normalize(prediction) == normalize(truth))


def f1(prediction, truth):
    p, g = normalize(prediction).split(), normalize(truth).split()
    if not p or not g:
        return int(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--max_examples", type=int)
    args = parser.parse_args()
    examples = load_examples(args.dataset)
    with open(args.features, "rb") as handle:
        features = pickle.load(handle)
    results, indexes = load_results(args.log, features)
    if args.max_examples is None and len(indexes) != len(examples):
        raise ValueError(f"full run covers {len(indexes)} of {len(examples)} questions")
    predictions = make_predictions(examples, features, results, indexes)
    em = total_f1 = 0.0
    for index in indexes:
        ex = examples[index]
        pred = predictions[ex["id"]]
        em += max(exact(pred, truth) for truth in ex["answers"])
        total_f1 += max(f1(pred, truth) for truth in ex["answers"])
    scores = {"exact_match": 100.0 * em / len(indexes), "f1": 100.0 * total_f1 / len(indexes)}
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
__BENCHMARK_EVALUATOR__

