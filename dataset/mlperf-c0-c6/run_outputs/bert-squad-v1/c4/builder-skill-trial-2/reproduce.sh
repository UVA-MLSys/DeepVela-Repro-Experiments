#!/usr/bin/env bash
set -euo pipefail

TARGET=""
STAGE="all"
FROM=""
MAX_EXAMPLES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --from) FROM="${2:-}"; shift 2 ;;
    --max-examples) MAX_EXAMPLES="${2:-}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "bert-squad" ]] || { echo "only --target bert-squad is supported" >&2; exit 2; }
[[ "$STAGE" =~ ^(all|build|run|evaluate)$ ]] || { echo "invalid --stage" >&2; exit 2; }
[[ -z "$FROM" || "$FROM" =~ ^(build|run|evaluate)$ ]] || { echo "invalid --from" >&2; exit 2; }
[[ -z "$MAX_EXAMPLES" || "$MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --max-examples" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$SCRIPT_DIR/bert-squad"
SRC_DIR="$WORK_ROOT/src"
VENV_DIR="$WORK_ROOT/venv"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/out}"
CACHE_ROOT="${CACHE_ROOT:-/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache}"
SQUAD_DIR="${SQUAD_DIR:-$CACHE_ROOT/bert-squad/data}"
MODEL_DIR="${MODEL_DIR:-$CACHE_ROOT/bert-squad/model}"
DEEPLEARNING_EXAMPLES_DIR="${DEEPLEARNING_EXAMPLES_DIR:-$CACHE_ROOT/bert-squad/DeepLearningExamples-c4}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_EXCLUDE_NEWER="2026-11-10"
export TORCH_HOME="$UV_CACHE_DIR/torch"
export HF_HOME="$UV_CACHE_DIR/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

case "$WORK_ROOT" in "$SCRIPT_DIR/bert-squad") ;; *) echo "unsafe work root" >&2; exit 2 ;; esac
mkdir -p "$WORK_ROOT" "$OUTPUT_DIR"

want_stage() {
  local candidate="$1"
  local order_build=1 order_run=2 order_evaluate=3 n start
  case "$candidate" in build) n=1 ;; run) n=2 ;; evaluate) n=3 ;; esac
  if [[ -n "$FROM" ]]; then
    case "$FROM" in build) start=1 ;; run) start=2 ;; evaluate) start=3 ;; esac
    (( n >= start ))
  elif [[ "$STAGE" == "all" ]]; then
    return 0
  else
    [[ "$STAGE" == "$candidate" ]]
  fi
}

verify_inputs() {
  [[ -f "$SCRIPT_DIR/repo/language/bert/run.py" ]] || { echo "fresh repo/ not found beside reproduce.sh" >&2; exit 1; }
  [[ -f "$SQUAD_DIR/dev-v1.1.json" && -f "$MODEL_DIR/model.pytorch" && -f "$MODEL_DIR/vocab.txt" ]] || { echo "required input missing" >&2; exit 1; }
  [[ -d "$DEEPLEARNING_EXAMPLES_DIR" ]] || { echo "DeepLearningExamples input missing" >&2; exit 1; }
  printf '%s  %s\n' \
    95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9 "$SQUAD_DIR/dev-v1.1.json" \
    71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f "$MODEL_DIR/model.pytorch" \
    07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3 "$MODEL_DIR/vocab.txt" | sha256sum -c -
}

write_evaluator() {
  local evaluator="$SRC_DIR/language/bert/accuracy-squad.py"
  cat > "$SRC_DIR/language/bert/tokenization.py" <<'PY'
"""Compatibility helpers required by create_squad_data in this snapshot."""
import six


def printable_text(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        if isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
    else:
        if isinstance(text, str):
            return text
        if isinstance(text, unicode):  # noqa: F821
            return text.encode("utf-8")
    raise ValueError("unsupported string type: %s" % type(text))


def whitespace_tokenize(text):
    text = text.strip()
    return text.split() if text else []
PY
  cat > "$evaluator" <<'PY'
#!/usr/bin/env python3
"""Independent benchmark-owned binding from LoadGen BERT logits to SQuAD scores."""
import argparse
import collections
import json
import os
import pickle
import re
import string

import numpy as np
from transformers import BasicTokenizer, BertTokenizer
from create_squad_data import convert_examples_to_features, read_squad_examples

MAX_SEQ_LENGTH = 384
MAX_QUERY_LENGTH = 64
DOC_STRIDE = 128


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction, truths):
    pred = normalize_answer(prediction)
    pred_tokens = pred.split()
    best_em = 0.0
    best_f1 = 0.0
    for truth in truths:
        gold = normalize_answer(truth)
        gold_tokens = gold.split()
        em = float(pred == gold)
        if not pred_tokens and not gold_tokens:
            f1 = 1.0
        elif not pred_tokens or not gold_tokens:
            f1 = 0.0
        else:
            overlap = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
            common = sum(overlap.values())
            if common == 0:
                f1 = 0.0
            else:
                precision = common / len(pred_tokens)
                recall = common / len(gold_tokens)
                f1 = 2.0 * precision * recall / (precision + recall)
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)
    return best_em, best_f1


def final_text(pred_text, orig_text):
    tok_text = " ".join(BasicTokenizer(do_lower_case=True).tokenize(orig_text))
    start = tok_text.find(pred_text)
    if start < 0:
        return orig_text
    end = start + len(pred_text) - 1

    def strip_spaces(text):
        ns_chars = []
        ns_to_s = {}
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            ns_to_s[len(ns_chars)] = i
            ns_chars.append(ch)
        return "".join(ns_chars), ns_to_s

    orig_ns, orig_map = strip_spaces(orig_text)
    tok_ns, tok_map = strip_spaces(tok_text)
    if len(orig_ns) != len(tok_ns):
        return orig_text
    tok_s_to_ns = {s: ns for ns, s in tok_map.items()}
    if start not in tok_s_to_ns or end not in tok_s_to_ns:
        return orig_text
    ns_start, ns_end = tok_s_to_ns[start], tok_s_to_ns[end]
    if ns_start not in orig_map or ns_end not in orig_map:
        return orig_text
    return orig_text[orig_map[ns_start] : orig_map[ns_end] + 1]


def load_logits(path, features):
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    by_feature = {}
    represented_examples = set()
    for record in records:
        qsl_idx = int(record["qsl_idx"])
        if qsl_idx >= len(features):
            raise ValueError(f"qsl_idx {qsl_idx} exceeds {len(features)} features")
        values = np.frombuffer(bytes.fromhex(record["data"]), dtype=np.float32)
        if values.size % 2:
            raise ValueError("odd logits value count")
        logits = values.reshape(-1, 2)
        feature = features[qsl_idx]
        by_feature[feature.unique_id] = (logits[:, 0], logits[:, 1])
        represented_examples.add(feature.example_index)
    return by_feature, represented_examples


def predict(examples, features, logits_by_feature, represented_examples):
    grouped = collections.defaultdict(list)
    for ordinal, feature in enumerate(features):
        if feature.unique_id in logits_by_feature:
            grouped[feature.example_index].append((ordinal, feature))
    predictions = {}
    for example_index in sorted(represented_examples):
        candidates = []
        for ordinal, feature in grouped[example_index]:
            start_logits, end_logits = logits_by_feature[feature.unique_id]
            starts = sorted(range(len(start_logits)), key=lambda i: (-float(start_logits[i]), i))[:20]
            ends = sorted(range(len(end_logits)), key=lambda i: (-float(end_logits[i]), i))[:20]
            for start in starts:
                for end in ends:
                    if start not in feature.token_to_orig_map or end not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start, False):
                        continue
                    if end < start or end - start + 1 > 30:
                        continue
                    candidates.append((
                        -(float(start_logits[start]) + float(end_logits[end])),
                        ordinal, start, end, feature,
                    ))
        candidates.sort(key=lambda row: row[:4])
        seen = set()
        texts = []
        example = examples[example_index]
        for _, _, start, end, feature in candidates:
            token_text = " ".join(feature.tokens[start : end + 1]).replace(" ##", "").replace("##", "")
            orig_start = feature.token_to_orig_map[start]
            orig_end = feature.token_to_orig_map[end]
            orig_text = " ".join(example.doc_tokens[orig_start : orig_end + 1])
            text = final_text(token_text, orig_text).strip()
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)
            if len(texts) == 20:
                break
        predictions[example.qas_id] = texts[0] if texts else ""
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--features_cache_file", required=True)
    parser.add_argument("--max_examples", type=int)
    args = parser.parse_args()

    examples = read_squad_examples(args.val_data, False, False)
    if os.path.exists(args.features_cache_file):
        with open(args.features_cache_file, "rb") as handle:
            features = pickle.load(handle)
    else:
        tokenizer = BertTokenizer(args.vocab_file)
        features = []
        convert_examples_to_features(
            examples, tokenizer, MAX_SEQ_LENGTH, DOC_STRIDE, MAX_QUERY_LENGTH,
            False, features.append, False,
        )
        with open(args.features_cache_file, "wb") as handle:
            pickle.dump(features, handle)

    logits, represented = load_logits(args.log_file, features)
    predictions = predict(examples, features, logits, represented)
    with open(args.val_data, encoding="utf-8") as handle:
        dataset = json.load(handle)["data"]
    truths = {
        qa["id"]: [answer["text"] for answer in qa["answers"]]
        for article in dataset for paragraph in article["paragraphs"] for qa in paragraph["qas"]
    }
    em_total = 0.0
    f1_total = 0.0
    for qid, prediction in predictions.items():
        em, f1 = answer_scores(prediction, truths[qid])
        em_total += em
        f1_total += f1
    count = len(predictions)
    if count == 0:
        raise RuntimeError("accuracy log contains no scored examples")
    result = {"exact_match": 100.0 * em_total / count, "f1": 100.0 * f1_total / count}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
PY
}

build_stage() {
  verify_inputs
  rm -f -- "$WORK_ROOT/.build.success" "$WORK_ROOT/.run.success" "$WORK_ROOT/.evaluate.success"
  if [[ -e "$SRC_DIR" || -e "$VENV_DIR" ]]; then
    rm -rf -- "$SRC_DIR" "$VENV_DIR"
  fi
  cp -a "$SCRIPT_DIR/repo" "$SRC_DIR"
  ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$SRC_DIR/language/bert/DeepLearningExamples"
  mkdir -p "$SRC_DIR/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2" "$SRC_DIR/language/bert/build/logs"
  ln -s "$SQUAD_DIR/dev-v1.1.json" "$SRC_DIR/language/bert/build/data/dev-v1.1.json"
  ln -s "$MODEL_DIR/model.pytorch" "$SRC_DIR/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
  ln -s "$MODEL_DIR/vocab.txt" "$SRC_DIR/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
  sed -i '/self.model.load_state_dict(torch.load(model_file), strict=False)/a\        print("pytorch device: {}".format(next(self.model.parameters()).device))' "$SRC_DIR/language/bert/pytorch_SUT.py"
  write_evaluator
  uv venv --python /home/mi3se/.local/bin/python3.11 "$VENV_DIR"
  uv pip install --python "$VENV_DIR/bin/python" \
    'torch==2.5.1' 'transformers==4.46.3' 'numpy==1.26.4' \
    'absl-py==2.5.1' 'pybind11==3.0.1' 'setuptools==80.9.0' 'six==1.17.0'
  uv pip install --python "$VENV_DIR/bin/python" "$SRC_DIR/loadgen"
  "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1 || true
  uv pip freeze --python "$VENV_DIR/bin/python"
  "$VENV_DIR/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "numpy", numpy.__version__)
PY
  touch "$WORK_ROOT/.build.success"
}

run_stage() {
  [[ -f "$WORK_ROOT/.build.success" ]] || { echo "build stage has not succeeded" >&2; exit 1; }
  verify_inputs
  rm -f -- "$OUTPUT_DIR/mlperf_log_accuracy.json" "$OUTPUT_DIR/mlperf_log_detail.txt" \
    "$OUTPUT_DIR/mlperf_log_summary.txt" "$OUTPUT_DIR/mlperf_log_trace.json" \
    "$OUTPUT_DIR/accuracy.txt" "$WORK_ROOT/.run.success" "$WORK_ROOT/.evaluate.success"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  [[ "$free_mib" =~ ^[0-9]+$ && "$free_mib" -ge 12000 ]] || { echo "insufficient free GPU memory: ${free_mib:-unknown} MiB" >&2; exit 1; }
  local extra=()
  [[ -z "$MAX_EXAMPLES" ]] || extra=(--max_examples "$MAX_EXAMPLES")
  (
    cd "$SRC_DIR/language/bert"
    export LOG_PATH="$OUTPUT_DIR"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export SKIP_VERIFY_ACCURACY=1
    "$VENV_DIR/bin/python" run.py --backend=pytorch --accuracy "${extra[@]}"
  )
  grep -q '^pytorch device: cuda' "$WORK_ROOT/run.log"
  [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" && -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]] || { echo "LoadGen outputs missing" >&2; exit 1; }
  touch "$WORK_ROOT/.run.success"
}

evaluate_stage() {
  [[ -f "$WORK_ROOT/.run.success" ]] || { echo "run stage has not succeeded" >&2; exit 1; }
  rm -f -- "$WORK_ROOT/.evaluate.success"
  local extra=()
  [[ -z "$MAX_EXAMPLES" ]] || extra=(--max_examples "$MAX_EXAMPLES")
  (
    cd "$SRC_DIR/language/bert"
    "$VENV_DIR/bin/python" accuracy-squad.py \
      --vocab_file "$MODEL_DIR/vocab.txt" \
      --val_data "$SQUAD_DIR/dev-v1.1.json" \
      --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
      --features_cache_file "$SRC_DIR/language/bert/eval_features.pickle" \
      "${extra[@]}"
  ) | tee "$OUTPUT_DIR/accuracy.txt"
  "$VENV_DIR/bin/python" - "$OUTPUT_DIR/accuracy.txt" <<'PY'
import json, sys
lines = [line.strip() for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
result = json.loads(lines[-1])
assert set(result) == {"exact_match", "f1"}
assert all(isinstance(result[key], (int, float)) for key in result)
PY
  touch "$WORK_ROOT/.evaluate.success"
}

if want_stage build; then
  build_stage 2>&1 | tee "$WORK_ROOT/build.log"
fi
if want_stage run; then
  run_stage 2>&1 | tee "$WORK_ROOT/run.log"
fi
if want_stage evaluate; then
  evaluate_stage 2>&1 | tee "$WORK_ROOT/evaluator.log"
fi
