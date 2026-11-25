#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
BASE_DIR="$(dirname "$SCRIPT_PATH")"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples-c4"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

TARGET=""
ONLY_STAGE=""
FROM_STAGE=""
MAX_EXAMPLES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --stage) ONLY_STAGE="$2"; shift 2 ;;
        --from) FROM_STAGE="$2"; shift 2 ;;
        --max-examples) MAX_EXAMPLES="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$TARGET" == "bert-squad" ]] || { echo "target must be bert-squad" >&2; exit 2; }
[[ -n "${OUTPUT_DIR:-}" ]] || { echo "OUTPUT_DIR must be set" >&2; exit 2; }

OUTPUT_DIR="$(readlink -m "$OUTPUT_DIR")"
TARGET_DIR="$(dirname "$OUTPUT_DIR")"
STATE_DIR="$TARGET_DIR/.reproduce-work"
WORK_REPO="$STATE_DIR/repo"
VENV="$STATE_DIR/venv"
BERT_DIR="$WORK_REPO/language/bert"
mkdir -p "$TARGET_DIR" "$OUTPUT_DIR"

archive_path() {
    local path="$1"
    if [[ -e "$path" || -L "$path" ]]; then
        mv "$path" "${path}.superseded.$(date +%Y%m%d%H%M%S)"
    fi
}

reset_from() {
    local stage="$1"
    case "$stage" in
        setup)
            archive_path "$STATE_DIR"
            archive_path "$OUTPUT_DIR"
            mkdir -p "$OUTPUT_DIR"
            ;;
        preprocess)
            archive_path "$STATE_DIR/preprocess.done"
            archive_path "$STATE_DIR/run.done"
            archive_path "$BERT_DIR/eval_features.pickle"
            ;;
        run)
            archive_path "$STATE_DIR/run.done"
            mkdir -p "$OUTPUT_DIR/superseded.$(date +%Y%m%d%H%M%S)"
            local archive="$OUTPUT_DIR/superseded.$(date +%Y%m%d%H%M%S)"
            mkdir -p "$archive"
            for name in mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt mlperf_log_trace.json accuracy.txt run.log evaluator.log evaluator.stderr.log gpu-before.log; do
                [[ ! -e "$OUTPUT_DIR/$name" ]] || mv "$OUTPUT_DIR/$name" "$archive/"
            done
            ;;
        *) echo "unknown stage for --from: $stage" >&2; exit 2 ;;
    esac
}

setup_stage() {
    [[ ! -e "$STATE_DIR/setup.done" ]] || return 0
    mkdir -p "$STATE_DIR" "$OUTPUT_DIR"
    {
        echo "setup start: $(date -Is)"
        sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
        [[ "$(sha256sum "$SQUAD_DIR/dev-v1.1.json" | cut -d' ' -f1)" == 95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9 ]]
        [[ "$(sha256sum "$MODEL_DIR/model.pytorch" | cut -d' ' -f1)" == 71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f ]]
        [[ "$(sha256sum "$MODEL_DIR/vocab.txt" | cut -d' ' -f1)" == 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3 ]]
        cp -a "$BASE_DIR/repo" "$WORK_REPO"
        ln -s "$DEEPLEARNING_EXAMPLES_DIR/PyTorch" "$WORK_REPO/language/bert/DeepLearningExamples/PyTorch"
        patch_line="$(awk '/^__BINDING_PATCH_BELOW__$/{print NR + 1; exit}' "$SCRIPT_PATH")"
        tail -n "+$patch_line" "$SCRIPT_PATH" | patch -p1 -d "$WORK_REPO"
        uv venv --python /home/mi3se/.local/bin/python3.11 "$VENV"
        uv pip install --python "$VENV/bin/python" --offline \
            'setuptools==69.5.1' 'wheel==0.44.0' 'pybind11==2.11.1' \
            'absl-py==2.1.0' 'numpy==1.26.4' 'six==1.16.0' \
            'torch==2.1.2' 'transformers==4.31.0' 'boto3==1.35.24'
        uv pip install --python "$VENV/bin/python" --offline --no-build-isolation "$WORK_REPO/loadgen"
        "$VENV/bin/python" -c 'import torch, transformers, mlperf_loadgen; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("transformers", transformers.__version__)'
        echo "setup finish: $(date -Is)"
    } 2>&1 | tee "$OUTPUT_DIR/setup.log"
    touch "$STATE_DIR/setup.done"
}

preprocess_stage() {
    [[ -e "$STATE_DIR/setup.done" ]] || { echo "setup stage has not succeeded" >&2; exit 1; }
    [[ ! -e "$STATE_DIR/preprocess.done" ]] || return 0
    {
        echo "preprocess start: $(date -Is)"
        cd "$BERT_DIR"
        /usr/bin/time -v env \
            PYTHONPATH="$BERT_DIR/DeepLearningExamples/PyTorch/LanguageModeling/BERT" \
            DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
            VOCAB_FILE="$MODEL_DIR/vocab.txt" \
            "$VENV/bin/python" -c 'from squad_QSL import get_squad_QSL; qsl=get_squad_QSL(); print("preprocessed feature count:", qsl.count); assert qsl.count == 10833'
        echo "preprocess finish: $(date -Is)"
    } 2>&1 | tee "$OUTPUT_DIR/preprocess.log"
    touch "$STATE_DIR/preprocess.done"
}

run_stage() {
    [[ -e "$STATE_DIR/preprocess.done" ]] || { echo "preprocess stage has not succeeded" >&2; exit 1; }
    [[ ! -e "$STATE_DIR/run.done" ]] || return 0
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits -i 0 | tee "$OUTPUT_DIR/gpu-before.log"
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')"
    [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "could not read free GPU memory" >&2; exit 1; }
    (( free_mib >= 8192 )) || { echo "GPU 0 has only ${free_mib} MiB free; need at least 8192 MiB" >&2; exit 1; }
    args=(--backend=pytorch --accuracy)
    if [[ -n "$MAX_EXAMPLES" ]]; then args+=(--max_examples="$MAX_EXAMPLES"); fi
    {
        echo "run start: $(date -Is)"
        cd "$BERT_DIR"
        /usr/bin/time -v env \
            LOG_PATH="$OUTPUT_DIR" \
            ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch" \
            DATASET_FILE="$SQUAD_DIR/dev-v1.1.json" \
            VOCAB_FILE="$MODEL_DIR/vocab.txt" \
            SKIP_VERIFY_ACCURACY=1 \
            "$VENV/bin/python" run.py "${args[@]}"
        echo "run finish: $(date -Is)"
    } 2>&1 | tee "$OUTPUT_DIR/run.log"
    eval_args=(
        --vocab_file "$MODEL_DIR/vocab.txt"
        --val_data "$SQUAD_DIR/dev-v1.1.json"
        --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json"
        --out_file "$OUTPUT_DIR/predictions.json"
        --features_cache_file "$BERT_DIR/eval_features.pickle"
    )
    if [[ -n "$MAX_EXAMPLES" ]]; then eval_args+=(--max_examples "$MAX_EXAMPLES"); fi
    cd "$BERT_DIR"
    "$VENV/bin/python" accuracy-squad.py "${eval_args[@]}" \
        2> >(tee "$OUTPUT_DIR/evaluator.stderr.log" >&2) \
        | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
    grep -q 'pytorch device: ' "$OUTPUT_DIR/run.log"
    grep -q '"exact_match"' "$OUTPUT_DIR/accuracy.txt"
    [[ -s "$OUTPUT_DIR/mlperf_log_accuracy.json" ]]
    [[ -s "$OUTPUT_DIR/mlperf_log_detail.txt" ]]
    touch "$STATE_DIR/run.done"
}

if [[ -n "$FROM_STAGE" ]]; then reset_from "$FROM_STAGE"; fi
if [[ -n "$ONLY_STAGE" ]]; then
    case "$ONLY_STAGE" in
        setup) setup_stage ;;
        preprocess) preprocess_stage ;;
        run) run_stage ;;
        *) echo "unknown stage: $ONLY_STAGE" >&2; exit 2 ;;
    esac
else
    setup_stage
    preprocess_stage
    run_stage
fi
exit 0

__BINDING_PATCH_BELOW__
--- a/language/bert/accuracy-squad.py
+++ b/language/bert/accuracy-squad.py
@@ -11,6 +11,7 @@
 import os
 import pickle
 import sys
+import subprocess
 
 import numpy as np
 import pkg_resources
@@ -65,9 +66,9 @@
 
 def write_predictions(raw_start_end_logits, feature_records, question_ids,
                       contexts, annotations, *args, **kwargs):
-    """Generic withheld interface: logits and records to SQuAD metrics."""
-    return _missing_evaluation_binding(raw_start_end_logits, feature_records,
-                                       question_ids, contexts, annotations)
+    from local_eval_binding import make_predictions
+    examples = kwargs.pop("examples")
+    return make_predictions(raw_start_end_logits, feature_records, examples)
 
 
 def load_loadgen_log(log_path, eval_features, dtype=np.float32,
@@ -132,9 +133,20 @@
     print("Loading LoadGen logs...")
     results = load_loadgen_log(args.log_file, eval_features, output_dtype,
                                args.output_transposed)
-    write_predictions(results, eval_features,
-                      [example.qas_id for example in eval_examples],
-                      [example.doc_tokens for example in eval_examples], None)
+    predictions = write_predictions(
+        results, eval_features,
+        [example.qas_id for example in eval_examples],
+        [example.doc_tokens for example in eval_examples], None,
+        examples=eval_examples)
+    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
+    with open(args.out_file, "w") as prediction_stream:
+        json.dump(predictions, prediction_stream, indent=2)
+    command = [sys.executable,
+               os.path.join(os.path.dirname(__file__), "evaluate_v1.1.py"),
+               args.val_data, args.out_file]
+    if args.max_examples:
+        command.extend(["--max_examples", str(args.max_examples)])
+    subprocess.check_call(command)
 
 
 if __name__ == "__main__":
--- a/language/bert/evaluate_v1.1.py
+++ b/language/bert/evaluate_v1.1.py
@@ -1,6 +1,15 @@
 """Interface for the withheld SQuAD logits-to-score evaluation binding."""
 
 import argparse
+import json
+
+from local_eval_binding import (
+    evaluate_files,
+    exact_match_score,
+    f1_score,
+    metric_max_over_ground_truths,
+    normalize_answer,
+)
 
 
 def _missing_evaluation_binding(*args, **kwargs):
@@ -9,22 +18,6 @@
     )
 
 
-def normalize_answer(*args, **kwargs):
-    return _missing_evaluation_binding(*args, **kwargs)
-
-
-def f1_score(*args, **kwargs):
-    return _missing_evaluation_binding(*args, **kwargs)
-
-
-def exact_match_score(*args, **kwargs):
-    return _missing_evaluation_binding(*args, **kwargs)
-
-
-def metric_max_over_ground_truths(*args, **kwargs):
-    return _missing_evaluation_binding(*args, **kwargs)
-
-
 def evaluate(raw_start_end_logits, feature_records, question_ids, contexts,
              annotations):
     """Return ``exact_match`` and ``f1`` for the supplied generic inputs."""
@@ -37,8 +30,11 @@
     parser.add_argument("dataset_file")
     parser.add_argument("prediction_file")
     parser.add_argument("--max_examples", type=int)
-    parser.parse_args()
-    return _missing_evaluation_binding()
+    args = parser.parse_args()
+    scores = evaluate_files(args.dataset_file, args.prediction_file,
+                            args.max_examples)
+    print(json.dumps(scores))
+    return scores
 
 
 if __name__ == "__main__":
--- /dev/null
+++ b/language/bert/local_eval_binding.py
@@ -0,0 +1,160 @@
+"""Local binding from BERT feature logits to SQuAD v1.1 scores.
+
+This module is implemented from the generic interfaces retained in this source
+tree.  It deliberately has no dependency on an external SQuAD postprocessor.
+"""
+
+import collections
+import json
+import re
+import string
+
+from transformers import BasicTokenizer
+
+
+def _best_indexes(logits, n_best_size):
+    return sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:n_best_size]
+
+
+def _strip_spaces(text):
+    non_space_to_original = {}
+    chars = []
+    for index, char in enumerate(text):
+        if char == " ":
+            continue
+        non_space_to_original[len(chars)] = index
+        chars.append(char)
+    return "".join(chars), non_space_to_original
+
+
+def project_text(prediction, original):
+    """Project a WordPiece prediction back onto the original whitespace token span."""
+    tokenizer = BasicTokenizer(do_lower_case=True)
+    tokenized_original = " ".join(tokenizer.tokenize(original))
+    start = tokenized_original.find(prediction)
+    if start < 0:
+        return original
+    end = start + len(prediction) - 1
+    original_no_space, original_map = _strip_spaces(original)
+    tokenized_no_space, tokenized_map = _strip_spaces(tokenized_original)
+    if len(original_no_space) != len(tokenized_no_space):
+        return original
+    tokenized_to_non_space = {v: k for k, v in tokenized_map.items()}
+    if start not in tokenized_to_non_space or end not in tokenized_to_non_space:
+        return original
+    original_start = original_map[tokenized_to_non_space[start]]
+    original_end = original_map[tokenized_to_non_space[end]]
+    return original[original_start:original_end + 1]
+
+
+def make_predictions(results, features, examples, n_best_size=20, max_answer_length=30):
+    result_by_id = {result.unique_id: result for result in results}
+    features_by_example = collections.defaultdict(list)
+    for feature in features:
+        features_by_example[feature.example_index].append(feature)
+
+    predictions = collections.OrderedDict()
+    for example_index, example in enumerate(examples):
+        candidates = []
+        for feature in features_by_example[example_index]:
+            result = result_by_id.get(feature.unique_id)
+            if result is None:
+                continue
+            start_indexes = _best_indexes(result.start_logits, n_best_size)
+            end_indexes = _best_indexes(result.end_logits, n_best_size)
+            for start_index in start_indexes:
+                for end_index in end_indexes:
+                    if start_index not in feature.token_to_orig_map:
+                        continue
+                    if end_index not in feature.token_to_orig_map:
+                        continue
+                    if not feature.token_is_max_context.get(start_index, False):
+                        continue
+                    if end_index < start_index:
+                        continue
+                    if end_index - start_index + 1 > max_answer_length:
+                        continue
+                    candidates.append((
+                        result.start_logits[start_index] + result.end_logits[end_index],
+                        feature,
+                        start_index,
+                        end_index,
+                    ))
+        candidates.sort(key=lambda item: item[0], reverse=True)
+        seen = set()
+        best_text = "empty"
+        for _, feature, start_index, end_index in candidates:
+            token_text = " ".join(feature.tokens[start_index:end_index + 1])
+            token_text = token_text.replace(" ##", "").replace("##", "")
+            token_text = " ".join(token_text.strip().split())
+            original_start = feature.token_to_orig_map[start_index]
+            original_end = feature.token_to_orig_map[end_index]
+            original_text = " ".join(example.doc_tokens[original_start:original_end + 1])
+            final_text = project_text(token_text, original_text)
+            if final_text in seen:
+                continue
+            seen.add(final_text)
+            best_text = final_text
+            break
+        predictions[example.qas_id] = best_text
+    return predictions
+
+
+def normalize_answer(text):
+    text = text.lower()
+    text = "".join(char for char in text if char not in set(string.punctuation))
+    text = re.sub(r"\b(a|an|the)\b", " ", text)
+    return " ".join(text.split())
+
+
+def exact_match_score(prediction, ground_truth):
+    return normalize_answer(prediction) == normalize_answer(ground_truth)
+
+
+def f1_score(prediction, ground_truth):
+    prediction_tokens = normalize_answer(prediction).split()
+    ground_truth_tokens = normalize_answer(ground_truth).split()
+    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
+    same = sum(common.values())
+    if same == 0:
+        return 0.0
+    precision = same / len(prediction_tokens)
+    recall = same / len(ground_truth_tokens)
+    return 2 * precision * recall / (precision + recall)
+
+
+def metric_max_over_ground_truths(metric, prediction, ground_truths):
+    return max(metric(prediction, truth) for truth in ground_truths)
+
+
+def score_dataset(dataset, predictions, max_examples=None):
+    exact_match = 0.0
+    f1 = 0.0
+    total = 0
+    for article in dataset["data"]:
+        for paragraph in article["paragraphs"]:
+            for question in paragraph["qas"]:
+                if max_examples is not None and total >= max_examples:
+                    break
+                total += 1
+                prediction = predictions.get(question["id"], "")
+                truths = [answer["text"] for answer in question["answers"]]
+                exact_match += metric_max_over_ground_truths(
+                    exact_match_score, prediction, truths)
+                f1 += metric_max_over_ground_truths(f1_score, prediction, truths)
+            if max_examples is not None and total >= max_examples:
+                break
+        if max_examples is not None and total >= max_examples:
+            break
+    return {
+        "exact_match": 100.0 * exact_match / total,
+        "f1": 100.0 * f1 / total,
+    }
+
+
+def evaluate_files(dataset_file, prediction_file, max_examples=None):
+    with open(dataset_file) as dataset_stream:
+        dataset = json.load(dataset_stream)
+    with open(prediction_file) as prediction_stream:
+        predictions = json.load(prediction_stream)
+    return score_dataset(dataset, predictions, max_examples)
--- a/language/bert/pytorch_SUT.py
+++ b/language/bert/pytorch_SUT.py
@@ -70,6 +70,7 @@
         self.model = BertForQuestionAnswering(config)
         self.model.to(self.dev)
         self.model.eval()
+        print("pytorch device: {}".format(next(self.model.parameters()).device))
         model_file = os.environ.get(
             "ML_MODEL_FILE_WITH_PATH",
             "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch",
