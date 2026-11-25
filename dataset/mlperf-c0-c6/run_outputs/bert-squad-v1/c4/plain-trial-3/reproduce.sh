#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != "--target" || ${2:-} != "bert-squad" || $# -ne 2 ]]; then
  echo "usage: $0 --target bert-squad" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=${REPO_ROOT:-"$SCRIPT_DIR/repo"}
if [[ ! -d "$REPO_ROOT/language/bert" && -d "$SCRIPT_DIR/language/bert" ]]; then
  REPO_ROOT="$SCRIPT_DIR"
fi
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/bert-squad/out"}
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples-c4"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10

WORK_DIR=$(dirname -- "$OUTPUT_DIR")
VENV="$WORK_DIR/.venv"
PYTHON_DIR="$WORK_DIR/python"
BERT_DIR="$REPO_ROOT/language/bert"
mkdir -p "$OUTPUT_DIR"

if [[ ! -x "$VENV/bin/python" ]]; then
  uv python install 3.11 --install-dir "$PYTHON_DIR" --no-bin >"$WORK_DIR/build.log" 2>&1
  MANAGED_PYTHON=$(find "$PYTHON_DIR" -path '*/bin/python3.11' -type f | head -n 1)
  test -n "$MANAGED_PYTHON"
  uv venv --python "$MANAGED_PYTHON" "$VENV" >>"$WORK_DIR/build.log" 2>&1
  uv pip install --python "$VENV/bin/python" \
    'torch==2.5.1' 'transformers==4.45.2' 'numpy==1.26.4' \
    'absl-py==1.4.0' 'pybind11==2.11.1' 'setuptools<81' 'six==1.17.0' \
    'boto3==1.41.5' \
    >>"$WORK_DIR/build.log" 2>&1
else
  : >"$WORK_DIR/build.log"
fi

if ! "$VENV/bin/python" -c 'import mlperf_loadgen' >/dev/null 2>&1; then
  uv pip install --python "$VENV/bin/python" --no-build-isolation "$REPO_ROOT/loadgen" >>"$WORK_DIR/build.log" 2>&1
fi

mkdir -p "$BERT_DIR/DeepLearningExamples"
ln -sfn "$DEEPLEARNING_EXAMPLES_DIR/PyTorch" "$BERT_DIR/DeepLearningExamples/PyTorch"
mkdir -p "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2" "$BERT_DIR/build/result"
ln -sfn "$SQUAD_DIR/dev-v1.1.json" "$BERT_DIR/build/data/dev-v1.1.json"
ln -sfn "$MODEL_DIR/model.pytorch" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
ln -sfn "$MODEL_DIR/vocab.txt" "$BERT_DIR/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"

# Install the reconstructed local post-processing binding into this fresh tree.
"$VENV/bin/python" - "$BERT_DIR" <<'PY_PATCH'
from pathlib import Path
import sys

bert = Path(sys.argv[1])

accuracy = r'''# Reconstructed SQuAD v1.1 logits-to-text binding.
import argparse, collections, json, math, os, pickle, subprocess, sys
import numpy as np
from transformers import BertTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
from create_squad_data import convert_examples_to_features, read_squad_examples

max_seq_length, max_query_length, doc_stride = 384, 64, 128
RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
dtype_map = {"int8":np.int8,"int16":np.int16,"int32":np.int32,"int64":np.int64,
             "float16":np.float16,"float32":np.float32,"float64":np.float64}

def _get_best_indexes(logits, n_best_size):
    return sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:n_best_size]

def _compute_softmax(scores):
    if not scores: return []
    m = max(scores); exps = [math.exp(x-m) for x in scores]; total = sum(exps)
    return [x/total for x in exps]

def get_final_text(pred_text, orig_text, do_lower_case=True):
    tok_text = " ".join(orig_text.strip().split())
    if do_lower_case: tok_text = tok_text.lower()
    pred = " ".join(pred_text.strip().split())
    if do_lower_case: pred = pred.lower()
    ns_orig = [(c,i) for i,c in enumerate(orig_text) if not c.isspace()]
    ns_tok = [(c,i) for i,c in enumerate(tok_text) if not c.isspace()]
    compact = pred.replace(" ", "")
    compact_tok = "".join(c for c,_ in ns_tok)
    compact_start = compact_tok.find(compact)
    if compact_start < 0 or compact_start + len(compact) > len(ns_orig): return orig_text
    return orig_text[ns_orig[compact_start][1]:ns_orig[compact_start+len(compact)-1][1]+1]

def write_predictions(results, features, examples, out_file, n_best_size=20, max_answer_length=30):
    by_example = collections.defaultdict(list)
    for f in features: by_example[f.example_index].append(f)
    by_id = {r.unique_id:r for r in results}
    predictions = collections.OrderedDict()
    for example_index, example in enumerate(examples):
        prelim = []
        for feature in by_example[example_index]:
            result = by_id.get(feature.unique_id)
            if result is None: continue
            starts = _get_best_indexes(result.start_logits, n_best_size)
            ends = _get_best_indexes(result.end_logits, n_best_size)
            for s in starts:
                for e in ends:
                    if s not in feature.token_to_orig_map or e not in feature.token_to_orig_map: continue
                    if not feature.token_is_max_context.get(s, False): continue
                    if e < s or e-s+1 > max_answer_length: continue
                    prelim.append((result.start_logits[s]+result.end_logits[e], feature, s, e))
        prelim.sort(key=lambda x:x[0], reverse=True)
        seen, texts = set(), []
        for _, feature, s, e in prelim:
            tok_text = " ".join(feature.tokens[s:e+1]).replace(" ##", "").replace("##", "")
            orig = " ".join(example.doc_tokens[feature.token_to_orig_map[s]:feature.token_to_orig_map[e]+1])
            text = get_final_text(tok_text, orig)
            if text not in seen:
                seen.add(text); texts.append(text)
            if len(texts) >= n_best_size: break
        predictions[example.qas_id] = texts[0] if texts else "empty"
    with open(out_file, "w") as f: json.dump(predictions, f, indent=2)

def load_loadgen_log(path, features, dtype=np.float32, output_transposed=False):
    with open(path) as f: entries=json.load(f)
    results=[]
    for entry in entries:
        logits=np.frombuffer(bytes.fromhex(entry["data"]),dtype)
        logits=logits.reshape(2,-1).T if output_transposed else logits.reshape(-1,2)
        start=np.full(max_seq_length,-10000.0); end=np.full(max_seq_length,-10000.0)
        start[:len(logits)]=logits[:,0]; end[:len(logits)]=logits[:,1]
        results.append(RawResult(features[entry["qsl_idx"]].unique_id,start.tolist(),end.tolist()))
    return results

def main():
    p=argparse.ArgumentParser(); p.add_argument("--vocab_file",default="build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt")
    p.add_argument("--val_data",default="build/data/dev-v1.1.json"); p.add_argument("--log_file",default="build/logs/mlperf_log_accuracy.json")
    p.add_argument("--out_file",default="build/result/predictions.json"); p.add_argument("--features_cache_file",default="eval_features.pickle")
    p.add_argument("--output_transposed",action="store_true"); p.add_argument("--output_dtype",default="float32",choices=dtype_map)
    p.add_argument("--max_examples",type=int); args=p.parse_args()
    print("Reading examples..."); examples=read_squad_examples(args.val_data,False,False)
    if os.path.exists(args.features_cache_file):
        print("Loading cached features from '%s'..."%args.features_cache_file)
        with open(args.features_cache_file,"rb") as f: features=pickle.load(f)
    else:
        print("Converting examples to features..."); features=[]; tok=BertTokenizer(args.vocab_file)
        convert_examples_to_features(examples,tok,max_seq_length,doc_stride,max_query_length,False,features.append,False)
        with open(args.features_cache_file,"wb") as f: pickle.dump(features,f)
    print("Loading LoadGen logs...")
    results=load_loadgen_log(args.log_file,features,dtype_map[args.output_dtype],args.output_transposed)
    write_predictions(results,features,examples,args.out_file)
    cmd=[sys.executable,os.path.join(os.path.dirname(__file__),"evaluate_v1.1.py"),args.val_data,args.out_file]
    if args.max_examples: cmd += ["--max_examples",str(args.max_examples)]
    subprocess.run(cmd,check=True)
if __name__ == "__main__": main()
'''

evaluator = r'''import argparse, collections, json, re, string
def normalize_answer(s):
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white(t): return " ".join(t.split())
    def punct(t): return "".join(c for c in t if c not in set(string.punctuation))
    return white(remove_articles(punct(s.lower())))
def f1_score(prediction, ground_truth):
    p=normalize_answer(prediction).split(); g=normalize_answer(ground_truth).split()
    common=collections.Counter(p)&collections.Counter(g); same=sum(common.values())
    if same==0: return 0
    precision=same/len(p); recall=same/len(g); return 2*precision*recall/(precision+recall)
def exact_match_score(prediction, ground_truth): return normalize_answer(prediction)==normalize_answer(ground_truth)
def metric_max_over_ground_truths(metric,prediction,truths): return max(metric(prediction,x) for x in truths)
def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset_file"); p.add_argument("prediction_file"); p.add_argument("--max_examples",type=int); a=p.parse_args()
    with open(a.dataset_file) as f: data=json.load(f)["data"]
    with open(a.prediction_file) as f: predictions=json.load(f)
    total=exact=f1=0
    for article in data:
      for paragraph in article["paragraphs"]:
       for qa in paragraph["qas"]:
        if a.max_examples is not None and total>=a.max_examples: break
        total+=1
        if qa["id"] not in predictions: continue
        truths=[x["text"] for x in qa["answers"]]; pred=predictions[qa["id"]]
        exact+=metric_max_over_ground_truths(exact_match_score,pred,truths)
        f1+=metric_max_over_ground_truths(f1_score,pred,truths)
       if a.max_examples is not None and total>=a.max_examples: break
      if a.max_examples is not None and total>=a.max_examples: break
    print(json.dumps({"exact_match":100.0*exact/total,"f1":100.0*f1/total}))
if __name__=="__main__": main()
'''

(bert / "accuracy-squad.py").write_text(accuracy)
(bert / "evaluate_v1.1.py").write_text(evaluator)
sut = bert / "pytorch_SUT.py"
text = sut.read_text()
needle = "        self.model.to(self.dev)\n        self.model.eval()"
replacement = "        self.model.to(self.dev)\n        print(\"pytorch device: {}\".format(next(self.model.parameters()).device))\n        self.model.eval()"
if needle in text:
    sut.write_text(text.replace(needle, replacement))
elif "pytorch device: " not in text:
    raise RuntimeError("could not instrument pytorch device log")
PY_PATCH

export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
export VOCAB_FILE="$MODEL_DIR/vocab.txt"
export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
export LOG_PATH="$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PATH="$VENV/bin:$PATH"
export SKIP_VERIFY_ACCURACY=1

{
  echo "GPU memory before run:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used \
    --format=csv,noheader,nounits
} >"$WORK_DIR/gpu-memory.log"

cd "$BERT_DIR"
"$VENV/bin/python" run.py --backend=pytorch --accuracy 2>&1 | tee "$WORK_DIR/run.log"

# run.py invokes this evaluator already; save a dedicated evaluator transcript too.
"$VENV/bin/python" accuracy-squad.py \
  --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" \
  --out_file "$BERT_DIR/build/result/predictions.json" \
  2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$WORK_DIR/evaluator.log"

test -s "$OUTPUT_DIR/mlperf_log_accuracy.json"
test -s "$OUTPUT_DIR/mlperf_log_detail.txt"
test -s "$OUTPUT_DIR/accuracy.txt"
