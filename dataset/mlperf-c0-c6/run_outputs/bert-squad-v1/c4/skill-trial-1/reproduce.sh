#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=""
STAGE=""
FROM=""
while (($#)); do
  case "$1" in
    --target) TARGET=${2:?missing target id}; shift 2 ;;
    --stage) STAGE=${2:?missing stage name}; shift 2 ;;
    --from) FROM=${2:?missing stage name}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "bert-squad" ]] || { echo "unsupported target: $TARGET" >&2; exit 2; }

: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR=$CACHE_ROOT/bert-squad/data
MODEL_DIR=$CACHE_ROOT/bert-squad/model
DEEPLEARNING_EXAMPLES_DIR=$CACHE_ROOT/bert-squad/DeepLearningExamples-c4
export UV_CACHE_DIR=$CACHE_ROOT/uv
export UV_EXCLUDE_NEWER=2026-11-10
export PYTHONPATH=$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
TARGET_ROOT=$(dirname "$OUTPUT_DIR")
WORK_DIR=$TARGET_ROOT/work
VENV=$TARGET_ROOT/venv
REPO_WORK=$WORK_DIR/repo
MARKERS=$TARGET_ROOT/markers
mkdir -p "$WORK_DIR" "$MARKERS"

stages=(environment prepare run evaluate)
want_stage() {
  local candidate=$1
  if [[ -n "$STAGE" ]]; then [[ "$candidate" == "$STAGE" ]]; return; fi
  if [[ -n "$FROM" ]]; then
    local seen=0 item
    for item in "${stages[@]}"; do
      [[ "$item" == "$FROM" ]] && seen=1
      if [[ "$item" == "$candidate" ]]; then ((seen)); return; fi
    done
    return 1
  fi
  return 0
}

if [[ -n "$FROM" ]]; then
  clear=0
  for item in "${stages[@]}"; do
    [[ "$item" == "$FROM" ]] && clear=1
    if ((clear)); then rm -f "$MARKERS/$item.ok"; fi
  done
fi

if want_stage environment && [[ ! -f "$MARKERS/environment.ok" ]]; then
  {
    echo "Creating Python environment"
    uv venv --python /home/mi3se/.local/bin/python3.11 --clear "$VENV"
    uv pip install --python "$VENV/bin/python" \
      torch==2.7.1 transformers==4.52.4 numpy==1.26.4 \
      absl-py==2.2.2 six==1.17.0 pybind11==2.13.6 boto3==1.40.0
  } 2>&1 | tee "$OUTPUT_DIR/environment.log"
  touch "$MARKERS/environment.ok"
fi

if want_stage prepare && [[ ! -f "$MARKERS/prepare.ok" ]]; then
  {
    echo "Staging pinned repository and supplied inputs"
    rm -rf "$REPO_WORK"
    cp -a "$SCRIPT_DIR/repo" "$REPO_WORK"
    ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$REPO_WORK/language/bert/DeepLearningExamples"
    mkdir -p "$REPO_WORK/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2"
    ln -s "$MODEL_DIR/model.pytorch" "$REPO_WORK/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
    ln -s "$MODEL_DIR/vocab.txt" "$REPO_WORK/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
    ln -s "$SQUAD_DIR/dev-v1.1.json" "$REPO_WORK/language/bert/build/data/dev-v1.1.json"

    python3 - "$REPO_WORK/language/bert/pytorch_SUT.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
s = s.replace('        self.version = transformers.__version__\n',
              '        self.version = transformers.__version__\n        print("pytorch device: {}".format(self.dev))\n')
s = s.replace('torch.load(model_file)', 'torch.load(model_file, map_location=self.dev, weights_only=True)')
p.write_text(s)
PY

    cat > "$REPO_WORK/language/bert/evaluate_v1.1.py" <<'PY'
import argparse, collections, json, re, string

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth):
    pred = normalize_answer(prediction).split()
    truth = normalize_answer(ground_truth).split()
    common = collections.Counter(pred) & collections.Counter(truth)
    same = sum(common.values())
    if same == 0: return 0.0
    precision = same / len(pred)
    recall = same / len(truth)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, truth) for truth in ground_truths)

def evaluate(dataset, predictions, max_examples=None):
    exact = f1 = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if max_examples is not None and total >= max_examples: break
                total += 1
                prediction = predictions.get(qa['id'], '')
                truths = [answer['text'] for answer in qa['answers']]
                exact += metric_max_over_ground_truths(exact_match_score, prediction, truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, truths)
            if max_examples is not None and total >= max_examples: break
        if max_examples is not None and total >= max_examples: break
    return {'exact_match': 100.0 * exact / total, 'f1': 100.0 * f1 / total}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file')
    parser.add_argument('prediction_file')
    parser.add_argument('--max_examples', type=int)
    args = parser.parse_args()
    with open(args.dataset_file) as f: dataset = json.load(f)['data']
    with open(args.prediction_file) as f: predictions = json.load(f)
    print(json.dumps(evaluate(dataset, predictions, args.max_examples)))

if __name__ == '__main__': main()
PY

    cat > "$REPO_WORK/language/bert/accuracy-squad.py" <<'PY'
import argparse, collections, json, os, pickle, subprocess, sys
import numpy as np
from transformers import BasicTokenizer, BertTokenizer
from create_squad_data import convert_examples_to_features, read_squad_examples

MAX_SEQ_LENGTH=384
RawResult=collections.namedtuple('RawResult', ['unique_id','start_logits','end_logits'])

def best_indexes(logits, n):
    return sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:n]

def strip_spaces(text):
    ns=[]; mapping={}
    for i,ch in enumerate(text):
        if ch == ' ': continue
        mapping[len(ns)] = i; ns.append(ch)
    return ''.join(ns), mapping

def final_text(pred_text, orig_text):
    tok_text=' '.join(BasicTokenizer(do_lower_case=True).tokenize(orig_text))
    start=tok_text.find(pred_text)
    if start == -1: return orig_text
    end=start+len(pred_text)-1
    orig_ns, orig_map=strip_spaces(orig_text)
    tok_ns, tok_map=strip_spaces(tok_text)
    if len(orig_ns) != len(tok_ns): return orig_text
    tok_to_ns={v:k for k,v in tok_map.items()}
    if start not in tok_to_ns or end not in tok_to_ns: return orig_text
    ns_start=tok_to_ns[start]; ns_end=tok_to_ns[end]
    if ns_start not in orig_map or ns_end not in orig_map: return orig_text
    return orig_text[orig_map[ns_start]:orig_map[ns_end]+1]

def make_predictions(results, features, examples):
    result_by_id={r.unique_id:r for r in results}
    features_by_example=collections.defaultdict(list)
    for feature in features: features_by_example[feature.example_index].append(feature)
    predictions=collections.OrderedDict()
    for example_index, example in enumerate(examples):
        candidates=[]
        for feature in features_by_example[example_index]:
            result=result_by_id.get(feature.unique_id)
            if result is None: continue
            for start in best_indexes(result.start_logits, 20):
                for end in best_indexes(result.end_logits, 20):
                    if start not in feature.token_to_orig_map: continue
                    if end not in feature.token_to_orig_map: continue
                    if not feature.token_is_max_context.get(start, False): continue
                    if end < start or end-start+1 > 30: continue
                    candidates.append((result.start_logits[start]+result.end_logits[end], feature, start, end))
        if not candidates:
            predictions[example.qas_id]=''; continue
        _, feature, start, end=max(candidates, key=lambda x:x[0])
        tok_text=' '.join(feature.tokens[start:end+1]).replace(' ##','').replace('##','').strip()
        orig_start=feature.token_to_orig_map[start]; orig_end=feature.token_to_orig_map[end]
        orig_text=' '.join(example.doc_tokens[orig_start:orig_end+1])
        predictions[example.qas_id]=final_text(tok_text, orig_text)
    return predictions

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vocab_file', default='build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt')
    p.add_argument('--val_data', default='build/data/dev-v1.1.json')
    p.add_argument('--log_file', default='build/logs/mlperf_log_accuracy.json')
    p.add_argument('--out_file', default='build/result/predictions.json')
    p.add_argument('--features_cache_file', default='eval_features.pickle')
    p.add_argument('--max_examples', type=int)
    args=p.parse_args()
    examples=read_squad_examples(args.val_data, False, False)
    with open(args.features_cache_file,'rb') as f: features=pickle.load(f)
    with open(args.log_file) as f: records=json.load(f)
    results=[]
    for record in records:
        feature=features[record['qsl_idx']]
        logits=np.frombuffer(bytes.fromhex(record['data']), np.float32).reshape(-1,2)
        start=np.full(MAX_SEQ_LENGTH,-10000.0); end=np.full(MAX_SEQ_LENGTH,-10000.0)
        start[:len(logits)]=logits[:,0]; end[:len(logits)]=logits[:,1]
        results.append(RawResult(feature.unique_id,start.tolist(),end.tolist()))
    predictions=make_predictions(results,features,examples)
    os.makedirs(os.path.dirname(args.out_file),exist_ok=True)
    with open(args.out_file,'w') as f: json.dump(predictions,f,indent=2)
    cmd=[sys.executable,os.path.join(os.path.dirname(__file__),'evaluate_v1.1.py'),args.val_data,args.out_file]
    if args.max_examples: cmd += ['--max_examples',str(args.max_examples)]
    subprocess.check_call(cmd)

if __name__ == '__main__': main()
PY

    uv pip install --python "$VENV/bin/python" "$REPO_WORK/loadgen"
    cd "$REPO_WORK/language/bert"
    "$VENV/bin/python" - <<'PY'
from squad_QSL import SQuAD_v1_QSL
qsl=SQuAD_v1_QSL()
print('feature_count:', len(qsl.eval_features))
PY
  } 2>&1 | tee "$OUTPUT_DIR/prepare.log"
  touch "$MARKERS/prepare.ok"
fi

if want_stage run && [[ ! -f "$MARKERS/run.ok" ]]; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "$OUTPUT_DIR/gpu-before-run.log"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  (( free_mib >= 12000 )) || { echo "insufficient free GPU memory: ${free_mib} MiB" | tee -a "$OUTPUT_DIR/run.log"; exit 1; }
  cd "$REPO_WORK/language/bert"
  run_args=(--backend=pytorch --accuracy)
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then run_args+=(--max_examples "$PREFLIGHT_MAX_EXAMPLES"); fi
  export LOG_PATH=$OUTPUT_DIR
  export SKIP_VERIFY_ACCURACY=1
  export ML_MODEL_FILE_WITH_PATH=$MODEL_DIR/model.pytorch
  export VOCAB_FILE=$MODEL_DIR/vocab.txt
  export DATASET_FILE=$SQUAD_DIR/dev-v1.1.json
  "$VENV/bin/python" run.py "${run_args[@]}" 2>&1 | tee "$OUTPUT_DIR/run.log"
  touch "$MARKERS/run.ok"
fi

if want_stage evaluate && [[ ! -f "$MARKERS/evaluate.ok" ]]; then
  cd "$REPO_WORK/language/bert"
  eval_args=(--vocab_file "$MODEL_DIR/vocab.txt" --val_data "$SQUAD_DIR/dev-v1.1.json" --log_file "$OUTPUT_DIR/mlperf_log_accuracy.json" --out_file "$WORK_DIR/predictions.json" --features_cache_file eval_features.pickle)
  if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then eval_args+=(--max_examples "$PREFLIGHT_MAX_EXAMPLES"); fi
  "$VENV/bin/python" accuracy-squad.py "${eval_args[@]}" 2>&1 | tee "$OUTPUT_DIR/accuracy.txt" "$OUTPUT_DIR/evaluator.log"
  touch "$MARKERS/evaluate.ok"
fi
