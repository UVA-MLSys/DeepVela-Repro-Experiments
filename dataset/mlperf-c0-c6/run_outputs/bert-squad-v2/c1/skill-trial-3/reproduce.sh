#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=/sfs/gpfs/tardis/project/MLSys/Agentic-Reproduce/.cache
SQUAD_DIR="$CACHE_ROOT/bert-squad/data"
MODEL_DIR="$CACHE_ROOT/bert-squad/model"
DEEPLEARNING_EXAMPLES_DIR="$CACHE_ROOT/bert-squad/DeepLearningExamples"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_EXCLUDE_NEWER=2026-11-10
export UV_LINK_MODE=copy

usage() {
  echo "usage: $0 --target bert-squad [--stage environment|prepare|run|evaluate] [--from environment|prepare|run|evaluate]" >&2
  exit 2
}

target=""
mode="all"
selected=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || usage; target="$2"; shift 2 ;;
    --stage) [[ $# -ge 2 ]] || usage; mode="stage"; selected="$2"; shift 2 ;;
    --from) [[ $# -ge 2 ]] || usage; mode="from"; selected="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$target" == "bert-squad" ]] || usage

root=$(pwd -P)
repo_dir="$root/repo"
target_dir="$root/$target"
output_dir=${OUTPUT_DIR:-"$target_dir/out"}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd -P)
venv="$root/.venv-bert-squad"
work_dir="$target_dir/work"

stages=(environment prepare run evaluate)
stage_index() {
  local needle=$1 i
  for i in "${!stages[@]}"; do
    [[ "${stages[$i]}" == "$needle" ]] && { echo "$i"; return; }
  done
  usage
}

if [[ "$mode" == "from" ]]; then
  from_i=$(stage_index "$selected")
  for ((i=from_i; i<${#stages[@]}; i++)); do
    rm -f "$target_dir/.stage.${stages[$i]}.ok"
    case "${stages[$i]}" in
      environment) rm -rf "$venv" "$target_dir/loadgen-src" ;;
      prepare) rm -rf "$work_dir" ;;
      run) rm -f "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" "$output_dir/mlperf_log_summary.txt" "$output_dir/mlperf_log_trace.json" ;;
      evaluate) rm -f "$output_dir/accuracy.txt" "$output_dir/predictions.json" ;;
    esac
  done
fi

run_environment() {
  [[ -f "$target_dir/.stage.environment.ok" ]] && return
  mkdir -p "$target_dir/loadgen-src"
  cp -a "$repo_dir/loadgen/." "$target_dir/loadgen-src/"
  {
    echo "Python environment build started: $(date -Is)"
    uv venv --python 3.11 --clear "$venv"
    uv pip install --python "$venv/bin/python" \
      setuptools==75.8.0 wheel==0.45.1 pybind11==2.11.1 \
      numpy==1.26.4 absl-py==2.1.0 six==1.16.0 \
      boto3==1.40.70 torch==2.5.1 transformers==4.46.3
    uv pip install --python "$venv/bin/python" --no-build-isolation "$target_dir/loadgen-src"
    "$venv/bin/python" - <<'PY'
import mlperf_loadgen, numpy, torch, transformers
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("loadgen", mlperf_loadgen.__file__)
PY
    echo "Python environment build finished: $(date -Is)"
  } 2>&1 | tee "$output_dir/build.log"
  touch "$target_dir/.stage.environment.ok"
}

run_prepare() {
  [[ -f "$target_dir/.stage.prepare.ok" ]] && return
  rm -rf "$work_dir"
  cp -a "$repo_dir/language/bert" "$work_dir"
  rm -rf "$work_dir/DeepLearningExamples"
  ln -s "$DEEPLEARNING_EXAMPLES_DIR" "$work_dir/DeepLearningExamples"
  mkdir -p "$work_dir/build/data/bert_tf_v1_1_large_fp32_384_v2" "$work_dir/build/result"
  ln -s "$SQUAD_DIR/dev-v1.1.json" "$work_dir/build/data/dev-v1.1.json"
  ln -s "$MODEL_DIR/model.pytorch" "$work_dir/build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch"
  ln -s "$MODEL_DIR/vocab.txt" "$work_dir/build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
  if ! grep -q 'pytorch device: ' "$work_dir/pytorch_SUT.py"; then
    patch -d "$work_dir" -p0 <<'PATCH'
--- pytorch_SUT.py
+++ pytorch_SUT.py
@@ -68,6 +68,7 @@
         )
         self.model.load_state_dict(torch.load(model_file), strict=False)
+        print("pytorch device: {}".format(next(self.model.parameters()).device))
 
         print("Constructing SUT...")
         self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
PATCH
  fi
  {
    echo "Input preparation started: $(date -Is)"
    sha256sum "$SQUAD_DIR/dev-v1.1.json" "$MODEL_DIR/model.pytorch" "$MODEL_DIR/vocab.txt"
    cd "$work_dir"
    "$venv/bin/python" - <<PY
import sys
sys.path.insert(0, "$DEEPLEARNING_EXAMPLES_DIR/PyTorch/LanguageModeling/BERT")
from squad_QSL import SQuAD_v1_QSL
qsl = SQuAD_v1_QSL()
print("prepared feature count:", qsl.count)
PY
    echo "Input preparation finished: $(date -Is)"
  } 2>&1 | tee "$output_dir/preprocess.log"
  touch "$target_dir/.stage.prepare.ok"
}

run_inference() {
  [[ -f "$target_dir/.stage.run.ok" ]] && return
  {
    echo "GPU check immediately before run: $(date -Is)"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  } | tee "$output_dir/gpu-before.log"
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if (( free_mib < 8192 )); then
    echo "Insufficient free GPU memory: ${free_mib} MiB; at least 8192 MiB required" | tee "$output_dir/run.log"
    return 1
  fi
  rm -f "$output_dir/mlperf_log_accuracy.json" "$output_dir/mlperf_log_detail.txt" "$output_dir/mlperf_log_summary.txt" "$output_dir/mlperf_log_trace.json"
  {
    echo "Accuracy run started: $(date -Is)"
    echo "Free GPU memory before model load: ${free_mib} MiB"
    cd "$work_dir"
    export LOG_PATH="$output_dir"
    export ML_MODEL_FILE_WITH_PATH="$MODEL_DIR/model.pytorch"
    export VOCAB_FILE="$MODEL_DIR/vocab.txt"
    export DATASET_FILE="$SQUAD_DIR/dev-v1.1.json"
    export SKIP_VERIFY_ACCURACY=1
    args=(--backend=pytorch --accuracy)
    if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
      args+=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
    fi
    "$venv/bin/python" run.py "${args[@]}"
    echo "Accuracy run finished: $(date -Is)"
  } 2>&1 | tee "$output_dir/run.log"
  test -s "$output_dir/mlperf_log_accuracy.json"
  test -s "$output_dir/mlperf_log_detail.txt"
  touch "$target_dir/.stage.run.ok"
}

run_evaluate() {
  [[ -f "$target_dir/.stage.evaluate.ok" ]] && return
  rm -f "$output_dir/accuracy.txt" "$output_dir/predictions.json"
  {
    cd "$work_dir"
    args=(--vocab_file "$MODEL_DIR/vocab.txt" --val_data "$SQUAD_DIR/dev-v1.1.json" --log_file "$output_dir/mlperf_log_accuracy.json" --out_file "$output_dir/predictions.json" --features_cache_file "$work_dir/eval_features.pickle")
    if [[ -n "${PREFLIGHT_MAX_EXAMPLES:-}" ]]; then
      args+=(--max_examples "$PREFLIGHT_MAX_EXAMPLES")
    fi
    "$venv/bin/python" accuracy-squad.py "${args[@]}"
  } 2> >(tee "$output_dir/evaluator.stderr.log" >&2) | tee "$output_dir/evaluator.log" "$output_dir/accuracy.txt"
  touch "$target_dir/.stage.evaluate.ok"
}

run_named() {
  case "$1" in
    environment) run_environment ;;
    prepare) run_prepare ;;
    run) run_inference ;;
    evaluate) run_evaluate ;;
    *) usage ;;
  esac
}

if [[ "$mode" == "stage" ]]; then
  run_named "$selected"
elif [[ "$mode" == "from" ]]; then
  from_i=$(stage_index "$selected")
  for ((i=from_i; i<${#stages[@]}; i++)); do run_named "${stages[$i]}"; done
else
  for stage in "${stages[@]}"; do run_named "$stage"; done
fi
