"""Generate the task dataset (docs/dataset-schema.md): MLPerf Inference, HeCBench, PDEBench, ClimateLearn, OGB.

Agent-visible files go to dataset/tasks/<source_id>.json in this repository. Gold files go to
$EVALUATION_BUNDLE_DIR/dataset/gold/<source_id>.json, outside the repository the agent reads.

Run from the repository root: EVALUATION_BUNDLE_DIR=../evaluation-profiles .venv/bin/python -m dataset.build_dataset
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MLPERF_REPO = ROOT.parent / "OSS" / "inference"
HECBENCH_REPO = ROOT.parent / "OSS" / "HeCBench"
INFERENCE_REPO = ROOT.parent / "OSS" / "inference"
VISIBLE_DIR = ROOT / "dataset" / "tasks"
GOLD_DIR = Path(os.environ["EVALUATION_BUNDLE_DIR"]).resolve() / "dataset" / "gold"
SUPPORT_LEVEL = "readme_only"

# Authored MLPerf task bundle -> (source id, directory of the source inside the inference repository).
# Tasks outside this table have no user-confirmed reference yet and are not in the dataset.
MLPERF_SOURCES = {
    "classification-and-detection-all": ("mlperf-vision-classification-and-detection", "vision/classification_and_detection"),
    "3d-unet-kits19": ("mlperf-vision-medical-imaging-3d-unet", "vision/medical_imaging/3d-unet-kits19"),
    "bert-squad": ("mlperf-language-bert", "language/bert"),
    "gpt-j": ("mlperf-language-gpt-j", "language/gpt-j"),
    "llama2-70b-openorca": ("mlperf-language-llama2-70b", "language/llama2-70b"),
    "llama3.1-8b-cnndm-datacenter": ("mlperf-language-llama3-1-8b", "language/llama3.1-8b"),
    "llama3.1-8b-cnndm-edge": ("mlperf-language-llama3-1-8b-edge", "language/llama3.1-8b"),
    "mixtral-8x7b-openorca-gsm8k-mbxp": ("mlperf-language-mixtral-8x7b", "language/mixtral-8x7b"),
    "whisper-librispeech": ("mlperf-speech2text-whisper", "speech2text"),
    "stable-diffusion-xl-coco2014": ("mlperf-text-to-image-sdxl", "text_to_image"),
}
# Bundle tasks whose targets repeat targets of another bundle task; checked for equal references.
MLPERF_DUPLICATES = {
    "resnet50-imagenet-multitarget": "classification-and-detection-all",
    "resnet50-pytorch-imagenet": "classification-and-detection-all",
}

# Agent-visible fields of each MLPerf source (docs/dataset-schema.md), taken from the task prompts the user
# confirmed in the authored bundles. `task_prompt` is the README's documented entry command, as in CORE-Bench and
# SocSci-Repro-Bench; the environment procedure stays out (CORE-Bench Hard).
MLPERF_FORBIDDEN = ("the MLPerf Inference results repositories (mlcommons/inference_results_*), published MLPerf "
                    "submissions, or the MLCommons results tables")
ONE_A100 = "one NVIDIA A100 80GB GPU per run."
TWO_A100 = "two NVIDIA A100 80GB GPUs per run."
# accuracy.txt is the MLPerf submission file holding the accuracy script's output (submission_checker/constants.py).
LOADGEN_OUTPUTS = {"mlperf_log_accuracy.json": "LoadGen's accuracy log", "mlperf_log_detail.txt": "LoadGen's detail log",
                   "accuracy.txt": "the evaluator's stdout"}
PYTORCH_DEVICE = {"pytorch device: ": "the device the model's parameters are on"}
TORCH_CUDA = {"torch cuda: ": "the value of `torch.cuda.is_available()` and the GPU names, written by the process that runs the model"}
CACHE = ".cache"


def mlperf_policy(inputs: dict, cap: str | None, platform: str = ONE_A100) -> dict:
    policy = {"compute": "slurm", "platform": platform, "inputs": inputs, "forbidden": MLPERF_FORBIDDEN}
    if cap:
        policy["wall_clock_cap"] = cap
    return policy


def imagenet_task(command: str, model: str, log_lines: dict | None, platform: str | None = None) -> dict:
    task = {"task_prompt": f"Run `{command}` (Offline scenario, LoadGen accuracy mode) on {model} over the ImageNet-2012 "
                           "validation set, then tools/accuracy-imagenet.py on its LoadGen accuracy log.",
            "results": ["accuracy"], "outputs": LOADGEN_OUTPUTS}
    if log_lines:
        task["log_lines"] = log_lines
    if platform:
        task["platform"] = platform
    return task


def retinanet_task(command: str, model: str, log_lines: dict) -> dict:
    return {"task_prompt": f"Run `{command}` (Offline scenario, LoadGen accuracy mode) on {model} over the OpenImages mlperf "
                           "validation set at 800x800, then tools/accuracy-openimages.py on its LoadGen accuracy log.",
            "results": ["mAP"], "outputs": LOADGEN_OUTPUTS, "log_lines": log_lines}


MLPERF_VISIBLE = {
    "classification-and-detection-all": {
        "domain": "image classification and object detection", "language": "Python",
        "policy": mlperf_policy({
            "IMAGENET_DIR": {"path": f"{CACHE}/resnet50-pytorch/imagenet_val/imagenet2012"},
            "OPENIMAGES_DIR": {"path": f"{CACHE}/classification-and-detection/openimages"},
            "TF_MODEL_DIR": {"path": f"{CACHE}/classification-and-detection/resnet50-tf/model"},
            "ONNX_MODEL_DIR": {"path": f"{CACHE}/resnet50-onnxruntime/model"},
            "PYTORCH_MODEL_DIR": {"path": f"{CACHE}/resnet50-pytorch/model"},
            "INT8_MODEL_DIR": {"path": f"{CACHE}/classification-and-detection/resnet50-pytorch-int8/model"},
            "NCNN_MODEL_DIR": {"path": f"{CACHE}/classification-and-detection/resnet50-ncnn/model"},
            "RETINANET_MODEL_DIR": {"path": f"{CACHE}/classification-and-detection/retinanet/model"},
        }, None),
        "tasks": {
            "resnet50-tf": imagenet_task("./run_local.sh tf resnet50 gpu --accuracy", "the fp32 graph resnet50_v1.pb",
                                         {"tensorflow gpus: ": "`tf.config.list_physical_devices('GPU')`"}),
            "resnet50-onnxruntime": imagenet_task("./run_local.sh onnxruntime resnet50 gpu --accuracy", "the fp32 model resnet50_v1.onnx",
                                                  {"onnxruntime providers: ": "the active providers of its inference session (`InferenceSession.get_providers()`)"}),
            "resnet50-pytorch": imagenet_task("./run_local.sh pytorch resnet50 gpu --accuracy", "the fp32 checkpoint resnet50-19c8e357.pth",
                                              PYTORCH_DEVICE),
            "resnet50-pytorch-int8": imagenet_task("./run_local.sh tvm-pytorch resnet50 cpu --accuracy",
                                                   "the A:int8 W:uint8 model resnet50_INT8bit_quantized.pt", None, "CPU only"),
            "resnet50-ncnn": imagenet_task("./run_local.sh ncnn resnet50 cpu --accuracy", "the fp32 ncnn model resnet50_v1",
                                           None, "CPU only"),
            "retinanet-pytorch": retinanet_task("./run_local.sh pytorch retinanet gpu --accuracy", "the fp32 checkpoint resnext50_32x4d_fpn.pth",
                                                PYTORCH_DEVICE),
            "retinanet-onnxruntime": retinanet_task("./run_local.sh onnxruntime retinanet gpu --accuracy", "the fp32 model resnext50_32x4d_fpn.onnx",
                                                    {"onnxruntime providers: ": "the active providers of its inference session (`InferenceSession.get_providers()`)"}),
        },
    },
    "3d-unet-kits19": {
        "domain": "medical imaging", "language": "Python",
        "policy": mlperf_policy({
            "KITS19_RAW_DIR": {"path": f"{CACHE}/3d-unet-kits19/kits19_raw_data/kits19/data",
                               "holds": "the raw imaging and segmentation of the cases in meta/inference_cases.json, except case_00400, "
                                        "the documented duplicate of case_00185"},
            "MODEL_DIR": {"path": f"{CACHE}/3d-unet-kits19/model", "holds": "3dunet_kits19_pytorch.ptc"},
        }, "1 hour"),
        "tasks": {"3d-unet": {
            "task_prompt": "Run `python3 run.py --backend=pytorch --scenario=Offline --accuracy` on the fp32 model in MODEL_DIR over the "
                           "KiTS19 cases in meta/inference_cases.json, then accuracy_kits.py on its LoadGen accuracy log.",
            "results": ["mean", "kidney", "tumor"], "outputs": LOADGEN_OUTPUTS, "log_lines": PYTORCH_DEVICE}},
    },
    "bert-squad": {
        "domain": "question answering", "language": "Python",
        "policy": mlperf_policy({
            "SQUAD_DIR": {"path": f"{CACHE}/bert-squad/data", "holds": "dev-v1.1.json"},
            "MODEL_DIR": {"path": f"{CACHE}/bert-squad/model", "holds": "model.pytorch and vocab.txt"},
            "DEEPLEARNING_EXAMPLES_DIR": {"path": f"{CACHE}/bert-squad/DeepLearningExamples",
                                          "holds": "a clone of https://github.com/NVIDIA/DeepLearningExamples.git at commit "
                                                   "729963dd47e7c8bd462ad10bfac7a7b0b604e6dd, the directory language/bert/pytorch_SUT.py imports "
                                                   "from as language/bert/DeepLearningExamples; the pinned repository lists it in .gitmodules "
                                                   "but does not track it, so it is not in ./repo"},
        }, "2 hours"),
        "tasks": {"bert-squad": {
            "task_prompt": "Run `python3 run.py --backend=pytorch --accuracy` (scenario Offline, the default) on the BERT-Large fp32 model in MODEL_DIR "
                           "over the SQuAD v1.1 validation set, then accuracy-squad.py (which calls evaluate_v1.1.py) on its LoadGen "
                           "accuracy log.",
            "results": ["exact_match", "f1"], "outputs": LOADGEN_OUTPUTS, "log_lines": PYTORCH_DEVICE}},
    },
    "gpt-j": {
        "domain": "text summarization", "language": "Python",
        "policy": mlperf_policy({
            "DATASET_FILE": {"path": f"{CACHE}/gpt-j/data/cnn_eval.json"},
            "MODEL_DIR": {"path": f"{CACHE}/gpt-j/model"},
        }, "26 hours"),
        "tasks": {"gpt-j": {
            "task_prompt": "Run `python main.py --scenario=Offline --accuracy --gpu` as in the README's \"Running the Benchmark\" section, on the "
                           "fine-tuned GPT-J-6B checkpoint in MODEL_DIR at the fp32 precision and beam size 4 of the README's reference ROUGE "
                           "scores, over the 13,368-sample CNN/DailyMail validation set in DATASET_FILE, then evaluation.py on its "
                           "LoadGen accuracy log.",
            "results": ["rouge1", "rouge2", "rougeL", "rougeLsum", "gen_len", "gen_num"], "outputs": LOADGEN_OUTPUTS,
            "log_lines": PYTORCH_DEVICE}},
    },
    "llama2-70b-openorca": {
        "domain": "text generation", "language": "Python",
        "policy": mlperf_policy({
            "DATASET_FILE": {"path": f"{CACHE}/llama2-70b/dataset/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz"},
            "MODEL_DIR": {"path": f"{CACHE}/llama2-70b/model", "holds": "the Llama-2-70b-chat-hf weights"},
        }, "6 hours", TWO_A100),
        "tasks": {"llama2-70b": {
            "task_prompt": "Run `python -u main.py --scenario Offline --accuracy` at the README precision over the 24,576-sample OpenOrca "
                           "validation set in DATASET_FILE, then evaluate-accuracy.py on its LoadGen accuracy log.",
            "results": ["rouge1", "rouge2", "rougeL", "tokens_per_sample"], "outputs": LOADGEN_OUTPUTS, "log_lines": TORCH_CUDA,
            "hints": ["Any route the benchmark's own main.py supports is valid."]}},
    },
    "llama3.1-8b-cnndm-datacenter": {
        "domain": "text summarization", "language": "Python",
        "policy": mlperf_policy({
            "DATASET_FILE": {"path": f"{CACHE}/llama3.1-8b/dataset/cnn_eval.json"},
            "MODEL_DIR": {"path": f"{CACHE}/llama3.1-8b/model", "holds": "the Llama-3.1-8B-Instruct weights"},
        }, "2 hours"),
        "tasks": {"llama3.1-8b": {
            "task_prompt": "Run `python -u main.py --scenario Offline --accuracy --vllm` (datacenter) at the README precision over the 13,368-sample "
                           "CNN/DailyMail validation set in DATASET_FILE, then evaluation.py on its LoadGen accuracy log.",
            "results": ["rouge1", "rouge2", "rougeL", "rougeLsum", "gen_len"], "outputs": LOADGEN_OUTPUTS, "log_lines": TORCH_CUDA}},
    },
    "llama3.1-8b-cnndm-edge": {
        "domain": "text summarization", "language": "Python",
        "policy": mlperf_policy({
            "DATASET_FILE": {"path": f"{CACHE}/llama3.1-8b/dataset/sample_cnn_eval_5000.json"},
            "MODEL_DIR": {"path": f"{CACHE}/llama3.1-8b/model", "holds": "the Llama-3.1-8B-Instruct weights"},
        }, "1 hour"),
        "tasks": {"llama3.1-8b-edge": {
            "task_prompt": "Run `python -u main.py --lg-model-name llama3_1-8b-edge --scenario Offline --accuracy --vllm` (edge) at the README "
                           "precision over the 5,000-sample CNN/DailyMail validation subset in DATASET_FILE, then evaluation.py on its "
                           "LoadGen accuracy log.",
            "results": ["rouge1", "rouge2", "rougeL", "rougeLsum", "gen_len"], "outputs": LOADGEN_OUTPUTS, "log_lines": TORCH_CUDA}},
    },
    "mixtral-8x7b-openorca-gsm8k-mbxp": {
        "domain": "text generation", "language": "Python",
        "policy": mlperf_policy({
            "DATASET_FILE": {"path": f"{CACHE}/mixtral-8x7b/dataset/09292024_mixtral_15k_mintoken2_v1.pkl"},
            "MODEL_DIR": {"path": f"{CACHE}/mixtral-8x7b/model", "holds": "the Mixtral-8x7B-Instruct-v0.1 weights"},
        }, "8 hours", TWO_A100),
        "tasks": {"mixtral-8x7b": {
            "task_prompt": "Run `python -u main.py --scenario Offline --accuracy` at the README precision over the 15,000-sample OpenOrca, "
                           "GSM8K and MBXP validation set in DATASET_FILE, then evaluate-accuracy.py on its LoadGen accuracy log.",
            "results": ["rouge1", "rouge2", "rougeL", "gsm8k", "mbxp", "tokens_per_sample"], "outputs": LOADGEN_OUTPUTS,
            "log_lines": TORCH_CUDA,
            "hints": ["Known properties of the reference code, measured on two A100: the checkpoint in MODEL_DIR is stored in fp32 (174 GB) "
                      "and the reference SUT loads it at that precision, which does not fit in 2 x 80 GB, so part of the model is offloaded "
                      "to the CPU and generation takes about 346 s per sample. main.py defaults --batch-size to 1 (13.7 s per sample with a "
                      "fp16 load, about 57 hours for 15,000 samples), and the reference code fails at batch sizes above 1. Size the run so "
                      "the full pass fits the time limit."]}},
    },
    "whisper-librispeech": {
        "domain": "speech recognition", "language": "Python",
        "policy": mlperf_policy({
            "LIBRISPEECH_DIR": {"path": f"{CACHE}/whisper/LibriSpeech"},
            "MODEL_DIR": {"path": f"{CACHE}/whisper/model", "holds": "the whisper-large-v3 weights"},
        }, "2 hours"),
        "tasks": {"whisper-large-v3": {
            "task_prompt": "Run `./reference_mlperf_accuracy.sh` (Offline scenario, LoadGen accuracy mode) at the reference precision over "
                           "LibriSpeech dev-clean and dev-other, then accuracy_eval.py on its LoadGen accuracy log.",
            "results": ["Word Error Rate"], "outputs": LOADGEN_OUTPUTS, "log_lines": TORCH_CUDA}},
    },
    "stable-diffusion-xl-coco2014": {
        "domain": "text-to-image generation", "language": "Python",
        "policy": mlperf_policy({
            "DATASET_DIR": {"path": f"{CACHE}/text_to_image/coco2014", "holds": "the COCO-2014 5,000-caption validation subset (captions, reference images, latents)"},
            "MODEL_DIR": {"path": f"{CACHE}/text_to_image/model", "holds": "the Stable Diffusion XL 1.0 fp16 checkpoint"},
        }, "6 hours"),
        "tasks": {"stable-diffusion-xl": {
            "task_prompt": "Run `python3 main.py --dataset \"coco-1024\" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --accuracy "
                           "--model-path model/` with --dtype fp16 (the fp16 checkpoint), --device cuda and --scenario Offline over the 5,000 "
                           "COCO-2014 captions in DATASET_DIR, then tools/accuracy_coco.py on its LoadGen accuracy log.",
            "results": ["FID_SCORE"], "outputs": LOADGEN_OUTPUTS, "log_lines": TORCH_CUDA}},
    },
}

# Source-level policy, answered once (user, 2026-10-08) instead of per task. Stated in each task prompt.
HECBENCH_POLICY = {
    "compute": "local",
    "platform": "this machine's GPU; the CUDA toolkit comes from the environment module system.",
    "inputs": {},
    "forbidden": "published HeCBench results or timings, or other implementations' outputs",
    "wall_clock_cap": "5 hours",
}
HECBENCH_OUTPUTS = {"build.log": "the build's stdout and stderr", "run.log": "the program's stdout and stderr"}
# Labels set by running the program, overriding the static grep (docs/dataset-schema.md): miniDGS's PASS match is in
# ParMetis headers, and a smoke run printed no verdict.
HECBENCH_VERDICT_RUN = {"miniDGS-cuda": "exit_0"}

# HeCBench README "Dependencies" section -> env group. Everything else is cuda-base.
HECBENCH_GROUPS = {
    "cuda-boost": ["hbc", "ge-spmm", "mmcsf", "warpsort"],
    "cuda-boost-bzip2": ["gerbil"],
    "cuda-mpi": ["miniDGS", "miniWeather", "sparkler", "allreduce", "halo-finder"],
    "cuda-mpi-nccl": ["ccl"],
    "cuda-eigen": ["xlqc"],
    "cuda-gdal": ["stsg"],
}


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def paper_contract(*, evaluator: str, result: str, value: float, rule: str, paper: str, paper_sha256: str, table: str, excerpt: str,
                   std: float | None = None) -> dict:
    """A gold entry that validates as evaluation.schema.QualityContract. The reference is a paper table, so the source
    excerpt records the PDF as downloaded (arXiv regenerates PDFs; the hash fixes what was read), not a repository file."""
    anchor = {"source": f"{paper} {table}", "value": value}
    if std is not None:
        anchor["std"] = std
    return {
        "evaluator_identity": evaluator,
        "result_name_type_and_direction": result,
        "reference_value": value,
        "acceptance_rule": rule,
        "source_path_sha256_excerpt": {"path": f"{paper} {table} (PDF as downloaded 2026-10-08)", "sha256": paper_sha256, "excerpt": excerpt},
        "matching_external_anchors": [anchor],
    }


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def build_mlperf() -> list[str]:
    root = Path(os.environ["EVALUATION_BUNDLE_DIR"])
    # `mlperf_inference_accuracy` is frozen; a task authored later lives only under `reproduce`.
    manifests, profile_of = {}, {}
    for profile in ("reproduce", "mlperf_inference_accuracy"):
        for p in sorted((root / profile / "tasks").glob("*/manifest.yaml")):
            if p.parent.name not in manifests:
                manifests[p.parent.name], profile_of[p.parent.name] = yaml.safe_load(p.read_text()), profile
    # The checkout is the upstream mlcommons/inference (it replaced a fork, whose pinned commit upstream lacks), and
    # `undocumented_commands` reads its docs, so the record pins the checkout's commit, not the bundle's.
    inference_head = git_head(INFERENCE_REPO)
    written = []
    for bundle_task, (source_id, subdir) in MLPERF_SOURCES.items():
        manifest = manifests[bundle_task]
        visible = MLPERF_VISIBLE[bundle_task]
        assert set(visible["tasks"]) == set(manifest["targets"]), (bundle_task, sorted(manifest["targets"]))
        visible_tasks = [{"id": t, **visible["tasks"][t]} for t in sorted(manifest["targets"])]
        gold_tasks = {t: spec["quality_contract"] for t, spec in sorted(manifest["targets"].items())}
        # Each target of these sources builds its own environment in the runs inspected.
        env_groups = [{"id": f"{t['id']}-env", "tasks": [t["id"]]} for t in visible_tasks]
        write(VISIBLE_DIR / f"{source_id}.json", {
            "source_id": source_id,
            "repo": {"path": "../OSS/inference", "commit": inference_head, "directory": subdir},
            "domain": visible["domain"], "language": visible["language"],
            "support_level": SUPPORT_LEVEL,
            "policy": visible["policy"],
            "env_groups": env_groups,
            "tasks": visible_tasks,
        })
        write(GOLD_DIR / f"{source_id}.json", {"source_id": source_id, "authored_in": f"{profile_of[bundle_task]}/{bundle_task}", "tasks": gold_tasks})
        written.append(source_id)
    for dup, original in MLPERF_DUPLICATES.items():
        for target_id, spec in (manifests[dup].get("targets") or {}).items():
            ref = spec["quality_contract"]["reference_value"]
            assert manifests[original]["targets"][target_id]["quality_contract"]["reference_value"] == ref, (dup, target_id)
        if "targets" not in manifests[dup] or not manifests[dup]["targets"]:
            ref = manifests[dup]["quality_contract"]["reference_value"]
            assert manifests[original]["targets"]["resnet50-pytorch"]["quality_contract"]["reference_value"] == ref, dup
    return written


def run_recipe(makefile: Path) -> list[str] | None:
    lines, inside, continued = [], False, False
    for line in makefile.read_text().splitlines():
        if re.match(r"^run\s*:", line):
            inside = True
            continue
        if inside:
            if continued:  # a line ending in a backslash continues the same command
                lines[-1] = f"{lines[-1][:-1].rstrip()} {line.strip()}"
            elif line.startswith("\t"):
                lines.append(line.strip())
            else:
                break
            continued = lines[-1].endswith("\\")
    return lines or None


def source_files(task_dir: Path) -> list[Path]:
    """The task's own sources plus what its Makefile reaches outside the directory: files named as
    `../dir/file` and headers in `-I../dir` include directories."""
    own = [p for p in task_dir.rglob("*") if p.is_file() and p.suffix in {".cu", ".cpp", ".cc", ".h", ".cuh"}]
    makefile = task_dir / "Makefile"
    if not makefile.exists():
        return own
    text = makefile.read_text(errors="ignore")
    extra: set[Path] = set()
    for include_dir in re.findall(r"-I\s*(\.\./[\w.\-/]+)", text):
        directory = (task_dir / include_dir).resolve()
        if directory.is_dir():
            extra |= {p for p in directory.iterdir() if p.is_file() and p.suffix in {".h", ".cuh", ".hpp"}}
    for ref in re.findall(r"(?<![\w/.\-])(\.\./[\w.\-/]+\.(?:h|cuh|hpp|cu|cpp|cc))", text):
        path = (task_dir / ref).resolve()
        if path.is_file():
            extra.add(path)
    return own + sorted(extra)


def build_hecbench() -> list[str]:
    meta = yaml.safe_load((HECBENCH_REPO / "benchmarks.yaml").read_text())
    group_of = {f"{name}-cuda": group for group, names in HECBENCH_GROUPS.items() for name in names}
    visible_tasks, gold_tasks, groups, excluded = [], {}, {}, []
    for name, entry in sorted(meta.items()):
        if "cuda" not in entry.get("models", []):
            continue
        task_id = f"{name}-cuda"
        task_dir = HECBENCH_REPO / "src" / task_id
        if not task_dir.is_dir():
            excluded.append({"id": task_id, "reason": "benchmarks.yaml lists cuda but src/ has no such directory"})
            continue
        makefile = task_dir / "Makefile"
        recipe = run_recipe(makefile) if makefile.exists() else None
        test = entry.get("test") or {}
        if recipe:
            entry_instruction = f"Build and run src/{task_id} as its Makefile `run:` recipe runs it: `{'; '.join(recipe)}`."
        else:
            args = " ".join(test.get("args") or [])
            entry_instruction = (f"Build src/{task_id} with its CMakeLists.txt and run the program"
                                 + (f" with the arguments benchmarks.yaml documents: `{args}`." if args else "."))
        # A target's inputs are the DVC files of its own directory and of every src/ directory its
        # files reference as `../<dir>/` (e.g. aes-cuda runs ../urng-sycl/URNG_Input.bmp), or as
        # `../data/<sub>/`.
        text = " ".join([*(recipe or [])] + [p.read_text(errors="ignore") for p in task_dir.rglob("*")
                                             if p.is_file() and p.suffix in {"", ".txt", ".cu", ".cpp", ".cc", ".c", ".h", ".cuh", ".hpp"}])
        dirs = [task_dir] + [HECBENCH_REPO / "src" / d for d in sorted(set(re.findall(r"\.\./(\w[\w.+\-]*)", text)) - {"data", task_id})
                             if (HECBENCH_REPO / "src" / d).is_dir()]
        inputs = sorted(str(p.relative_to(HECBENCH_REPO)) for d in dirs for p in d.glob("*.dvc"))
        for sub in sorted(set(re.findall(r"\.\./data/([^/\s]+)/", text))):
            inputs += sorted(str(p.relative_to(HECBENCH_REPO)) for p in (HECBENCH_REPO / "src" / "data" / sub).rglob("*.dvc"))
        sources = [p.read_text(errors="ignore") for p in source_files(task_dir)]
        gold = {"expected_verdict": HECBENCH_VERDICT_RUN.get(task_id) or ("PASS" if any("PASS" in s for s in sources) else "exit_0")}
        task = {"id": task_id, "task_prompt": entry_instruction,
                "results": ["exit_status", "verdict"] if gold["expected_verdict"] == "PASS" else ["exit_status"],
                "outputs": HECBENCH_OUTPUTS,
                "log_lines": {"exit status: ": "the program's exit code, as the last line of run.log"}}
        if inputs:
            task["inputs"] = inputs
        visible_tasks.append(task)
        if test.get("regex"):
            gold["timing_regex"] = test["regex"]
        gold_tasks[task_id] = gold
        groups.setdefault(group_of.get(task_id, "cuda-base"), []).append(task_id)
    categories = {n: (meta[n].get("categories") or [None])[0] for n in meta}
    source_id = "hecbench-cuda"
    write(VISIBLE_DIR / f"{source_id}.json", {
        "source_id": source_id,
        "repo": {"path": "../OSS/HeCBench", "commit": git_head(HECBENCH_REPO)},
        "language": "CUDA",
        "support_level": SUPPORT_LEVEL,
        "policy": HECBENCH_POLICY,
        "env_groups": [{"id": g, "tasks": t} for g, t in sorted(groups.items())],
        "tasks": visible_tasks,
        "excluded": excluded,
        "domain": {t["id"]: categories[t["id"].removesuffix("-cuda")] for t in visible_tasks},
    })
    write(GOLD_DIR / f"{source_id}.json", {"source_id": source_id, "tasks": gold_tasks})
    return [source_id]


PDEBENCH_COMMIT = "4ff3e3a4aa1561721b5571fa3a048a0a463e0568"
PDEBENCH_PDF_SHA256 = "756882062e8e62f70ffdb4339b7f0824201783b336c876c873d83a89cf646386"
PDEBENCH_TABLE_NAME = {"diff-sorp": "Diffusion-sorption", "diff-react": "2D diffusion-reaction", "swe": "Shallow-water equation"}

# PDE -> (config under pdebench/models/config/args, PINN config, data file).
# Darcy flow is not included: the paper reports it per parameter value (Tables 6-13), not in Table 5.
PDEBENCH_PDES = {
    "diff-sorp": ("config_diff-sorp.yaml", "config_pinn_diff-sorp.yaml", "1D_diff-sorp_NA_NA.h5"),
    "diff-react": ("config_diff-react.yaml", "config_pinn_diff-react.yaml", "2D_diff-react_NA_NA.h5"),
    "swe": ("config_rdb.yaml", "config_pinn_swe2d.yaml", "2D_rdb_NA_NA.h5"),
}
# Table 5 of arXiv:2210.07182 (RMSE). U-Net is the pushforward variant: Sec. 4 says it is the U-Net baseline.
PDEBENCH_RMSE = {
    "diff-sorp": {"FNO": 5.9e-4, "Unet": 5.8e-2, "PINN": 9.9e-2},
    "diff-react": {"FNO": 8.1e-3, "Unet": 6.1e-2, "PINN": 1.9e-1},
    "swe": {"FNO": 4.5e-3, "Unet": 8.6e-2, "PINN": 1.7e-2},
}
# pdebench/models/config/README.md: overrides that produced the released U-Net / FNO checkpoints.
PDEBENCH_MODEL_ARGS = {
    "FNO": {"model_name": "FNO"},
    "Unet": {"model_name": "Unet", "ar_mode": True, "pushforward": True, "unroll_step": 20},
    "PINN": {"model_name": "PINN"},
}
# The tasks train, so the released checkpoints (DaRUS doi:10.18419/darus-2987) are forbidden, not an input.
PDEBENCH_POLICY = {
    "compute": "slurm", "platform": ONE_A100,
    "inputs": {"DATA_DIR": {"path": f"{CACHE}/pdebench", "holds": ", ".join(f for _, _, f in PDEBENCH_PDES.values())
                            + ", downloaded from DaRUS doi:10.18419/darus-2986 and md5-checked against "
                              "pdebench/data_download/pdebench_data_urls.csv"}},
    "forbidden": "the PDEBench paper's result tables, the pretrained models of DaRUS doi:10.18419/darus-2987, "
                 "or other published PDEBench results",
    # User, 2026-10-19: one A100 per target; the longest target (FNO, 500 epochs) measured at about 16 h.
    "wall_clock_cap": "24 hours",
}
# Each train.py pickles metrics.py's errors to <model_name>.pickle; config.yaml disables hydra's job logging, so
# metrics.py's RMSE log line is not printed, and analyse_result_forward.py (README) turns the pickle into Results.csv.
PDEBENCH_OUTPUTS = {"<model_name>.pt": "the trained model checkpoint train.py saves",
                    "<model_name>.pickle": "the errors train.py pickles after evaluation",
                    "Results.csv": "the CSV pdebench/models/analyse_result_forward.py writes from that pickle"}


def build_pdebench() -> list[str]:
    visible_tasks, gold_tasks = [], {}
    for pde, (config, pinn_config, data_file) in PDEBENCH_PDES.items():
        for model, args in PDEBENCH_MODEL_ARGS.items():
            task_id = f"{pde}-{model.lower()}"
            overrides = " ".join(f"{k}={v}" for k, v in args.items())
            visible_tasks.append({"id": task_id, "task_prompt": (
                f"Train and evaluate with the config pdebench/models/config/args/{pinn_config if model == 'PINN' else config} "
                f"and the overrides `{overrides}` on {data_file}; pdebench/models/metrics.py computes the RMSE."),
                "results": ["RMSE"], "outputs": PDEBENCH_OUTPUTS})
            value = PDEBENCH_RMSE[pde][model]
            gold_tasks[task_id] = paper_contract(
                evaluator="pdebench/models/metrics.py", result="RMSE, float, lower_is_better", value=value,
                rule=f"relative error to the Table 5 RMSE {value:g} at most 0.01 (user choice 2026-10-08: 1%, widen only if larger deviations recur)",
                paper="arXiv:2210.07182", paper_sha256=PDEBENCH_PDF_SHA256, table="Table 5",
                excerpt=f"{PDEBENCH_TABLE_NAME[pde]} RMSE, {model}: {value:g}")
    source_id = "pdebench"
    write(VISIBLE_DIR / f"{source_id}.json", {
        "source_id": source_id,
        "repo": {"path": "../OSS/PDEBench", "commit": PDEBENCH_COMMIT},
        "domain": "PDE surrogate modelling", "language": "Python",
        "support_level": SUPPORT_LEVEL,
        "policy": PDEBENCH_POLICY,
        "env_groups": [{"id": "pdebench-env", "tasks": [t["id"] for t in visible_tasks]}],
        "tasks": visible_tasks,
    })
    write(GOLD_DIR / f"{source_id}.json", {"source_id": source_id, "tasks": gold_tasks})
    return [source_id]


CLIMATELEARN_COMMIT = "b48fb0242acc47e365af86bfbd9dd86e9dcbd6d2"
CLIMATELEARN_PDF_SHA256 = "b9e8e1ccf930d65c52bd6fd7c3a1ce15b11305e3355d8b8c69ce211e4dde71c4"

# Table 10 of arXiv:2307.01909: latitude-weighted T2M RMSE (K) on the normal ERA5 test split, direct forecasting.
# Keys are lead times in hours, the `pred_range` choices of experiments/forecasting/era5_era5_deep_learning.py.
# The same training run also outputs Z500 and T850, which the paper shows only in Figure 2 (no numbers), so they are not scored.
CLIMATELEARN_T2M_RMSE = {
    "resnet": {6: 0.72, 24: 0.94, 72: 1.50, 120: 2.20, 240: 2.78},
    "unet": {6: 0.76, 24: 1.04, 72: 1.65, 120: 2.26, 240: 2.76},
    "vit": {6: 0.78, 24: 1.09, 72: 1.71, 120: 2.38, 240: 2.78},
}


def build_climatelearn() -> list[str]:
    visible_tasks, gold_tasks = [], {}
    for model, by_lead in CLIMATELEARN_T2M_RMSE.items():
        for hours, rmse in by_lead.items():
            task_id = f"forecast-era5-{model}-{hours}h"
            visible_tasks.append({"id": task_id, "task_prompt": (
                f"Run experiments/forecasting/era5_era5_deep_learning.py with model {model}, direct forecasting and pred_range "
                f"{hours} on WeatherBench ERA5 5.625 deg (download utility: src/climate_learn/data/download.py)."),
                "results": ["T2M latitude-weighted RMSE"]})
            gold_tasks[task_id] = paper_contract(
                evaluator="src/climate_learn/metrics/metrics.py LatWeightedRMSE, run by trainer.test in the forecasting script",
                result="T2M latitude-weighted RMSE (K), normal ERA5 test split, float, lower_is_better", value=rmse,
                rule=f"relative error to the Table 10 normal-split T2M RMSE {rmse:g} at most 0.01 (user choice 2026-10-08: 1%, widen only if larger deviations recur)",
                paper="arXiv:2307.01909", paper_sha256=CLIMATELEARN_PDF_SHA256, table="Table 10",
                excerpt=f"T2M, {model}, {hours} h lead, normal split: {rmse:g}")
    source_id = "climatelearn-forecasting"
    write(VISIBLE_DIR / f"{source_id}.json", {
        "source_id": source_id,
        "repo": {"path": "../OSS/climate-learn", "commit": CLIMATELEARN_COMMIT},
        "domain": "weather forecasting", "language": "Python",
        "support_level": SUPPORT_LEVEL,
        "env_groups": [{"id": "climatelearn-env", "tasks": [t["id"] for t in visible_tasks]}],
        "tasks": visible_tasks,
    })
    write(GOLD_DIR / f"{source_id}.json", {"source_id": source_id, "tasks": gold_tasks})
    return [source_id]


OGB_COMMIT = "61e9784ca76edeaa6e259ba0f836099608ff0586"
OGB_PDF_SHA256 = "6fd0295c67e835a2c1d7cf131de1dec708dbf4fae33967b4d6319349349b20de"

# Tables 5, 9, 11 of arXiv:2005.00687: test metric, mean and std over 10 runs. Each example script runs 10 seeds by default
# (`--runs 10`) and prints the same mean and std. Rows left out: MLP and Node2Vec on ogbl-ppa / ogbl-ddi (the MLP script
# needs a separate `node2vec.py` step and the paper does not say which input its MLP row used), Node2Vec on ogbn-proteins,
# and ogbg-ppa (Table 17 not read).
OGB_TASKS = {
    "ogbn-proteins": ("examples/nodeproppred/proteins", "ROC-AUC (%)", "Table 5", {
        "mlp": ("mlp.py", [], 72.04, 0.48), "gcn": ("gnn.py", [], 72.51, 0.35), "sage": ("gnn.py", ["--use_sage"], 77.68, 0.20)}),
    "ogbl-ppa": ("examples/linkproppred/ppa", "Hits@100 (%)", "Table 9", {
        "gcn": ("gnn.py", [], 18.67, 1.32), "sage": ("gnn.py", ["--use_sage"], 16.55, 2.40), "mf": ("mf.py", [], 32.29, 0.94)}),
    "ogbl-ddi": ("examples/linkproppred/ddi", "Hits@20 (%)", "Table 11", {
        "gcn": ("gnn.py", [], 37.07, 5.07), "sage": ("gnn.py", ["--use_sage"], 53.90, 4.74), "mf": ("mf.py", [], 13.68, 4.75)}),
}
# The zips are the files ogb's own download() fetches (urls in ogb/{node,link}proppred/master.csv); processing them into
# the PyG format is the run's work.
OGB_POLICY = {
    "compute": "slurm", "platform": ONE_A100,
    "inputs": {"DATA_DIR": {"path": f"{CACHE}/ogb-biology",
                            "holds": "proteins.zip, ppassoc.zip and ddi.zip, the ogbn-proteins, ogbl-ppa and ogbl-ddi archives "
                                     "from snap.stanford.edu/ogb/data that ogb's dataset classes download"}},
    "forbidden": "the OGB paper's result tables, the OGB leaderboards at ogb.stanford.edu, or other published OGB results",
    # Preflight 2026-10-19 on one A100, scripts' defaults: proteins 2.7-4.3 h per target, ddi under 1 h, ppa 8-70 h (user:
    # the ppa targets are left out of the evaluated task).
    "wall_clock_cap": "8 hours",
}
# The example scripts write no file; the result is the `All runs:` block Logger.print_statistics() prints at the end.
OGB_OUTPUTS = {"run.log": "the example script's stdout and stderr"}


def build_ogb() -> list[str]:
    visible_tasks, gold_tasks = [], {}
    for dataset, (directory, metric, table, models) in OGB_TASKS.items():
        for model, (script, flags, mean, std) in models.items():
            task_id = f"{dataset}-{model}"
            command = " ".join(["python", script, *flags])
            visible_tasks.append({"id": task_id, "task_prompt": f"In {directory}, run `{command}` on {dataset}; the `Final Test` line "
                                                                "under `All runs:` prints the test mean ± std over its runs.",
                                  "results": [f"test {metric} mean", f"test {metric} std"],
                                 "outputs": OGB_OUTPUTS})
            gold_tasks[task_id] = paper_contract(
                evaluator=f"ogb.{'nodeproppred' if dataset.startswith('ogbn') else 'linkproppred'}.Evaluator(name='{dataset}'), called by the example script",
                result=f"test {metric} mean, over the script's 10 runs, float, higher_is_better", value=mean, std=std,
                rule=f"absolute difference to the paper mean {mean:g} at most 2 x the paper std {std:g}, or 1% relative if that is larger (user choice 2026-10-08)",
                paper="arXiv:2005.00687", paper_sha256=OGB_PDF_SHA256, table=table,
                excerpt=f"{dataset}, {model}, test {metric}: {mean:g}±{std:g}")
    source_id = "ogb-biology"
    write(VISIBLE_DIR / f"{source_id}.json", {
        "source_id": source_id,
        "repo": {"path": "../OSS/ogb", "commit": OGB_COMMIT},
        "domain": "biological graphs", "language": "Python",
        "support_level": SUPPORT_LEVEL,
        "policy": OGB_POLICY,
        "env_groups": [{"id": "ogb-env", "tasks": [t["id"] for t in visible_tasks]}],
        "tasks": visible_tasks,
    })
    write(GOLD_DIR / f"{source_id}.json", {"source_id": source_id, "tasks": gold_tasks})
    return [source_id]


ENDPOINTS_REPO = ROOT.parent / "OSS" / "endpoints"
EDGE_AGENTIC_README = "examples/11_Edge_Agentic_Example/README.md"
EDGE_AGENTIC_CAP = "4 hours"


def build_edge_agentic() -> list[str]:
    """MLPerf Inference v6.1 Edge Agentic: the code and config live in mlcommons/endpoints, not in mlcommons/inference."""
    readme = (ENDPOINTS_REPO / EDGE_AGENTIC_README).read_text()
    source_id = "mlperf-edge-agentic-bfcl-v4"
    write(VISIBLE_DIR / f"{source_id}.json", {
        "source_id": source_id,
        "repo": {"path": "../OSS/endpoints", "commit": git_head(ENDPOINTS_REPO), "directory": "examples/11_Edge_Agentic_Example"},
        "domain": "function calling", "language": "Python",
        "support_level": SUPPORT_LEVEL,
        "policy": {
            "compute": "slurm", "platform": ONE_A100,
            "inputs": {"MODEL_FILE": {"path": f"{CACHE}/edge-agentic/bfcl_v4/models/Qwen3.6-27B-Q4_K_M.gguf",
                                      "holds": "the Qwen3.6-27B Q4_K_M GGUF"}},
            "forbidden": MLPERF_FORBIDDEN + ", or the language/edge-agentic README of mlcommons/inference",
            "wall_clock_cap": EDGE_AGENTIC_CAP,
        },
        "env_groups": [{"id": "edge-agentic-env", "tasks": ["edge-agentic"]}],
        "tasks": [{
            "id": "edge-agentic",
            "task_prompt": "Run `inference-endpoint benchmark from-config --config online_edge_full_run.yaml --accuracy-only` (BFCL v4 "
                           "single-turn, about 995 samples) against a llama.cpp `llama-server` serving the Qwen3.6-27B Q4_K_M GGUF in "
                           "MODEL_FILE, at the README's server settings.",
            "results": ["overall_accuracy", "normalized_single_turn_score"],
            "outputs": {"accuracy_results.json": "the scorer's accuracy_results.json from the run's report directory",
                        "config.yaml": "the resolved config.yaml the run wrote to its report directory",
                        "llama-server.log": "the llama-server log"},
        }],
    })
    quote = "| Overall         | 86.23%    | **≥ 83.64%**            |"
    assert quote in readme
    write(GOLD_DIR / f"{source_id}.json", {"source_id": source_id, "authored_in": "reproduce/edge-agentic-bfcl-v4", "tasks": {"edge-agentic": {
        "evaluator_identity": "inference-endpoint benchmark from-config, bfcl_v4 scorer (accuracy/accuracy_results.json)",
        "result_name_type_and_direction": "overall_accuracy and normalized_single_turn_score, float (percent), higher_is_better",
        "reference_value": {"overall_accuracy": 86.23, "normalized_single_turn_score": 87.96},
        "acceptance_rule": "README gate (user choice 2026-10-15): overall_accuracy at least 0.97 x 86.23 = 83.6431 and "
                           "normalized_single_turn_score at least 0.97 x 87.96 = 85.3212, no upper bound",
        "source_path_sha256_excerpt": {"path": EDGE_AGENTIC_README, "sha256": hashlib.sha256(readme.encode()).hexdigest(), "excerpt": quote},
    }}})
    return [source_id]


LAMMPS_REPO = ROOT.parent / "OSS" / "lammps"
LAMMPS_HNS = "examples/reaxff/HNS"
LAMMPS_REF_LOG = f"{LAMMPS_HNS}/log.30Nov23.reaxff.hns.g++.4"
LAMMPS_REF_ROW = "         0   300           -113.27833      437.52125     -111.57687     -1.7014647      27418.867    "
# Step-0 thermo of the repository's own 2x2x2 reference log. `thermo_modify norm yes` makes these per-atom, so they equal the
# 16x8x12 run's step-0 values (checked on a CPU run, 2026-10-19); pressure and volume scale with the system and are not used.
LAMMPS_REFERENCE = {"PotEng": -113.27833, "E_vdwl": -111.57687, "E_coul": -1.7014647}


def build_lammps() -> list[str]:
    log = (LAMMPS_REPO / LAMMPS_REF_LOG).read_text()
    assert LAMMPS_REF_ROW in log
    source_id = "lammps-reaxff-hns"
    write(VISIBLE_DIR / f"{source_id}.json", {
        "source_id": source_id,
        "repo": {"path": "../OSS/lammps", "commit": git_head(LAMMPS_REPO), "directory": LAMMPS_HNS},
        "domain": "molecular dynamics", "language": "C++",
        "support_level": SUPPORT_LEVEL,
        "policy": {
            "compute": "cpu",
            "platform": "CPU only, no GPU: one node of Slurm partition standard.",
            "inputs": {},
            "forbidden": "published LAMMPS benchmark results or logs (including the log.* files under examples/), or an installed "
                         "LAMMPS module or binary (build lmp from the repository)",
            "wall_clock_cap": "30 minutes",
        },
        "env_groups": [{"id": "lammps-env", "tasks": ["hns-16812"]}],
        "tasks": [{
            "id": "hns-16812",
            "task_prompt": "Build `lmp` from the repository with the REAXFF, QEQ and KOKKOS (OpenMP backend, double precision) packages "
                           "and run `lmp -k on t 1 -sf kk -pk kokkos neigh half neigh/qeq full newton on -v x 16 -v y 8 -v z 12 "
                           "-in examples/reaxff/HNS/in.reaxff.hns -nocite` under MPI on one node (466,944 atoms, 100 steps). Report the "
                           "thermo row of step 0.",
            "results": list(LAMMPS_REFERENCE),
            "outputs": {"build.log": "the build's stdout and stderr", "run.log": "the run's stdout and stderr",
                        "log.lammps": "the LAMMPS log (-log)"},
            "log_lines": {"exit status: ": "the run's exit code, as the last line of run.log",
                          "Loop time of ": "the loop time, ranks, steps and atom count"},
        }],
    })
    write(GOLD_DIR / f"{source_id}.json", {"source_id": source_id, "authored_in": "reproduce/lammps-hns-16812", "tasks": {"hns-16812": {
        "evaluator_identity": "none: the thermo output of lmp (step 0 of log.lammps)",
        "result_name_type_and_direction": "PotEng, E_vdwl and E_coul at step 0, float (real units per atom), expected value",
        "reference_value": LAMMPS_REFERENCE,
        "acceptance_rule": "run exits 0 and reports 100 steps with 466944 atoms; each step-0 value within 0.01% relative error of the "
                           "repository's reference log (tolerance proposed by the assistant 2026-10-19, not separately confirmed)",
        "source_path_sha256_excerpt": {"path": LAMMPS_REF_LOG, "sha256": hashlib.sha256(log.encode()).hexdigest(), "excerpt": LAMMPS_REF_ROW.strip()},
    }}})
    return [source_id]


if __name__ == "__main__":
    for source_id in build_mlperf() + build_hecbench() + build_pdebench() + build_climatelearn() + build_ogb() + build_lammps():
        print("wrote", source_id)
