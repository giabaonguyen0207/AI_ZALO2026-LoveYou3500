"""Build the maintained GPU notebook from the shared Settings and batch pipeline."""
import json
from pathlib import Path

SETUP = '''from pathlib import Path
import os, sys, json, zipfile, stat, subprocess, runpy, time, hashlib

PLATFORM = os.environ.get("DATAMATE_PLATFORM", "colab")  # colab | kaggle | gpu
MODEL_KEY = os.environ.get("DATAMATE_MODEL_KEY", "qwen35-9b")
BACKEND = None  # None: config/env; "torch" or "vllm" to override
RAG_MODE = None  # None: config/env; "none", "dense", "dense_rerank"
OWNER = "hai"  # thinh or your name
RUN_NAME = time.strftime("%Y%m%d-%H%M%S")
RUN_LIMIT = None
IDS_FILE = None  # JSON array of benchmark question IDs
RESUME = False
if PLATFORM == "colab":
    from google.colab import drive
    drive.mount("/content/drive")
    STORAGE = Path("/content/drive/MyDrive/DataMate")
    WORK = Path("/content/datamate")
    SOURCE_ZIP = STORAGE / "datamate-colab-source.zip"
    TRAIN_ROOT = STORAGE / "data/train"
    TRAIN_ZIP = STORAGE / "train.zip"
    MODEL_SOURCE_DIR = STORAGE / "models" / MODEL_KEY
elif PLATFORM == "kaggle":
    WORK = Path("/kaggle/working/datamate")
    STORAGE = Path("/kaggle/working/datamate-results")
    SOURCE_ZIP = Path("/kaggle/input/datamate-source/datamate-colab-source.zip")
    TRAIN_ROOT = Path("/kaggle/input/datamate-data/train")
    TRAIN_ZIP = None
    MODEL_SOURCE_DIR = Path("/kaggle/input/datamate-model") / MODEL_KEY
elif PLATFORM == "gpu":
    WORK = Path("/work/datamate")
    STORAGE = Path("/work/datamate-results")
    SOURCE_ZIP = Path("/work/inputs/datamate-colab-source.zip")
    TRAIN_ROOT = Path("/work/data/train")
    TRAIN_ZIP = None
    MODEL_SOURCE_DIR = Path("/work/models") / MODEL_KEY
else:
    raise ValueError("PLATFORM must be colab, kaggle or gpu")
WORK = Path(os.environ.get("DATAMATE_WORK_DIR", str(WORK)))
SOURCE_ZIP = Path(os.environ.get("DATAMATE_SOURCE_ZIP", str(SOURCE_ZIP)))
TRAIN_ROOT = Path(os.environ.get("DATAMATE_SPLIT_ROOT", str(TRAIN_ROOT)))
MODEL_SOURCE_DIR = Path(os.environ.get("DATAMATE_MODEL_DIR", str(MODEL_SOURCE_DIR)))
EMBEDDING_SOURCE_DIR = Path(os.environ.get("DATAMATE_RAG_EMBEDDING_DIR", str(MODEL_SOURCE_DIR.parent / "vietnamese-embedding")))
RERANKER_SOURCE_DIR = Path(os.environ.get("DATAMATE_RAG_RERANKER_DIR", str(MODEL_SOURCE_DIR.parent / "vietnamese-reranker")))
COPY_MODEL_TO_SCRATCH = PLATFORM == "colab"
SAVE = Path(os.environ.get("DATAMATE_RESULT_DIR", str(STORAGE / "runs" / OWNER / RUN_NAME)))
SAVE.mkdir(parents=True, exist_ok=True)
if SOURCE_ZIP.is_file():
    with SOURCE_ZIP.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    WORK = WORK / ("source-" + source_hash[:12])
    receipt = WORK / ".source_archive_sha256"
    if WORK.exists() and any(WORK.iterdir()) and (not receipt.is_file() or receipt.read_text().strip() != source_hash):
        raise RuntimeError("Source workspace differs from ZIP; choose a new WORK folder")
    WORK.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        for entry in archive.infolist():
            target = (WORK / entry.filename).resolve()
            if not target.is_relative_to(WORK.resolve()) or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Unsafe source ZIP path")
        archive.extractall(WORK)
    receipt.write_text(source_hash + "\\n", encoding="utf-8")
elif not (WORK / "src/datamate/__init__.py").is_file():
    raise FileNotFoundError("Provide SOURCE_ZIP or an existing repository at WORK")
loaded = sys.modules.get("datamate")
if loaded and not Path(loaded.__file__).resolve().is_relative_to(WORK / "src"):
    raise RuntimeError("Another DataMate source is loaded; restart the kernel")
CONFIG_FILE = Path(os.environ["DATAMATE_CONFIG"]) if os.environ.get("DATAMATE_CONFIG") else WORK / "configs" / (MODEL_KEY + ".json")
if not TRAIN_ROOT.is_dir():
    if TRAIN_ZIP is None or not TRAIN_ZIP.is_file():
        raise FileNotFoundError("Set TRAIN_ROOT to an extracted split containing questions.jsonl")
    helpers = runpy.run_path(str(WORK / "tools/colab_archives.py"))
    TRAIN_ROOT = helpers["prepare_train_archive"](TRAIN_ZIP, WORK, WORK / "configs/training_sources.json")
os.chdir(WORK)
sys.path.insert(0, str(WORK))
os.environ.update(HF_HOME=str(WORK / ".cache/huggingface"), TORCH_HOME=str(WORK / ".cache/torch"),
                  PIP_CACHE_DIR=str(WORK / ".cache/pip"))
print("Source:", WORK, "| data:", TRAIN_ROOT, "| results:", SAVE)
'''

INSTALL = '''import importlib
from importlib.metadata import version, PackageNotFoundError
source_dir = (WORK / "src").resolve()
if not (source_dir / "datamate/__init__.py").is_file():
    raise RuntimeError("Source ZIP lacks src/datamate; upload a complete package")
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))
importlib.invalidate_caches()
from datamate.config import Settings
from datamate.models.registry import inference_extras
overrides = {"backend":globals().get("BACKEND")}
if globals().get("RAG_MODE") is not None:
    overrides["rag"] = {"mode":RAG_MODE}
selected = Settings.load(globals().get("CONFIG_FILE"), **overrides)
extras = inference_extras(selected)
def installed(name):
    try: return version(name)
    except PackageNotFoundError: return None
loaded_versions = {name:installed(name) for name in ("torch","transformers","tokenizers","numpy","vllm") if name in sys.modules}
install = [sys.executable,"-m","pip","install","-e",".[" + extras + "]"]
if selected.backend != "vllm":
    import torch
    constraints = WORK / "colab-constraints.txt"
    constraints.write_text("torch==" + torch.__version__ + "\\n", encoding="utf-8")
    install += ["-c",str(constraints)]
subprocess.run(install,cwd=WORK,check=True)
changed = [name for name,before in loaded_versions.items() if installed(name) != before]
if changed:
    raise RuntimeError("Packages changed while loaded: " + str(changed) + "; restart kernel and rerun before inference")
if globals().get("CHECK_CUDA",True):
    import torch
    subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv"],check=True)
    assert torch.cuda.is_available(), "Select a CUDA GPU runtime first"
    print("PyTorch:",torch.__version__,"CUDA:",torch.version.cuda)
import datamate
print("DataMate import OK:", datamate.__version__, datamate.__file__, "extras:",extras)
'''

PREPARE = '''from datamate.config import Settings
from datamate.models.registry import model_paths, relocate_models, get_model
from datamate.models.artifacts import verify_artifact
from pathlib import Path
import runpy

verify_runtime = runpy.run_path(str(WORK / "tools/colab_runtime_check.py"))["verify_runtime"]
print("Preflight:", verify_runtime(WORK))
overrides = {"backend":globals().get("BACKEND")}
if globals().get("RAG_MODE") is not None:
    overrides["rag"] = {"mode":RAG_MODE}
settings = Settings.load(CONFIG_FILE, model_key=MODEL_KEY, model_dir=str(MODEL_SOURCE_DIR),
    split_root=str(TRAIN_ROOT), cache_dir=str(WORK / ".cache/datamate"), result_dir=str(SAVE), **overrides)
if globals().get("EMBEDDING_SOURCE_DIR") is not None:
    settings.rag.embedding_dir = str(EMBEDDING_SOURCE_DIR)
if globals().get("RERANKER_SOURCE_DIR") is not None:
    settings.rag.reranker_dir = str(RERANKER_SOURCE_DIR)
paths = {}
for key,source in model_paths(settings).items():
    source = Path(source)
    if not (source / "artifact_manifest.json").is_file():
        raise FileNotFoundError("Stage the sealed checkpoint first: python -m datamate stage-model --model " + key + " --output " + str(source))
    verify_artifact(source,key)
    if COPY_MODEL_TO_SCRATCH:
        prepare_local_model = runpy.run_path(str(WORK / "tools/colab_model_cache.py"))["prepare_local_model"]
        source = prepare_local_model(source,WORK / ".cache/models",key)
    paths[key] = source
relocate_models(settings,paths)
settings.validate()
print("Prepared all active checkpoints; no model loaded yet.")
print("Backend:",settings.backend,"RAG:",settings.rag.mode)
'''

RUN = '''from datamate.execution import run_isolated
from pathlib import Path
import json

selected_ids = set(json.loads(Path(IDS_FILE).read_text(encoding="utf-8"))) if IDS_FILE else None
summary = run_isolated(settings, ids=selected_ids, limit=RUN_LIMIT, resume=RESUME)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Submission:", SAVE / "submission.jsonl", "| logs:", SAVE / "logs")
'''


def cell(kind, text, identifier):
    item = {"cell_type":kind,"id":identifier,"metadata":{},"source":text.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None,outputs=[])
    return item


def build(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    cells = [
        cell("markdown","# DataMate · GPU inference\n\nChọn backend Torch/vLLM và RAG độc lập trong ô đầu. "
             "Dùng application.run_batch chung với CLI/Docker, có tiến trình giám sát deadline. "
             "Chuẩn bị tất cả checkpoint đã seal trước; xem docs/run-gpu.md.\n","gpu-intro"),
        cell("code",SETUP,"gpu-setup"),
        cell("markdown","## Cài dependency GPU\nExtras lấy từ cùng Settings như inference. Nếu pip đổi gói đã import, restart kernel trước khi đo.\n","gpu-install-help"),
        cell("code",INSTALL,"gpu-install"),
        cell("markdown","## Chuẩn bị checkpoint local\nKiểm tra/copy toàn bộ generator, embedding, reranker đang bật. Chưa load model.\n","gpu-prepare-help"),
        cell("code",PREPARE,"gpu-prepare"),
        cell("markdown","## Chạy inference\nBaseline trước, rồi tích hợp trên cùng tập. Mỗi lượt dùng RUN_NAME mới. "
             "So sánh tự động ba cấu hình bằng tools/compare_system.py; gold chỉ dùng sau inference.\n","gpu-run-help"),
        cell("code",RUN,"gpu-run"),
    ]
    for item in cells:
        if item['cell_type']=='code':
            compile(''.join(item['source']),item['id'],'exec')
    notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"accelerator":"GPU"},"cells":cells}
    path = root / 'notebooks/DataMate_Colab_Inference.ipynb'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return path


if __name__=='__main__':
    print(build())
