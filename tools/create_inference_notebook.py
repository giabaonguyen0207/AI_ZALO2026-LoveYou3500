"""Build the shared GPU inference notebook; no historical notebook patching."""
import json
from pathlib import Path

SETUP = "from pathlib import Path\nimport os, sys, json, zipfile, stat, subprocess, runpy, time, hashlib\n\nPLATFORM = \"colab\"  # \"colab\", \"kaggle\", \"gpu\"\nMODEL_KEY = \"qwen35-9b\"  # or \"gemma4-12b\"\nOWNER = \"hai\"  # \"thinh\" or your name\nRUN_NAME = time.strftime(\"%Y%m%d-%H%M%S\")\nRUN_LIMIT = None\nIDS_FILE = None  # optional JSON array of selected question IDs\nRESUME = False\nif PLATFORM == \"colab\":\n    from google.colab import drive\n    drive.mount(\"/content/drive\")\n    STORAGE = Path(\"/content/drive/MyDrive/DataMate\")\n    WORK = Path(\"/content/datamate\")\n    SOURCE_ZIP = STORAGE / \"datamate-colab-source.zip\"\n    TRAIN_ROOT = STORAGE / \"data/train\"\n    TRAIN_ZIP = STORAGE / \"train.zip\"\n    MODEL_SOURCE_DIR = STORAGE / \"models\" / MODEL_KEY\nelif PLATFORM == \"kaggle\":\n    WORK = Path(\"/kaggle/working/datamate\")\n    STORAGE = Path(\"/kaggle/working/datamate-results\")\n    SOURCE_ZIP = Path(\"/kaggle/input/datamate-source/datamate-colab-source.zip\")\n    TRAIN_ROOT = Path(\"/kaggle/input/datamate-data/train\")\n    TRAIN_ZIP = None\n    MODEL_SOURCE_DIR = Path(\"/kaggle/input/datamate-model\") / MODEL_KEY\nelif PLATFORM == \"gpu\":\n    WORK = Path(\"/work/datamate\")\n    STORAGE = Path(\"/work/datamate-results\")\n    SOURCE_ZIP = Path(\"/work/inputs/datamate-colab-source.zip\")\n    TRAIN_ROOT = Path(\"/work/data/train\")\n    TRAIN_ZIP = None\n    MODEL_SOURCE_DIR = Path(\"/work/models\") / MODEL_KEY\nelse:\n    raise ValueError(\"PLATFORM must be colab, kaggle or gpu\")\n# Edit these paths to match your uploaded datasets and sealed checkpoint.\nCOPY_MODEL_TO_SCRATCH = PLATFORM == \"colab\"\nSAVE = STORAGE / \"runs\" / OWNER / RUN_NAME\nSAVE.mkdir(parents=True, exist_ok=True)\nif SOURCE_ZIP.is_file():\n    with SOURCE_ZIP.open(\"rb\") as handle:\n        source_hash = hashlib.file_digest(handle, \"sha256\").hexdigest()\n    WORK = WORK / (\"source-\" + source_hash[:12])\n    receipt = WORK / \".source_archive_sha256\"\n    if WORK.exists() and any(WORK.iterdir()) and (not receipt.is_file() or receipt.read_text().strip() != source_hash):\n        raise RuntimeError(\"Source workspace is incomplete or differs from ZIP; choose a new WORK folder\")\n    WORK.mkdir(parents=True, exist_ok=True)\n\n    with zipfile.ZipFile(SOURCE_ZIP) as archive:\n        for entry in archive.infolist():\n            target = (WORK / entry.filename).resolve()\n            if not target.is_relative_to(WORK.resolve()) or stat.S_ISLNK(entry.external_attr >> 16):\n                raise ValueError(\"Unsafe source ZIP path\")\n        archive.extractall(WORK)\n    receipt.write_text(source_hash + \"\\n\", encoding=\"utf-8\")\nelif not (WORK / \"src/datamate/__init__.py\").is_file():\n    raise FileNotFoundError(\"Provide SOURCE_ZIP or an existing repository at WORK\")\nloaded = sys.modules.get(\"datamate\")\nif loaded and not Path(loaded.__file__).resolve().is_relative_to(WORK / \"src\"):\n    raise RuntimeError(\"Another DataMate source is already loaded; restart the kernel\")\nCONFIG_FILE = WORK / \"configs\" / (MODEL_KEY + \".json\")\nif not TRAIN_ROOT.is_dir():\n    if TRAIN_ZIP is None or not TRAIN_ZIP.is_file():\n        raise FileNotFoundError(\"Set TRAIN_ROOT to an extracted split containing questions.jsonl\")\n    helpers = runpy.run_path(str(WORK / \"tools/colab_archives.py\"))\n    TRAIN_ROOT = helpers[\"prepare_train_archive\"](TRAIN_ZIP, WORK, WORK / \"configs/training_sources.json\")\nos.chdir(WORK)\nsys.path.insert(0, str(WORK))\nos.environ.update(HF_HOME=str(WORK / \".cache/huggingface\"), TORCH_HOME=str(WORK / \".cache/torch\"),\n                  PIP_CACHE_DIR=str(WORK / \".cache/pip\"))\nprint(\"Source:\", WORK, \"| data:\", TRAIN_ROOT, \"| results:\", SAVE)\n"
INSTALL = "import torch\nconstraints = WORK / \"colab-constraints.txt\"\nconstraints.write_text(\"torch==\" + torch.__version__ + \"\\n\")\nsubprocess.run([sys.executable,\"-m\",\"pip\",\"install\",\"-e\",\".[llm]\",\"-c\",str(constraints)],cwd=WORK,check=True)\nsubprocess.run([\"nvidia-smi\",\"--query-gpu=name,memory.total,driver_version\",\"--format=csv\"],check=True)\nassert torch.cuda.is_available(), \"Select a CUDA GPU runtime first\"\nprint(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.version.cuda)\n\nimport importlib\nsource_dir = (WORK / \"src\").resolve()\nif not (source_dir / \"datamate/__init__.py\").is_file():\n    raise RuntimeError(\"Gói source thiếu src/datamate. Thay source ZIP mới rồi chạy ô đầu.\")\nif str(source_dir) not in sys.path:\n    sys.path.insert(0, str(source_dir))\nimportlib.invalidate_caches()\nimport datamate\nprint(\"DataMate import OK:\", datamate.__version__, datamate.__file__)\n"
PREPARE = "from datamate.config import Settings\nfrom pathlib import Path\nimport runpy\n\nverify_runtime = runpy.run_path(str(WORK / \"tools/colab_runtime_check.py\"))[\"verify_runtime\"]\nprint(\"Preflight:\", verify_runtime(WORK))\nif not (MODEL_SOURCE_DIR / \"artifact_manifest.json\").is_file():\n    raise FileNotFoundError(\n        \"Prepare the pinned checkpoint first: python -m datamate stage-model --model \"\n        + MODEL_KEY + \" --output \" + str(MODEL_SOURCE_DIR)\n    )\nMODEL_DIR = MODEL_SOURCE_DIR\nif COPY_MODEL_TO_SCRATCH:\n    prepare_local_model = runpy.run_path(str(WORK / \"tools/colab_model_cache.py\"))[\"prepare_local_model\"]\n    MODEL_DIR = prepare_local_model(MODEL_SOURCE_DIR, WORK / \".cache/models\", MODEL_KEY)\nsettings = Settings.load(CONFIG_FILE, model_key=MODEL_KEY, model_dir=str(MODEL_DIR),\n                         split_root=str(TRAIN_ROOT), cache_dir=str(WORK / \".cache/datamate\"),\n                         result_dir=str(SAVE))\nprint(\"Prepared config; the next cell loads one model and runs the shared pipeline.\")\n"
RUN = "from datamate.application import run_batch\nfrom pathlib import Path\nimport json\n\nselected_ids = set(json.loads(Path(IDS_FILE).read_text(encoding=\"utf-8\"))) if IDS_FILE else None\nsummary = run_batch(settings, ids=selected_ids, limit=RUN_LIMIT, resume=RESUME)\nprint(json.dumps(summary, ensure_ascii=False, indent=2))\nprint(\"Submission:\", SAVE / \"submission.jsonl\", \"| logs:\", SAVE / \"logs\")\n"


def cell(kind, text, identifier):
    item = {"cell_type": kind, "id": identifier, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def build(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    cells = [
        cell("markdown", "# DataMate · GPU inference\n\nChọn Colab, Kaggle hoặc GPU thuê trong ô đầu; sửa đường dẫn theo dữ liệu đã upload. "
             "Notebook dùng chung application.run_batch với CLI/Docker, không khóa benchmark. "
             "Chuẩn bị checkpoint đã seal bằng stage-model trước khi inference. "
             "Kaggle lưu output vào /kaggle/working; tải kết quả trước khi kết thúc phiên. Xem docs/run-gpu.md.\n", "gpu-intro"),
        cell("code", SETUP, "gpu-setup"),
        cell("markdown", "## Cài dependency GPU\nGiữ PyTorch/CUDA đã có; xử lý lỗi môi trường trước khi đo baseline.\n", "gpu-install-help"),
        cell("code", INSTALL, "gpu-install"),
        cell("markdown", "## Chuẩn bị cấu hình và checkpoint local\nChưa load model; bước chạy sau tính thời gian từ trước load.\n", "gpu-prepare-help"),
        cell("code", PREPARE, "gpu-prepare"),
        cell("markdown", "## Chạy inference\nBaseline trước rồi bản tích hợp trên cùng bộ đã chọn. Mỗi cấu hình dùng RUN_NAME mới; "
             "chỉ RESUME=True khi source/model/config không đổi. Log được ghi từng câu.\n", "gpu-run-help"),
        cell("code", RUN, "gpu-run"),
    ]
    for item in cells:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), item["id"], "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}, "cells": cells}
    path = root / "notebooks/DataMate_Colab_Inference.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build())
