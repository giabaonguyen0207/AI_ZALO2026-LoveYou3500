"""Generate a portable, source-controlled Colab notebook without notebook deps."""
import json
from pathlib import Path

cells = []
def md(text):
    cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)})
def code(text):
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(keepends=True)})

md("""# DataMate · Colab LoRA / QLoRA

Notebook dùng cùng mã nguồn modular monolith với máy local và Docker. **Chưa có kết quả train/accuracy được tuyên bố.**

1. Chọn GPU trong Runtime → Change runtime type. Kiểm tra tên GPU thực tế.
2. Đưa `datamate-colab-source.zip` và **chỉ gói train** lên Google Drive cá nhân; sửa hai đường dẫn ở ô đầu.
3. Chọn Qwen 3.5 9B hoặc Gemma 4 12B. Notebook khóa revision trong registry, không dùng `main`.
4. Chạy baseline → lấy các trajectory có đáp án đúng trên train → LoRA. Không dùng public/private test để train.

`format` chỉ là bài thử học định dạng, không dạy SQL hay suy luận. Muốn train agent, dùng `trajectories`.
""")
code('''from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import os, subprocess, sys, json, zipfile, runpy

SOURCE_ZIP = Path("/content/drive/MyDrive/DataMate/datamate-colab-source.zip")
TRAIN_ZIP = Path("/content/drive/MyDrive/DataMate/train.zip")
WORK = Path("/content/datamate")
SAVE = Path("/content/drive/MyDrive/DataMate/runs/experiment-01")
MODEL_KEY = "qwen35-9b"  # or "gemma4-12b"
METHOD = "lora"        # "lora" for large VRAM; "qlora" for 4bit base
MAX_LENGTH = 16384
DATA_KIND = "trajectories"  # "format" = format-only pipeline smoke test
COLLECT_BASELINE = True
MAX_STEPS = 50           # first measure 50 steps; use -1 for full epochs
RESUME_CHECKPOINT = None # e.g. str(SAVE / "adapter/checkpoint-50")
assert MODEL_KEY in {"qwen35-9b", "gemma4-12b"}
WORK.mkdir(parents=True, exist_ok=True)
SAVE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(WORK / ".cache/huggingface")
os.environ["TORCH_HOME"] = str(WORK / ".cache/torch")
os.environ["PIP_CACHE_DIR"] = str(WORK / ".cache/pip")
def extract_checked(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("ZIP contains an unsafe path")
        z.extractall(destination)
extract_checked(SOURCE_ZIP, WORK)
archive_helper = WORK / "tools/colab_archives.py"
if not archive_helper.is_file():
    raise RuntimeError("Gói source cũ. Chạy tools/package_colab.py trên máy, thay datamate-colab-source.zip trên Drive rồi chạy lại ô này.")
archives = runpy.run_path(str(archive_helper))
TRAIN_ROOT = archives["prepare_train_archive"](
    TRAIN_ZIP, WORK, WORK / "configs/training_sources.json"
)
os.chdir(WORK)
print("Workspace:", WORK, "Train:", TRAIN_ROOT, "Persistent outputs:", SAVE)
''')
md("## Cài runtime và kiểm tra GPU\nKhông tự thay PyTorch/CUDA của Colab. Nếu không tương thích kiến trúc model, xử lý môi trường trước khi tính ngân sách training.")
code('''import torch
import importlib, sys
if "WORK" not in globals():
    raise RuntimeError("Chạy ô 1 để giải nén mã nguồn trước khi cài thư viện.")
source_dir = (WORK / "src").resolve()
if not (source_dir / "datamate/__init__.py").is_file():
    raise RuntimeError("Thiếu WORK/src/datamate. Chạy lại ô 1 với đúng gói source.")
# Preserve Colab's installed CUDA PyTorch. Fail dependency resolution instead of
# silently replacing that build with an incompatible wheel.
constraints = WORK / "colab-constraints.txt"
constraints.write_text("torch==" + torch.__version__ + "\\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-colab.txt", "-c", str(constraints)], cwd=WORK, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=WORK, check=True)
# Also update the already-running notebook kernel, not only new CLI processes.
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))
importlib.invalidate_caches()
import datamate
from datamate.config import Settings
print("DataMate import OK:", datamate.__file__)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], check=True)
import torch, transformers, peft
print({"torch":torch.__version__, "cuda":torch.version.cuda, "transformers":transformers.__version__, "peft":peft.__version__})
assert torch.cuda.is_available(), "Cần GPU CUDA để LoRA"
assert hasattr(transformers, "AutoModelForImageTextToText")
with (SAVE / "pip-freeze.txt").open("w") as f:
    subprocess.run([sys.executable,"-m","pip","freeze"],stdout=f,check=True)
def cli(*args):
    subprocess.run([sys.executable, "-m", "datamate", *map(str,args)], cwd=WORK, check=True)
''')
md("## Chuẩn bị checkpoint và tách doanh nghiệp\nÔ tải model là bước **online chuẩn bị**, không nằm trong inference dự thi. Các lần inference sau dùng file local và tắt tải tài nguyên.")
code('''MODEL_DIR = SAVE / "models" / MODEL_KEY
if not (MODEL_DIR / "artifact_manifest.json").exists():
    cli("stage-model", "--model", MODEL_KEY, "--output", MODEL_DIR)
else:
    from datamate.models.artifacts import verify_artifact
    verify_artifact(MODEL_DIR, MODEL_KEY)

FORMAT_DIR = SAVE / "format-data"
if not (FORMAT_DIR / "manifest.json").exists():
    cli("prepare-training", "--split-root", TRAIN_ROOT, "--output", FORMAT_DIR,
        "--kind", "format", "--sources", WORK / "configs/training_sources.json")
holdout = json.loads((FORMAT_DIR / "manifest.json").read_text())
print({k:holdout[k] for k in ["train_businesses","validation_businesses","train_examples","validation_examples"]})
''')
md("## Thu trajectory bằng agent\nBaseline không nhận đáp án chuẩn trong prompt. Trình chuẩn bị dataset chỉ giữ log có đáp án cuối khớp nhãn train. Đây là kiểm chứng đáp án cuối, chưa chứng minh mọi bước lập luận đều đúng. Nên kiểm tra một mẫu log train. Các doanh nghiệp validation luôn giữ riêng. Chạy phần này có thể tốn nhiều thời gian GPU; có thể resume batch khi phiên Colab ngắt.")
code('''BASELINE_DIR = SAVE / "baseline-train"
baseline_config = {
    "backend":"transformers", "model_key":MODEL_KEY, "model_dir":str(MODEL_DIR),
    "split_root":str(TRAIN_ROOT), "cache_dir":str(WORK/".cache/datamate"),
    "result_dir":str(BASELINE_DIR), "quantization":"4bit" if METHOD=="qlora" else "none",
    "max_context_tokens":MAX_LENGTH, "max_new_tokens":1024,
    "mode":"single", "max_steps":10, "question_timeout_seconds":90
}
baseline_path = SAVE / "baseline-config.json"
baseline_path.write_text(json.dumps(baseline_config,indent=2))
if DATA_KIND == "trajectories" and COLLECT_BASELINE:
    args = ["run", "--config", baseline_path, "--ids", FORMAT_DIR/"train_ids.json"]
    if (BASELINE_DIR/"submission.jsonl").exists(): args += ["--resume"]
    cli(*args)

DATASET_DIR = FORMAT_DIR
if DATA_KIND == "trajectories":
    DATASET_DIR = SAVE / "trajectory-data"
    if not (DATASET_DIR/"manifest.json").exists():
        cli("prepare-training", "--split-root", TRAIN_ROOT, "--output", DATASET_DIR,
            "--kind", "trajectories", "--logs", BASELINE_DIR/"logs",
            "--sources", WORK/"configs/training_sources.json")
print("Training dataset:", DATASET_DIR)
''')
md("## Chạy thử LoRA / QLoRA\n50 bước đầu để đo thời gian và peak VRAM. Checkpoint lưu thẳng Drive để tiếp tục khi Colab dừng. Training loss không phải accuracy DataMate. Nếu chỉ thu trace trên train businesses, chưa có validation loss; phần đánh giá generation ở dưới dùng holdout đầy đủ.")
code('''train_config = {
    "model_key":MODEL_KEY, "base_dir":str(MODEL_DIR),
    "dataset_dir":str(DATASET_DIR), "output_dir":str(SAVE/"adapter"),
    "method":METHOD, "max_length":MAX_LENGTH, "rank":16, "alpha":32,
    "epochs":2, "max_steps":MAX_STEPS, "batch_size":1,
    "gradient_accumulation_steps":8, "learning_rate":0.0001,
    "save_steps":50, "seed":42, "cache_dir":str(WORK/".cache")
}
training_path=SAVE/"training-config.json"
training_path.write_text(json.dumps(train_config,indent=2))
args=["train","--config",training_path]
if RESUME_CHECKPOINT: args += ["--resume",RESUME_CHECKPOINT]
cli(*args)
report=json.loads((SAVE/"adapter/training_report.json").read_text())
print({k:report[k] for k in ["metrics","tokens_per_epoch","peak_cuda_bytes","skipped_overlength_train"]})
''')
md("## Merge và đánh giá holdout\nMerge bỏ phần LoRA bổ sung khỏi inventory triển khai. Dùng cùng câu hỏi giữ riêng để so base và fine-tuned. Nếu muốn đo Gemma assistant, tải checkpoint assistant đã pin và dùng preset speculative riêng; không gộp Qwen 9B + Gemma 12B vì vượt 15B.")
code('''MERGED_DIR=SAVE/"merged-model"
if not (MERGED_DIR/"artifact_manifest.json").exists():
    cli("merge","--model",MODEL_KEY,"--base",MODEL_DIR,"--adapter",SAVE/"adapter","--output",MERGED_DIR)
configs=[]
for label,model_path in [("base",MODEL_DIR),("finetuned",MERGED_DIR)]:
    cfg={**baseline_config,"model_dir":str(model_path),"result_dir":str(SAVE/f"eval-{label}")}
    path=SAVE/f"eval-{label}.json"
    path.write_text(json.dumps(cfg,indent=2))
    configs.append(path)
cli("compare","--configs",*configs,"--ids",FORMAT_DIR/"validation_ids.json",
    "--references",TRAIN_ROOT/"answers.jsonl","--output",SAVE/"comparison")
print((SAVE/"comparison/comparison.json").read_text())
''')
md("""## Mang về ổ E: và đóng gói

- Tải `merged-model/`, `training_report.json`, `pip-freeze.txt`, `comparison/` về dự án ở ổ E:.
- Đổi `model_dir` sang thư mục merged model; không tải checkpoint ở inference.
- Giữ nguyên manifest và hash. Chạy lại trên Docker `--network none`, mount `/data:ro`.
- Dockerfile dự thi cần base image chính thức; workspace chưa có guideline đó. Không coi Docker dev smoke test là chứng nhận tương thích RTX 5090 hay tốc độ 6 giờ.
""")

notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"colab":{"name":"DataMate_Colab_LoRA.ipynb"},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"cells":cells}
path=Path(__file__).resolve().parents[1]/"notebooks/DataMate_Colab_LoRA.ipynb"
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding="utf-8")
print(path)
