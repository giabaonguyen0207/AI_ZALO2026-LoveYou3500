"""Create a short Colab notebook for interactive agent inference only."""
import json
from pathlib import Path

cells = []
def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)})

md("""# DataMate · Thử agent trên Colab

Notebook này chạy **một câu hỏi mỗi lần**, lưu JSON và hiện dấu vết SQL. Không có bước training.

1. Theo phần D1–D2 của `docs/huong-dan-chay-tung-buoc.md`, đưa `datamate-colab-source.zip` và `train.zip` vào Drive/MyDrive/DataMate.
2. Chọn GPU trong Runtime → Change runtime type.
3. Chạy lần lượt bốn ô code bên dưới. Ở ô cuối, sửa câu hỏi rồi chạy lại để tiếp tục thử, không cần load lại model.

Mặc định dùng Qwen 3.5 4B 4bit. Có thể chọn `qwen35-9b` hoặc `gemma4-12b`; vẫn cần kiểm tra VRAM của GPU thực tế. Notebook chưa được chạy trên Colab trong phiên triển khai.
""")
code('''from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import os, sys, json, zipfile, subprocess, runpy

BASE = Path("/content/drive/MyDrive/DataMate")
SOURCE_ZIP = BASE / "datamate-colab-source.zip"
TRAIN_ZIP = BASE / "train.zip"
WORK = Path("/content/datamate-inference")
SAVE = BASE / "runs/inference-01"
MODEL_KEY = "qwen35-4b"  # "qwen35-9b" or "gemma4-12b"
QUANTIZATION = "4bit"    # "none" only when GPU/RAM is sufficient
MODEL_SOURCE_DIR = BASE / "models" / MODEL_KEY
MODEL_DIR = MODEL_SOURCE_DIR

WORK.mkdir(parents=True, exist_ok=True)
SAVE.mkdir(parents=True, exist_ok=True)
def extract_checked(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        if any(not (destination/name).resolve().is_relative_to(destination) for name in z.namelist()):
            raise ValueError("Unsafe ZIP path")
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
os.environ["HF_HOME"] = str(WORK/".cache/huggingface")
os.environ["TORCH_HOME"] = str(WORK/".cache/torch")
os.environ["PIP_CACHE_DIR"] = str(WORK/".cache/pip")
print("Train:", TRAIN_ROOT, "Model:", MODEL_KEY, "Outputs:", SAVE)
''')
md("## 2. Cài thư viện, kiểm tra GPU\nGiữ bản PyTorch CUDA do Colab cung cấp. Nếu resolver báo xung đột thì dừng và kiểm tra, không tự đổi PyTorch.")
code('''import torch
import importlib, sys
if "WORK" not in globals():
    raise RuntimeError("Chạy ô 1 để giải nén mã nguồn trước khi cài thư viện.")
source_dir = (WORK / "src").resolve()
if not (source_dir / "datamate/__init__.py").is_file():
    raise RuntimeError("Thiếu WORK/src/datamate. Chạy lại ô 1 với đúng gói source.")
constraints = WORK / "colab-constraints.txt"
constraints.write_text("torch==" + torch.__version__ + "\\n")
subprocess.run([sys.executable,"-m","pip","install","-e",".[llm]","-c",str(constraints)],cwd=WORK,check=True)
# Editable .pth paths are normally read when a Python process starts.
# Make the source visible to this already-running notebook kernel as well.
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))
importlib.invalidate_caches()
import datamate
from datamate.config import Settings
print("DataMate import OK:", datamate.__file__)
subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv"],check=True)
assert torch.cuda.is_available(), "Select a CUDA GPU runtime first"
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
''')
md("## 3. Chuẩn bị và load model\nLượt đầu chuẩn bị trên Drive, chép sang /content/models rồi kiểm tra SHA-256 local trước khi load. Các lượt sau dùng lại cache cùng runtime. Nếu đổi model, chạy lại các ô theo thứ tự. Tải xong model vẫn phải fit GPU khi load/inference.")
code((Path(__file__).resolve().parent / 'colab_load_model.py').read_text(encoding='utf-8'))

md("## 4. Đặt câu hỏi, xem SQL\nMỗi lần bấm chạy ô này chỉ hỏi một câu. Đáp án chuẩn trong answers.jsonl không được đưa cho agent. Đúng một câu thử không phải benchmark accuracy.")
code('''from datamate.domain import Question, json_text
from datetime import datetime, timezone

BUSINESS_ID = "biz_101"
AS_OF_DATE = "2026-06-15"
QUESTION = "Dữ liệu bán hàng trong file ban_hang.csv có tất cả bao nhiêu dòng? Đếm COUNT(*) toàn bảng, không lọc ngày hay trạng thái và không loại trùng."
# Then try: "Tổng doanh thu tháng này là bao nhiêu?"
question_id = "interactive_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
run = agent.run(Question(question_id,BUSINESS_ID,QUESTION,AS_OF_DATE))
payload = {"model_key":MODEL_KEY,"business_id":BUSINESS_ID,"as_of_date":AS_OF_DATE,
           "question":QUESTION,**run.to_dict(question_id)}
output = SAVE / (question_id + ".json")
output.write_text(json_text(payload,indent=2),encoding="utf-8")
print(json_text(payload,indent=2))
print("Saved:",output)
''')
md("""Khi muốn chuyển sang fine-tune, dùng `DataMate_Colab_LoRA.ipynb` sau khi đã kiểm tra baseline.
Các checkpoint được khóa trong registry trước cutoff; giữ tổng tham số <=15B. Notebook này dùng train để thử, không dùng public/private test để tạo dữ liệu training.
""")

notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{
    "colab":{"name":"DataMate_Colab_Inference.ipynb"},
    "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"}},"cells":cells}
path = Path(__file__).resolve().parents[1] / "notebooks/DataMate_Colab_Inference.ipynb"
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding="utf-8")
print(path)
