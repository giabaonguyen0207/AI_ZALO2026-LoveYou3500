"""Build a dedicated Colab notebook and sealed input kit for the frozen 200 questions."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


SETUP = '''from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import hashlib, importlib, json, os, runpy, shutil, stat, subprocess, sys, time, uuid, zipfile

BASE = Path("/content/drive/MyDrive/DataMate")  # Sửa nếu Drive của bạn dùng thư mục khác.
KIT_ZIP = BASE / "datamate-team200-colab-kit.zip"
PARTICIPANT = "toi"                # Tên dùng trong file nộp nhóm.
MODEL_KEY = "qwen35-9b"             # Hoặc "gemma4-12b", "qwen35-upscaled14b-v2", "qwen35-4b".
QUANTIZATION = "none"              # G4 96 GB: none; GPU ít VRAM có thể chọn 4bit.
MODE = "self_review"               # Giữ một cấu hình cho cả lượt chạy.
RUN_NAME = f"{PARTICIPANT}-{MODEL_KEY}-team200-rollback-pipeline6-Nov8th-01"
AUTO_DISCONNECT = True
RUN_LIMIT = None                    # None = đủ 200; chỉ đặt 5 nếu muốn chạy thử rồi resume.
MAX_RUN_MINUTES = 180               # Ngân sách MỖI phiên batch, không gồm chuẩn bị/load.
COMPUTE_UNITS_USED = None           # Chỉ điền số đo thật; không suy từ 10 unit/giờ.
SAVE = BASE / "runs/team200" / RUN_NAME
MODEL_SOURCE_DIR = BASE / "models" / MODEL_KEY
if AUTO_DISCONNECT and not SAVE.resolve().is_relative_to(Path("/content/drive").resolve()):
    raise RuntimeError("AUTO_DISCONNECT cần lưu kết quả trên Drive; sửa BASE hoặc tắt tự ngắt.")

EXPECTED = __EXPECTED_ARCHIVES__
if any(name == "datamate" or name.startswith("datamate.") for name in sys.modules):
    raise RuntimeError("Kernel đã import DataMate. Restart session/kernel rồi chạy từ ô đầu; giữ cache /content/models. Muốn resume cùng agent chỉ chạy lại ô batch cuối.")
if not KIT_ZIP.is_file():
    raise FileNotFoundError(f"Đưa datamate-team200-colab-kit.zip lên Drive tại {KIT_ZIP}")

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def extract_checked(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP lỗi CRC")
        seen = set()
        for info in archive.infolist():
            name = info.filename.replace("\\\\", "/")
            target = (destination / name).resolve()
            if (name in seen or not target.is_relative_to(destination)
                    or ":" in name or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"ZIP path không hợp lệ: {name}")
            seen.add(name)
            if not info.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)

RUNTIME_ROOT = Path("/content/datamate-team200")
INPUTS = RUNTIME_ROOT / "inputs"
INPUTS.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(KIT_ZIP) as archive:
    for name, expected in EXPECTED.items():
        target = INPUTS / name
        with archive.open(name) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
        if digest(target) != expected:
            raise ValueError(f"Gói sai phiên bản/khác hash: {name}")
SOURCE_SHA256 = EXPECTED["datamate-team200-source.zip"]
WORK = RUNTIME_ROOT / ("source-" + SOURCE_SHA256[:12])
extract_checked(INPUTS / "datamate-team200-source.zip", WORK)
BENCHMARK_FOLDER = RUNTIME_ROOT / "benchmark-v2"
extract_checked(INPUTS / "participants.zip", BENCHMARK_FOLDER)
PARTICIPANT_ROOT = BENCHMARK_FOLDER / "participant"
TRAIN_ROOT = PARTICIPANT_ROOT / "dataset"  # Chỉ là tên biến; không có answers.jsonl.
REFERENCE_ZIP = INPUTS / "coordinator-private.zip"  # Chỉ mở sau vòng inference để review.
check = runpy.run_path(str(PARTICIPANT_ROOT / "benchmark.py"))
benchmark, questions = check["verify"](PARTICIPANT_ROOT)
assert benchmark["benchmark_id"] == "datamate-train200-seed4000930841"
assert benchmark["questions_sha256"] == "7ae2079219d0a79ca1dd66952610ceb091e36c0ecf65359b11dac8263c67b782"
assert len(questions) == 200 and not (TRAIN_ROOT / "answers.jsonl").exists()
os.chdir(WORK)
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "src"))
os.environ.update(HF_HOME=str(WORK / ".cache/huggingface"), TORCH_HOME=str(WORK / ".cache/torch"),
                  PIP_CACHE_DIR=str(WORK / ".cache/pip"), TOKENIZERS_PARALLELISM="false")
print("Đúng 200 câu, seed 4000930841; không bốc lại mẫu.")
print("Model:", MODEL_KEY, "| mode:", MODE, "| kết quả:", SAVE)
'''

INSTALL = '''import torch
constraints = WORK / "colab-constraints.txt"
constraints.write_text("torch==" + torch.__version__ + "\\n")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[llm]", "-c", str(constraints)], cwd=WORK, check=True)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], check=True)
if not torch.cuda.is_available():
    raise RuntimeError("Chọn GPU runtime trước khi nạp model")
importlib.invalidate_caches()
verify_runtime = runpy.run_path(str(WORK / "tools/colab_runtime_check.py"))["verify_runtime"]
print("Preflight:", verify_runtime(WORK))
'''

LOAD = '''from datamate.config import Settings
from datamate.application import create_agent
verify_runtime(WORK)
if "agent" in globals():
    raise RuntimeError("Model đã load. Chạy lại ô batch để resume; không load chồng hai model.")
# Đây là bước chuẩn bị online, tách khỏi thời gian load/inference benchmark.
if not (MODEL_SOURCE_DIR / "artifact_manifest.json").is_file():
    stage_env = {**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
    subprocess.run([sys.executable, "-m", "datamate", "stage-model", "--model", MODEL_KEY,
                    "--output", str(MODEL_SOURCE_DIR)], cwd=WORK, env=stage_env, check=True)
LOAD_SESSION_ID = uuid.uuid4().hex
load_started = time.monotonic()
prepare_local_model = runpy.run_path(str(WORK / "tools/colab_model_cache.py"))["prepare_local_model"]
MODEL_DIR = prepare_local_model(MODEL_SOURCE_DIR, Path("/content/models"), MODEL_KEY)
settings = Settings(model_key=MODEL_KEY, model_dir=str(MODEL_DIR), mode=MODE,
                    quantization=QUANTIZATION, split_root=str(TRAIN_ROOT),
                    cache_dir=str(WORK / ".cache/datamate-team200"), result_dir=str(SAVE),
                    max_context_tokens=16384, max_new_tokens=768, max_steps=6,
                    question_timeout_seconds=90, schema_mode="code_first", adaptive_preparation=False,
                    require_plan=True, auto_execute_plan=True,
                    schema_repair_timeout_seconds=20, schema_repair_max_new_tokens=512)
agent = create_agent(settings)
LOAD_SECONDS = round(time.monotonic() - load_started, 4)
print("Loaded:", MODEL_KEY, "| pipeline:", agent.pipeline_version, "| load:", LOAD_SECONDS, "giây")
print("Generation tokens:", getattr(agent.backend, "generation_tokens", None))
'''

BATCH = '''runner = runpy.run_path(str(WORK / "tools/infer_team200.py"))
# Biên nhận riêng cho lần này, không dùng lại ZIP cũ nếu gặp lỗi trước khi lưu.
INVOCATION_ID = uuid.uuid4().hex
try:
    summary, group_zip, diagnostic_zip = runner["run_team200"](
        agent, PARTICIPANT_ROOT, SAVE, REFERENCE_ZIP,
        participant=PARTICIPANT, load_seconds=LOAD_SECONDS, load_session_id=LOAD_SESSION_ID,
        source_archive_sha256=SOURCE_SHA256, limit=RUN_LIMIT,
        wall_budget_seconds=MAX_RUN_MINUTES * 60, units_used=COMPUTE_UNITS_USED,
        invocation_id=INVOCATION_ID,
    )
finally:
    receipt_path = SAVE / "report_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
    if receipt.get("invocation_id") == INVOCATION_ID:
        print("Đã lưu hai ZIP trên Drive:", SAVE, flush=True)
        runner["finish_and_disconnect"](SAVE, enabled=AUTO_DISCONNECT)
    else:
        print("Chưa xác nhận có báo cáo mới đầy đủ; giữ runtime để bạn kiểm tra lỗi.", flush=True)
'''


def cell(kind, source, name):
    result = {"cell_type": kind, "metadata": {}, "id": name, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(root):
    root = Path(root).resolve()
    release = root / "outputs/team-benchmark-train200-2026-11-07-v2"
    frozen = json.loads((release / "release_verification.json").read_text(encoding="utf-8"))
    for entry in frozen["packages"]:
        with (release / entry["file"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError("Benchmark v2 đã thay đổi: " + entry["file"])
    output = root / "outputs/team200-colab"
    output.mkdir(parents=True, exist_ok=True)
    source = output / "datamate-team200-source.zip"
    helpers = ["infer_team200.py", "diagnose_train.py", "colab_model_cache.py", "colab_runtime_check.py"]
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((root / "src").rglob("*.py")):
            archive.write(path, path.relative_to(root).as_posix())
        # Runtime assets are required by FastAPI imports only when serving;
        # include them as the same source distribution without any datasets.
        for path in sorted((root / "src/datamate/web").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
        for name in helpers:
            archive.write(root / "tools" / name, "tools/" + name)
        archive.writestr("tools/__init__.py", "\"\"\"Notebook runtime helpers.\"\"\"\n")
        for name in ("pyproject.toml", "requirements.txt"):
            archive.write(root / name, name)
    packages = {"datamate-team200-source.zip": source, "participants.zip": release / "participants.zip",
                "coordinator-private.zip": release / "coordinator-private.zip"}
    expected = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in packages.items()}
    description = """# DataMate · Colab inference đúng 200 câu benchmark nhóm

Mẫu cố định: **seed 4000930841**, 200 câu TRAIN, 28 doanh nghiệp. Notebook này
chạy agent hiện có của bạn; không thay cách chấm hay yêu cầu của các đồng đội.

1. Upload notebook này lên Colab. Đưa **datamate-team200-colab-kit.zip** vào
   **MyDrive/DataMate/**. Model đã seal ở MyDrive/DataMate/models/ được dùng lại.
2. Chọn GPU G4, sửa tên/model/đường dẫn ở ô đầu, rồi **Run all**.
3. Giữ RUN_LIMIT=None để chạy đủ 200; AUTO_DISCONNECT=True để lưu ZIP,
   đồng bộ Drive rồi tự ngắt runtime. Mặc định Qwen 9B/self_review như pipeline
   6 đã khôi phục; có thể chọn Gemma 12B hoặc checkpoint đã hỗ trợ ở ô đầu.
   Preflight phải hiện pipeline 6/backend 3. Dùng tên lượt rollback mới,
   không resume vào kết quả v7 hoặc lượt v6 đã chạy trước đó.

Kết quả ở MyDrive/DataMate/runs/team200/<RUN_NAME>/:
- **group_submission.zip**: chỉ submission.jsonl theo schema cuộc thi và
  run_metadata.json sáu trường. Đây là ZIP gửi nhóm.
- **diagnostic_bundle.zip**: review.csv/jsonl/md, summary/runtime/manifest,
  prompt/raw output/SQL/tool trong logs/*.json và logs/*.md. Có nhãn chuẩn,
  giữ riêng để phân tích sau khi khóa kết quả.

PA trên review có mẫu số 200 kể cả nếu dừng sớm. Nhãn chỉ được đọc sau vòng
inference để review, không đưa cho agent. Thư viện/model chỉ tải ở bước chuẩn
bị; inference dùng checkpoint local và chế độ offline. Đây là benchmark nội
bộ TRAIN, không phải holdout độc lập.

Resume: giữ cùng RUN_NAME, model, code, config và loại GPU/runtime. Nếu vẫn
cùng kernel chỉ chạy lại ô batch; kernel mới thì Run all. Log đã ghi được bỏ
qua, không chọn/ghép đáp án. Khác nguồn/cấu hình/runtime bị chặn. Nếu máy bị
tắt đột ngột không đo đủ thời gian, metadata giữ timing null thay vì bịa số.
Unit giữ null khi chưa đo thực tế; không dùng 10 unit/giờ làm unit đã tiêu.
"""
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "colab": {"name": "DataMate_Colab_Team200.ipynb"}, "accelerator": "GPU"},
                "cells": [cell("markdown", description, "team200-intro"),
                          cell("code", SETUP.replace("__EXPECTED_ARCHIVES__", json.dumps(expected, indent=2)), "team200-setup"),
                          cell("markdown", "## Cài dependency và kiểm tra GPU\nGiữ PyTorch CUDA có sẵn của Colab; preflight chạy trước khi nạp trọng số.\n", "team200-install-help"),
                          cell("code", INSTALL, "team200-install"),
                          cell("markdown", "## Chuẩn bị checkpoint và nạp model\nCheckpoint/revision được stage-model kiểm tra và seal; không tự dùng main.\n", "team200-load-help"),
                          cell("code", LOAD, "team200-load"),
                          cell("markdown", "## Chạy 200 câu, lưu review/log và tự ngắt\nZIP được kiểm tra nội dung rồi đồng bộ Drive trước khi kết thúc runtime. Lượt partial cũng được lưu để resume.\n", "team200-run-help"),
                          cell("code", BATCH, "team200-run")]}
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), item["id"], "exec")
    path = root / "notebooks/DataMate_Colab_Team200.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(path, output / path.name)
    (output / "README_COLAB_200.md").write_text(description, encoding="utf-8")
    kit = output / "datamate-team200-colab-kit.zip"
    with zipfile.ZipFile(kit, "w", zipfile.ZIP_STORED) as archive:
        for name, package in packages.items():
            archive.write(package, name)
        archive.write(path, path.name)
        archive.write(output / "README_COLAB_200.md", "README_COLAB_200.md")
    with zipfile.ZipFile(kit) as archive:
        if archive.testzip() is not None:
            raise ValueError("KIT ZIP lỗi CRC")
        for name, digest in expected.items():
            with archive.open(name) as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                    raise ValueError("KIT khác hash: " + name)
    report = {"notebook": str(path), "kit": str(kit), "kit_bytes": kit.stat().st_size,
              "kit_sha256": hashlib.sha256(kit.read_bytes()).hexdigest(), "inner_sha256": expected,
              "question_seed": 4000930841, "question_count": 200,
              "pipeline_version": 6, "source_package_has_weights": False,
              "model_inference_performed": False}
    (output / "build_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    build(Path(__file__).resolve().parents[1])
