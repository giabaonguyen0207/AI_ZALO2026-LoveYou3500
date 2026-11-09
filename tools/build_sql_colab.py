"""Create a separate Colab notebook for the sealed Qwen SQL SFT release."""
from pathlib import Path
import argparse
import ast
import json


def build(output,source_sha256,source_name,dataset_name):
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(True)})
    def code(text):
        ast.parse(text)
        cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(True)})
    md('''# Qwen3.5-9B · SQL SFT · Colab

Dùng hai ZIP cùng release, chép vào `MyDrive/DataMateSQLSFT/`. Bộ train được tách theo doanh nghiệp; validation là tập phát triển đã được xem khi audit. Đọc README trong ZIP trước khi chạy.

Mặc định **QLoRA, smoke 1 bước trên mẫu train dài nhất rồi pilot 5 bước**. Chọn GPU và High RAM. A100 là lựa chọn thuận tiện cho context 16K; L4 phải vượt smoke/pilot mới biết có vừa bộ nhớ. T4 16GB bị chặn bởi profile này. Chưa có kết quả training GPU từ bước chuẩn bị dữ liệu.

Để chạy toàn bộ sau pilot, đổi `STAGE="full"`, dùng RUN_TAG mới. Model luôn lấy đúng base revision đã pin. Notebook này không sửa notebook inference đang có.''')
    code(f'''from pathlib import Path
METHOD = "qlora"  # "qlora" hoặc "lora"
STAGE = "pilot"   # pilot: 5 steps; full: 1 epoch
RUN_TAG = "qwen35-9b-sql-v1-pilot-01"
DRIVE_ROOT = Path("/content/drive/MyDrive/DataMateSQLSFT")
SOURCE_ZIP = DRIVE_ROOT / {source_name!r}
DATASET_ZIP = DRIVE_ROOT / {dataset_name!r}
EXPECTED_SOURCE_SHA256 = {source_sha256!r}
BASE_DIR = DRIVE_ROOT / "models/qwen35-9b-c202236"
SEED = 42
assert METHOD in {{"qlora","lora"}} and STAGE in {{"pilot","full"}}
assert RUN_TAG and Path(RUN_TAG).name == RUN_TAG
''')
    code('''from google.colab import drive
drive.mount("/content/drive")
import hashlib, json, os, stat, subprocess, sys, zipfile
from pathlib import PurePosixPath
assert SOURCE_ZIP.is_file() and DATASET_ZIP.is_file(), "Chép hai ZIP vào DRIVE_ROOT trước"
with SOURCE_ZIP.open("rb") as handle:
    actual = hashlib.file_digest(handle,"sha256").hexdigest()
assert actual == EXPECTED_SOURCE_SHA256, "Source ZIP sai release"
WORK = Path("/content") / ("datamate-sql-" + actual[:12])
with zipfile.ZipFile(SOURCE_ZIP) as handle:
    assert len(set(handle.namelist())) == len(handle.infolist())
    assert sum(x.file_size for x in handle.infolist()) < 64*1024**2
    for item in handle.infolist():
        path=PurePosixPath(item.filename)
        assert not path.is_absolute() and ".." not in path.parts and "\\\\" not in item.filename and ":" not in item.filename
        assert not stat.S_ISLNK(item.external_attr >> 16)
    WORK.mkdir(parents=True,exist_ok=True)
    handle.extractall(WORK)
if "datamate" in sys.modules:
    assert Path(sys.modules["datamate"].__file__).resolve().is_relative_to(WORK), "Restart runtime: đang giữ source khác trong RAM"
import importlib.metadata as metadata
torch_version=metadata.version("torch")
constraints=WORK/"colab-torch-constraints.txt"
constraints.write_text("torch=="+torch_version+"\\n",encoding="utf-8")
subprocess.run([sys.executable,"-m","pip","install","-q","-c",str(constraints),"-e",str(WORK)+"[train]"],check=True)
sys.path.insert(0,str(WORK/"src"))
print("Source:",actual,"Torch được giữ:",torch_version)
''')
    code('''import torch
from datamate.training.colab import gpu_preflight
from datamate.training.sql_dataset import unpack_sql_archive
from datamate.models.artifacts import stage_model, verify_artifact
from datamate.models.registry import get_model
gpu = gpu_preflight(torch,min_vram_gib=22 if METHOD=="qlora" else 38)
print(json.dumps(gpu,ensure_ascii=False,indent=2))
DATASET = WORK/"sql_dataset"
manifest=unpack_sql_archive(DATASET_ZIP,DATASET,WORK/"configs/qwen35-9b-sql-release.json")
print("Verified examples:",manifest["train_examples"],"train,",manifest["validation_examples"],"validation")
# Online staging is separate from training/inference. No full model is loaded here.
os.environ.pop("HF_HUB_OFFLINE",None)
os.environ.pop("TRANSFORMERS_OFFLINE",None)
if (BASE_DIR/"artifact_manifest.json").is_file():
    verify_artifact(BASE_DIR,"qwen35-9b")
else:
    stage_model("qwen35-9b",BASE_DIR)
print("Pinned base:",get_model("qwen35-9b").to_dict())
OUT = DRIVE_ROOT/"runs"/RUN_TAG
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"environment.json").write_text(json.dumps({"gpu":gpu,"source_sha256":EXPECTED_SOURCE_SHA256,
    "dataset_manifest_sha256":hashlib.sha256((DATASET/"manifest.json").read_bytes()).hexdigest(),
    "base":get_model("qwen35-9b").to_dict(),"method":METHOD,"stage":STAGE},ensure_ascii=False,indent=2),encoding="utf-8")
freeze=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
(OUT/"pip-freeze.txt").write_text(freeze,encoding="utf-8")
''')
    code('''profile=json.loads((WORK/"configs"/("sql-qwen35-9b-"+METHOD+".json")).read_text(encoding="utf-8"))
profile.update(base_dir=str(BASE_DIR),dataset_dir=str(DATASET),seed=SEED,
               output_dir=str(OUT/"adapter"),max_length=manifest["max_length"],
               max_steps=5 if STAGE=="pilot" else -1)
smoke={**profile,"output_dir":str(OUT/"smoke_adapter"),"max_steps":1,
       "gradient_accumulation_steps":1,"smoke_longest_only":True,"save_steps":1}
smoke_path=OUT/"smoke_config.json";smoke_path.write_text(json.dumps(smoke,indent=2),encoding="utf-8")
if (OUT/"smoke_adapter/training_report.json").exists():
    saved=json.loads((OUT/"smoke_adapter/training_report.json").read_text(encoding="utf-8"))
    assert saved["training_config"]==smoke, "Đổi RUN_TAG cho cấu hình mới"
else:
    subprocess.run([sys.executable,"-m","datamate","train","--config",str(smoke_path)],cwd=WORK,check=True)
print("Smoke training đã ghi report. Bộ nhớ GPU của child process đã được giải phóng.")
''')
    code('''config_path=OUT/"training_config.json"
if config_path.exists():
    assert json.loads(config_path.read_text(encoding="utf-8"))==profile, "Cấu hình thay đổi: dùng RUN_TAG mới"
config_path.write_text(json.dumps(profile,indent=2),encoding="utf-8")
assert not (OUT/"adapter").exists(), "RUN_TAG đã có adapter/checkpoint; dùng tag mới hoặc resume có kiểm tra bằng CLI"
subprocess.run([sys.executable,"-m","datamate","train","--config",str(config_path)],cwd=WORK,check=True)
report=json.loads((OUT/"adapter/training_report.json").read_text(encoding="utf-8"))
print(json.dumps({k:report[k] for k in ["train_sequences","validation_sequences","trainable_parameters","peak_cuda_bytes","metrics"]},ensure_ascii=False,indent=2))
print("Adapter và báo cáo đã lưu:",OUT/"adapter")
print("Sau pilot: đổi STAGE='full' và RUN_TAG mới để chạy từ cùng base.")
''')
    md('''Training loss và validation loss không thay cho chất lượng agent. Trước khi chốt adapter, chạy baseline và bản fine-tune trên cùng validation IDs, cùng code/prompt/mode/ngân sách; so PA, sai, abstain, số calls và thời gian. Dataset dùng `mode=single` với auto-execute; không lấy PA self-review cũ làm baseline tương đương. Giữ adapter cùng base revision; merge/seal bằng CLI `datamate merge` trên GPU đủ bộ nhớ rồi mới đưa vào inference offline.''')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'},'cells':cells}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source-sha256',required=True)
    p.add_argument('--source-name',required=True);p.add_argument('--dataset-name',required=True);a=p.parse_args()
    build(a.output,a.source_sha256,a.source_name,a.dataset_name)
