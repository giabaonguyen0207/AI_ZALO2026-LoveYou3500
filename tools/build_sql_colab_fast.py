"""Colab SQL training with verified FLA kernels and persistent automatic resume."""
from pathlib import Path
import ast
import json

try:
    from .build_sql_colab import build as build_base
except ImportError:
    from build_sql_colab import build as build_base


def build(output, source_sha256, source_name, dataset_name):
    build_base(output,source_sha256,source_name,dataset_name)
    notebook=json.loads(Path(output).read_text(encoding='utf-8'))
    cells=notebook['cells']
    logger=next(c for c in cells if c['cell_type']=='code' and ''.join(c['source']).startswith('def run_logged'))
    cells.remove(logger)
    cells.insert(2,logger)  # Available even for installation failures in VS Code.
    for cell in cells:
        source=''.join(cell['source'])
        if cell['cell_type']=='markdown' and source.startswith('# Qwen'):
            source+='\n\n## v1.3 — kernels và resume\nGiữ nguyên data/context/rank/batch/accumulation. FLA 0.6.0 chạy cả DeltaNet và conv Triton; probe forward/backward trước khi load 9B. Giữ Torch/Triton đang có. Mặc định pilot 5 bước; chưa có số đo speedup mới. `RESUME="auto"` dùng checkpoint hoàn tất mới nhất, bỏ qua bản ghi dở. Giữ nguyên RUN_TAG và cấu hình khi nối lại lượt đang train. Full dùng tag mới. Lưu sau bước 1, mỗi 5 bước hoặc ngưỡng 10 phút tại ranh giới bước, giữ 3 bản; evaluation mỗi 50 bước. Phần sau checkpoint gần nhất vẫn có thể phải tính lại. Chỉ chạy một server cho một RUN_TAG.\n'
        elif source.startswith('from pathlib import Path\nMETHOD'):
            source=source.replace('qwen35-9b-sql-v1-pilot-01','qwen35-9b-sql-v1.3-pilot-01')
            source+='\nRESUME = "auto"  # None: lượt mới; auto: tiếp từ checkpoint đầy đủ nếu có\n'
        elif source.startswith('from google.colab import drive'):
            source=source.replace('constraints.write_text("torch=="+torch_version+"\\n",encoding="utf-8")',
                '''pins=["torch=="+torch_version]
try: pins.append("triton=="+metadata.version("triton"))
except metadata.PackageNotFoundError: pass
constraints.write_text("\\n".join(pins)+"\\n",encoding="utf-8")''')
            source=source.replace('subprocess.run([sys.executable,"-m","pip","install","-q","-c",str(constraints),"-e",str(WORK)+"[train]"],check=True)',
                '''from uuid import uuid4
run_logged([sys.executable,"-u","-m","pip","install","-c",str(constraints),
            "-r",str(WORK/"configs/sql-qwen35-fast-requirements.txt"),"-e",str(WORK)+"[train]"],
           cwd=WORK,log_path=DRIVE_ROOT/"setup-logs"/("install-"+uuid4().hex+".log"))
assert metadata.version("torch")==torch_version, "Torch thay đổi: dừng và kiểm tra cài đặt"''')
        elif source.startswith('import torch\nfrom datamate.training.colab'):
            source='from uuid import uuid4\n'+source
            source=source.replace('OUT.mkdir(parents=True,exist_ok=True)',
                                  'OUT.mkdir(parents=True,exist_ok=True)\ninvocation_id=uuid4().hex')
            source=source.replace('(OUT/"environment.json")','(OUT/("environment-"+invocation_id+".json"))')
            source=source.replace('(OUT/"pip-freeze.txt")','(OUT/("pip-freeze-"+invocation_id+".txt"))')
            source=source.replace('if (BASE_DIR/"artifact_manifest.json").is_file():\n    verify_artifact(BASE_DIR,"qwen35-9b")\nelse:\n    stage_model("qwen35-9b",BASE_DIR)',
                '''if not (BASE_DIR/"artifact_manifest.json").is_file():
    stage_model("qwen35-9b",BASE_DIR)
# Copy from persistent Drive to local SSD, then verify the complete sealed artifact.
# No marker is placed inside the sealed model directory.
import shutil, time
drive_base=BASE_DIR
base_sha=hashlib.sha256((drive_base/"artifact_manifest.json").read_bytes()).hexdigest()
BASE_DIR=Path("/content/datamate-models")/("qwen35-9b-"+base_sha[:16])
local_valid=False
if (BASE_DIR/"artifact_manifest.json").is_file():
    try:
        verify_artifact(BASE_DIR,"qwen35-9b")
        local_valid=True
    except Exception as exc:
        print("Local model cache chưa hoàn tất, chép lại từ Drive:",exc)
copy_started=time.monotonic()
if not local_valid:
    base_manifest=json.loads((drive_base/"artifact_manifest.json").read_text(encoding="utf-8"))
    needed=sum(item["bytes"] for item in base_manifest["files"].values())
    assert shutil.disk_usage("/content").free>needed, "Thiếu local disk để stage model"
    BASE_DIR.mkdir(parents=True,exist_ok=True)
    for name in [*base_manifest["files"],"artifact_manifest.json"]:
        original=(drive_base/name).resolve()
        target=(BASE_DIR/name).resolve()
        assert original.is_relative_to(drive_base.resolve()) and target.is_relative_to(BASE_DIR.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        print("Copy model:",name,flush=True)
        shutil.copy2(original,target)
    verify_artifact(BASE_DIR,"qwen35-9b")
print("Model local đã xác minh:",BASE_DIR,"; copy/verify seconds:",round(time.monotonic()-copy_started,2))''')
            source+='''
from uuid import uuid4
kernel_report=OUT/("kernel-check-"+uuid4().hex+".json")
run_logged([sys.executable,"-u","-m","datamate.training.kernels","--output",str(kernel_report)],
           cwd=WORK,log_path=OUT/"logs"/("kernel-check-"+uuid4().hex+".log"))
print("Kernel forward/backward report:",kernel_report)
'''
        elif source.startswith('config_path=OUT'):
            source='''config_path=OUT/"training_config.json"
from datamate.training.checkpoints import completed_report, ensure_training_config
ensure_training_config(config_path,profile)
from uuid import uuid4
report=completed_report(OUT/"adapter",profile)
if report is None:
    command=[sys.executable,"-u","-m","datamate","train","--config",str(config_path)]
    if RESUME is not None: command += ["--resume",str(RESUME)]
    run_logged(command,cwd=WORK,log_path=OUT/"logs"/("train-"+uuid4().hex+".log"))
    report=completed_report(OUT/"adapter",profile)
    assert report is not None, "Thiếu báo cáo hoàn tất; xem log/checkpoint"
else:
    print("Lượt train này đã hoàn tất; hiển thị kết quả đã xác minh, không chạy lại.")
print(json.dumps({k:report.get(k) for k in ["global_step","train_sequences","validation_sequences",
    "trainable_parameters","peak_cuda_bytes","metrics","kernel_dispatch"]},ensure_ascii=False,indent=2))
print("Adapter và checkpoint trên Drive:",OUT/"adapter")
print("Sau pilot: đổi STAGE='full' và RUN_TAG mới. Sau ngắt kết nối: giữ cùng RUN_TAG, STAGE và cấu hình để resume.")
'''
        cell['source']=source.splitlines(True)
        if cell['cell_type']=='code': ast.parse(source)
    # Preserve the old executed notebooks; this builder produces a new clean release.
    Path(output).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
