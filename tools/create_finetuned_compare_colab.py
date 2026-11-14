"""Colab Pro three-arm comparison using the selected completed fine-tuned artifact."""
import json
from pathlib import Path

try:
    from .create_inference_notebook import cell
except ImportError:
    from create_inference_notebook import cell

PROFILE = json.loads((Path(__file__).resolve().parents[1]/'configs/comparison/finetuned-artifact.json').read_text(encoding='utf-8'))

SETUP = '''from google.colab import drive
drive.mount("/content/drive")
from pathlib import Path
import os,sys,json,zipfile,stat,subprocess,runpy,time,hashlib

STORAGE = Path(os.environ.get("DATAMATE_STORAGE_DIR","/content/drive/MyDrive/DataMate"))
SOURCE_ZIP = Path(os.environ.get("DATAMATE_SOURCE_ZIP",str(STORAGE / "datamate-colab-source.zip")))
WORK = Path(os.environ.get("DATAMATE_WORK_DIR","/content/datamate"))
TRAIN_ROOT = Path(os.environ.get("DATAMATE_SPLIT_ROOT",str(STORAGE / "data/train")))
TRAIN_ZIP = STORAGE / "train.zip"  # Optional known TRAIN archive if TRAIN_ROOT is not extracted.
RUN_NAME = "stage2-" + time.strftime("%Y%m%d-%H%M%S")
SAVE = Path(os.environ.get("DATAMATE_RESULT_DIR",str(STORAGE / "runs/Bảo" / RUN_NAME)))
SAVE.mkdir(parents=True,exist_ok=True)
if SOURCE_ZIP.is_file():
    with SOURCE_ZIP.open("rb") as handle:
        source_hash = hashlib.file_digest(handle,"sha256").hexdigest()
    WORK = WORK / ("source-" + source_hash[:12])
    receipt = WORK / ".source_archive_sha256"
    if WORK.exists() and any(WORK.iterdir()) and (not receipt.is_file() or receipt.read_text().strip()!=source_hash):
        raise RuntimeError("Source folder differs from ZIP; choose a new WORK path")
    WORK.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        if len(set(archive.namelist()))!=len(archive.namelist()) or archive.testzip() is not None:
            raise ValueError("Duplicate entries or corrupt source ZIP")
        for entry in archive.infolist():
            target = (WORK / entry.filename).resolve()
            if not target.is_relative_to(WORK.resolve()) or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Unsafe source ZIP path")
        archive.extractall(WORK)
    receipt.write_text(source_hash + "\\n",encoding="utf-8")
elif not (WORK / "src/datamate/__init__.py").is_file():
    raise FileNotFoundError("Upload SOURCE_ZIP or point WORK to a complete source folder")
loaded = sys.modules.get("datamate")
if loaded and not Path(loaded.__file__).resolve().is_relative_to(WORK / "src"):
    raise RuntimeError("Another DataMate source is loaded; restart kernel before comparison")
if not TRAIN_ROOT.is_dir():
    if not TRAIN_ZIP.is_file():
        raise FileNotFoundError("Point TRAIN_ROOT to the extracted benchmark split")
    helpers = runpy.run_path(str(WORK / "tools/colab_archives.py"))
    TRAIN_ROOT = helpers["prepare_train_archive"](TRAIN_ZIP,WORK,WORK / "configs/training_sources.json")
os.chdir(WORK)
sys.path.insert(0,str(WORK))
os.environ.update(HF_HOME=str(WORK / ".cache/huggingface"),TORCH_HOME=str(WORK / ".cache/torch"),
                  PIP_CACHE_DIR=str(WORK / ".cache/pip"))
print("Source:",WORK,"| data:",TRAIN_ROOT,"| results:",SAVE)
'''
SETUP += f'''
ADAPTER_ZIP = STORAGE / {PROFILE['archive_name']!r}
TRAINING_STORAGE = Path("/content/drive/MyDrive/DataMateSQLSFT")
BASE_SOURCE_DIR = TRAINING_STORAGE / "models/qwen35-9b-c202236"  # Training workflow default; hash checked below.
MERGED_STORE = TRAINING_STORAGE / "models/qwen35-9b-stage2-comparison-merged"
EXPECTED_ADAPTER_SHA256 = {PROFILE['adapter_sha256']!r}
EXPECTED_TRAINING_REPORT_SHA256 = {PROFILE['source_training_report_sha256']!r}
EXPECTED_PACKAGE_SHA256 = {PROFILE['archive_sha256']!r}
IDS_FILE = STORAGE / "benchmark/ids.json"  # Same frozen set for all three arms.
REFERENCES_FILE = None  # Optional gold JSONL; read only after inference.
COMPARE_OUTPUT = SAVE / "comparison"  # Keep fresh; preparation logs stay outside.
DTYPE = "float16"  # Common generator precision for all three arms.
'''

INSTALL = '''# Keep Colab's kernel packages separate from the inference stack.
runtime_key = hashlib.sha256((WORK / "pyproject.toml").read_bytes()
    + b"peft==0.21.2|psutil>=5" + str(sys.version_info[:2]).encode()).hexdigest()[:16]
runtime_dir = Path(os.environ.get("DATAMATE_RUNTIME_DIR", "/content/datamate-envs/" + runtime_key))
RUNTIME_PYTHON = str(runtime_dir / "bin/python")
RUNTIME_ENV = {**os.environ,"PYTHONNOUSERSITE":"1",
    "PYTHONPATH":str(WORK / "src") + os.pathsep + str(WORK)}
RUNTIME_ENV.pop("PYTHONHOME",None)
if not Path(RUNTIME_PYTHON).is_file():
    subprocess.run([sys.executable,"-m","venv","--without-pip",str(runtime_dir)],env=RUNTIME_ENV,check=True)
if "include-system-site-packages = false" not in (runtime_dir / "pyvenv.cfg").read_text().lower():
    raise RuntimeError("Runtime must be an isolated venv; choose a new DATAMATE_RUNTIME_DIR")
print("Installing/checking private runtime:",RUNTIME_PYTHON,flush=True)
# pip --python works even when Debian/Colab venv has no ensurepip.
subprocess.run([sys.executable,"-m","pip","--python",RUNTIME_PYTHON,"install","pip",
    "-e",".[llm,vllm,rag]","peft==0.21.2","psutil>=5"],cwd=WORK,env=RUNTIME_ENV,check=True)
checked = subprocess.run([RUNTIME_PYTHON,"-m","pip","check"],env=RUNTIME_ENV,capture_output=True,text=True)
check_output = (checked.stdout or "") + (checked.stderr or "")
print(check_output,flush=True)
(SAVE / "pip-check.txt").write_text(check_output,encoding="utf-8")
if checked.returncode:
    raise RuntimeError("Private inference runtime has dependency conflicts:\\n" + check_output)
frozen = subprocess.run([RUNTIME_PYTHON,"-m","pip","freeze"],env=RUNTIME_ENV,capture_output=True,text=True,check=True)
(SAVE / "runtime-freeze.txt").write_text(frozen.stdout,encoding="utf-8")
subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv"],check=True)
sys.path.insert(0,str(WORK / "src"))
print("Runtime ready. All three arms use:",RUNTIME_PYTHON)
'''

PREPARE = '''started = time.monotonic()
from tools.checkpoint_inference import extract_package,verify_package
from tools.colab_model_cache import prepare_local_model
from datamate.models.artifacts import verify_artifact,sha256
from datamate.data.split import DatasetSplit

if sha256(ADAPTER_ZIP) != EXPECTED_PACKAGE_SHA256:
    raise RuntimeError("Adapter ZIP differs from the selected completed stage2 artifact; upload the verified ZIP")
if not Path(IDS_FILE).is_file():
    raise FileNotFoundError("Provide IDS_FILE: a frozen JSON array of benchmark IDs before preparing models")
selected_ids = json.loads(Path(IDS_FILE).read_text(encoding="utf-8-sig"))
if (not isinstance(selected_ids,list) or not selected_ids or any(not isinstance(q,str) for q in selected_ids)
        or len(set(selected_ids))!=len(selected_ids)):
    raise ValueError("IDS_FILE must contain a nonempty array of unique question IDs")
if not set(selected_ids)<={q.id for q in DatasetSplit(TRAIN_ROOT).questions()}:
    raise ValueError("IDS_FILE contains IDs outside the selected benchmark split")
package_root = extract_package(ADAPTER_ZIP,WORK / ".cache/finetuned-adapter" / EXPECTED_PACKAGE_SHA256[:12])
manifest = verify_package(package_root)
if (manifest.get("source_kind") != "completed_adapter"
        or manifest["files"]["adapter/adapter_model.safetensors"] != EXPECTED_ADAPTER_SHA256
        or manifest["source_training_report_sha256"] != EXPECTED_TRAINING_REPORT_SHA256):
    raise RuntimeError("Select the completed stage2 adapter with its original training_report")
print("Verified completed stage:",manifest["source_global_step"],"requested total epochs:",manifest["verified_requested_total_epochs"])
base = BASE_SOURCE_DIR
if not (MERGED_STORE / "artifact_manifest.json").is_file():
    candidates = [BASE_SOURCE_DIR,Path("/content/datamate-models") / ("qwen35-9b-" + manifest["expected_base_manifest_sha256"][:16])]
    base = next((p for p in candidates if (p / "artifact_manifest.json").is_file()
                 and sha256(p / "artifact_manifest.json") == manifest["expected_base_manifest_sha256"]),None)
    if base is None:
        raise RuntimeError("BASE_SOURCE_DIR must point to the exact sealed training base (manifest " + manifest["expected_base_manifest_sha256"] + "); do not substitute another base")
    base = prepare_local_model(base,WORK / ".cache/models","qwen35-9b")
print("Merging/reusing completed stage2 adapter on Colab; output:",MERGED_STORE,flush=True)
with (SAVE / "prepare-merge.log").open("w",encoding="utf-8") as log:
    subprocess.run([RUNTIME_PYTHON,"-u",str(WORK / "tools/checkpoint_inference.py"),"merge",
        "--package-root",str(package_root),"--base",str(base),"--output",str(MERGED_STORE)],
        cwd=WORK,env=RUNTIME_ENV,check=True,stdout=log,stderr=subprocess.STDOUT)
verify_artifact(MERGED_STORE,"qwen35-9b")
MODEL_DIR = prepare_local_model(MERGED_STORE,WORK / ".cache/models","qwen35-9b")
paths = {}
for key in ("vietnamese-embedding","vietnamese-reranker"):
    source = STORAGE / "models" / key
    if not (source / "artifact_manifest.json").is_file():
        # Explicit ONLINE preparation only, never inside benchmark inference.
        env = {**RUNTIME_ENV,"HF_HUB_OFFLINE":"0","TRANSFORMERS_OFFLINE":"0"}
        subprocess.run([RUNTIME_PYTHON,"-m","datamate","stage-model","--model",key,"--output",str(source)],
                       cwd=WORK,env=env,check=True)
    verify_artifact(source,key)
    paths[key] = prepare_local_model(source,WORK / ".cache/models",key)
EMBEDDING_DIR,RERANKER_DIR = paths["vietnamese-embedding"],paths["vietnamese-reranker"]
(SAVE / "preparation.json").write_text(json.dumps({"seconds":time.monotonic()-started,
    "merged_model_dir":str(MODEL_DIR),"adapter_sha256":EXPECTED_ADAPTER_SHA256,
    "training_report_sha256":EXPECTED_TRAINING_REPORT_SHA256,"adapter_package_sha256":EXPECTED_PACKAGE_SHA256,
    "note":"Cell3 verification/copy/one-time merge/RAG staging; source extraction and pip are separate"},indent=2))
print("All arms will use this merged artifact:",MODEL_DIR)
'''

COMPARE = '''command = [RUNTIME_PYTHON,"-u",str(WORK / "tools/compare_system.py"),
    "--config",str(WORK / "configs/comparison/baseline.json"),"--ids",str(IDS_FILE),
    "--split-root",str(TRAIN_ROOT),"--model-dir",str(MODEL_DIR),
    "--embedding-dir",str(EMBEDDING_DIR),"--reranker-dir",str(RERANKER_DIR),
    "--dtype",DTYPE,"--output",str(COMPARE_OUTPUT)]
if REFERENCES_FILE is not None:
    command += ["--references",str(REFERENCES_FILE)]
subprocess.run(command,cwd=WORK,env=RUNTIME_ENV,check=True)
print("Results:",COMPARE_OUTPUT / "comparisons.json")
'''


def build(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    nb = {'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'accelerator':'GPU'},'cells':[
        cell('markdown','# Fine-tuned stage 2 · Colab Pro · baseline / vLLM / RAG\n\n'
             'Dùng adapter hoàn tất qwen35-9b-sql-stage2-v1-full-01, giữ training_report gốc. '
             'Upload source ZIP, qwen35-9b-stage2-final-inference.zip, dataset/ids và đúng base đã train vào Drive. '
             'Sửa BASE_SOURCE_DIR/IDS_FILE/TRAIN_ROOT theo nơi đã lưu. Chọn GPU và đủ RAM hệ thống cho bước merge CPU trên Colab. '
             'Cài dependency chung một lần; merge một lần, cả ba lượt dùng cùng model merged. '
             'Đã đối chiếu hash report/adapter: epoch 1 cha + 0,5 epoch stage 2. '
             'Baseline là model đã fine-tune, RAG tắt. Không load/thử full model trên laptop. '
             'Xem docs/colab-finetuned-compare.md.\n','finetuned-intro'),
        cell('code',SETUP,'finetuned-setup'),
        cell('markdown','## Cài môi trường inference riêng\nTorch/vLLM/RAG dùng chung một venv sạch, tách khỏi thư viện có sẵn của Colab. Kết quả pip check và pip freeze được lưu trên Drive.\n','finetuned-install-help'),
        cell('code',INSTALL,'finetuned-install'),
        cell('markdown','## Chuẩn bị adapter và model\nKhông train thêm. CPU merge cần RAM còn trống theo preflight; xem prepare-merge.log trên Drive nếu lỗi.\n','finetuned-prepare-help'),
        cell('code',PREPARE,'finetuned-prepare'),
        cell('markdown','## Chạy baseline trước rồi vLLM và RAG\nCùng source/IDs/checkpoint/dtype/budget. Gold chỉ được dùng sau inference. Dùng thư mục comparison mới cho mỗi phép so.\n','finetuned-compare-help'),
        cell('code',COMPARE,'finetuned-compare')]}
    for item in nb['cells']:
        if item['cell_type']=='code': compile(''.join(item['source']),item['id'],'exec')
    path = root/'notebooks/DataMate_Colab_Finetuned_Compare.ipynb'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return path


if __name__=='__main__':
    print(build())
