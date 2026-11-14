"""Colab Pro preparation and three-arm inference from Bảo's checkpoint-198."""
import json
from pathlib import Path

try:
    from .create_inference_notebook import SETUP as GPU_SETUP,cell
except ImportError:
    from create_inference_notebook import SETUP as GPU_SETUP,cell


SETUP = GPU_SETUP.replace('PLATFORM = os.environ.get("DATAMATE_PLATFORM", "colab")','PLATFORM = "colab"')
SETUP = SETUP.replace('MODEL_KEY = os.environ.get("DATAMATE_MODEL_KEY", "qwen35-9b")','MODEL_KEY = "qwen35-9b"')
SETUP = SETUP.replace('OWNER = "hai"','OWNER = "Bảo"')
SETUP += '''
# Upload ZIP source as datamate-colab-source.zip and the checkpoint ZIP below.
CHECKPOINT_ZIP = STORAGE / "checkpoint-198-inference.zip"
BASE_SOURCE_DIR = STORAGE / "models/qwen35-9b"  # Exact sealed base used for training.
MERGED_STORE = STORAGE / "models/qwen35-9b-checkpoint-198-merged"
EXPECTED_ADAPTER_SHA256 = "f346ad51cd37abe3ea5f9d127bd506f4c45f1341c836c8fbd54c86c6426a4ebe"
IDS_FILE = STORAGE / "benchmark/ids.json"  # Same frozen set for all three arms.
REFERENCES_FILE = None  # Optional gold JSONL; read only after inference.
COMPARE_OUTPUT = SAVE / "comparison"  # Keep fresh; preparation logs stay outside.
DTYPE = "float16"  # Common generator precision for all three arms.
'''

INSTALL = '''from importlib.metadata import version,PackageNotFoundError
def installed(name):
    try: return version(name)
    except PackageNotFoundError: return None
loaded = {n:installed(n) for n in ("torch","transformers","tokenizers","numpy","peft","vllm") if n in sys.modules}
subprocess.run([sys.executable,"-m","pip","install","-e",".[llm,vllm,rag]","peft==0.21.2","psutil>=5"],cwd=WORK,check=True)
changed = [n for n,before in loaded.items() if installed(n)!=before]
if changed:
    raise RuntimeError("Restart kernel before inference; loaded packages changed: " + str(changed))
subprocess.run([sys.executable,"-m","pip","check"],check=True)
subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv"],check=True)
sys.path.insert(0,str(WORK / "src"))
print("Common stack installed. Model merge runs in a fresh CPU child; inference in fresh GPU workers.")
'''

PREPARE = '''from tools.checkpoint_inference import extract_package,verify_package
from tools.colab_model_cache import prepare_local_model
from datamate.models.artifacts import verify_artifact,sha256

package_root = extract_package(CHECKPOINT_ZIP,WORK / ".cache/checkpoint-198")
manifest = verify_package(package_root)
if manifest["files"]["adapter/adapter_model.safetensors"] != EXPECTED_ADAPTER_SHA256:
    raise RuntimeError("Uploaded checkpoint differs from the selected checkpoint-198")
print("Verified checkpoint:",manifest["source_global_step"],"stage epoch:",manifest["source_stage_epoch"])
base = BASE_SOURCE_DIR
if not (MERGED_STORE / "artifact_manifest.json").is_file():
    if not (base / "artifact_manifest.json").is_file() or sha256(base / "artifact_manifest.json") != manifest["expected_base_manifest_sha256"]:
        raise RuntimeError("BASE_SOURCE_DIR must point to the exact sealed training base (manifest " + manifest["expected_base_manifest_sha256"] + "); do not substitute another base")
    base = prepare_local_model(base,WORK / ".cache/models","qwen35-9b")
print("Merging/reusing checkpoint-198 on Colab; output:",MERGED_STORE,flush=True)
started = time.monotonic()
with (SAVE / "prepare-merge.log").open("w",encoding="utf-8") as log:
    subprocess.run([sys.executable,"-u",str(WORK / "tools/checkpoint_inference.py"),"merge",
        "--package-root",str(package_root),"--base",str(base),"--output",str(MERGED_STORE)],
        cwd=WORK,check=True,stdout=log,stderr=subprocess.STDOUT)
verify_artifact(MERGED_STORE,"qwen35-9b")
MODEL_DIR = prepare_local_model(MERGED_STORE,WORK / ".cache/models","qwen35-9b")
paths = {}
for key in ("vietnamese-embedding","vietnamese-reranker"):
    source = STORAGE / "models" / key
    if not (source / "artifact_manifest.json").is_file():
        # Explicit ONLINE preparation only, never inside benchmark inference.
        env = {**os.environ,"HF_HUB_OFFLINE":"0","TRANSFORMERS_OFFLINE":"0"}
        subprocess.run([sys.executable,"-m","datamate","stage-model","--model",key,"--output",str(source)],
                       cwd=WORK,env=env,check=True)
    verify_artifact(source,key)
    paths[key] = prepare_local_model(source,WORK / ".cache/models",key)
EMBEDDING_DIR,RERANKER_DIR = paths["vietnamese-embedding"],paths["vietnamese-reranker"]
(SAVE / "preparation.json").write_text(json.dumps({"seconds":time.monotonic()-started,
    "merged_model_dir":str(MODEL_DIR),"checkpoint_adapter_sha256":EXPECTED_ADAPTER_SHA256,
    "note":"Online staging/copy/one-time merge outside the three-arm inference timers"},indent=2))
print("All arms will use this merged artifact:",MODEL_DIR)
'''

COMPARE = '''command = [sys.executable,"-u",str(WORK / "tools/compare_system.py"),
    "--config",str(WORK / "configs/comparison/baseline.json"),"--ids",str(IDS_FILE),
    "--split-root",str(TRAIN_ROOT),"--model-dir",str(MODEL_DIR),
    "--embedding-dir",str(EMBEDDING_DIR),"--reranker-dir",str(RERANKER_DIR),
    "--dtype",DTYPE,"--output",str(COMPARE_OUTPUT)]
if REFERENCES_FILE is not None:
    command += ["--references",str(REFERENCES_FILE)]
subprocess.run(command,cwd=WORK,check=True)
print("Results:",COMPARE_OUTPUT / "comparisons.json")
'''


def build(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    nb = {'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'accelerator':'GPU'},'cells':[
        cell('markdown','# Checkpoint-198 · Colab Pro · baseline / vLLM / RAG\n\n'
             'Upload source ZIP, checkpoint-198-inference.zip, dataset/ids và đúng base đã train vào Drive. '
             'Sửa BASE_SOURCE_DIR/IDS_FILE/TRAIN_ROOT theo nơi đã lưu. Chọn GPU và đủ RAM hệ thống cho bước merge CPU trên Colab. '
             'Cài dependency chung một lần; merge một lần, cả ba lượt dùng cùng model merged. '
             'Checkpoint ghi stage ~0,5 epoch và parent adapter hash; tổng1,5 epoch là thông tin chủ dự án. '
             'Không load/thử full model trên laptop. Xem docs/colab-checkpoint198.md.\n','ckpt198-intro'),
        cell('code',SETUP,'ckpt198-setup'),
        cell('markdown','## Cài stack chung trước baseline\nNếu đã import thư viện rồi pip đổi phiên bản, restart kernel và chạy lại.\n','ckpt198-install-help'),
        cell('code',INSTALL,'ckpt198-install'),
        cell('markdown','## Chuẩn bị checkpoint và model\nKhông train thêm. CPU merge cần RAM còn trống theo preflight; xem prepare-merge.log trên Drive nếu lỗi.\n','ckpt198-prepare-help'),
        cell('code',PREPARE,'ckpt198-prepare'),
        cell('markdown','## Chạy baseline trước rồi vLLM và RAG\nCùng source/IDs/checkpoint/dtype/budget. Gold chỉ được dùng sau inference. Dùng thư mục comparison mới cho mỗi phép so.\n','ckpt198-compare-help'),
        cell('code',COMPARE,'ckpt198-compare')]}
    for item in nb['cells']:
        if item['cell_type']=='code': compile(''.join(item['source']),item['id'],'exec')
    path = root/'notebooks/DataMate_Colab_Checkpoint198_Compare.ipynb'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return path


if __name__=='__main__':
    print(build())
