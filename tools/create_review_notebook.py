"""Generate the current-working-tree Colab notebook, retaining the proven stack."""
import ast
import json
from pathlib import Path


SETUP = '''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import hashlib, json, os, stat, subprocess, sys, time, zipfile

STORAGE = Path('/content/drive/MyDrive/DataMate')
SOURCE_ZIP = STORAGE / 'datamate-review-source.zip'
SOURCE_MANIFEST = Path(str(SOURCE_ZIP) + '.manifest.json')
FROZEN_ENV = STORAGE / 'runtime-freeze.txt'  # Exact freeze from the successful Public1000 run.
SPLIT_KIND = 'train'  # train: smoke/benchmark; public: all public questions.
DATA_SOURCE = STORAGE / 'train.zip'  # ZIP or a directory containing ONE split.
LIMIT = 20  # train only; set None for a complete split. public requires None.
PROFILE = 'optimized'  # optimized | default (old baseline) | correction | hybrid | concurrency2 | concurrency4 | prefix | context20k
RUN_TAG = 'train-optimized-01'  # Use a NEW tag for every source/config/data/profile change.
RESUME = False
MODEL_PRESET = 'qwen35-9b-finetuned'  # qwen35-9b-finetuned | gemma4-12b
MERGED_SOURCE = Path('/content/drive/MyDrive/DataMateSQLSFT/models/qwen35-9b-stage2-comparison-merged')
GEMMA_SOURCE = STORAGE / 'models/gemma4-12b'  # Full sealed checkpoint, not a LoRA adapter.
EMBEDDING_SOURCE = STORAGE / 'models/vietnamese-embedding'
RERANKER_SOURCE = STORAGE / 'models/vietnamese-reranker'
CORRECTION_SOURCE = STORAGE / 'models/vietnamese-error-correction'
EXISTING_RUNTIME_PYTHON = ''  # Optional: absolute Python of an existing proven Colab venv.

model_presets = {
    'qwen35-9b-finetuned': ('qwen35-9b', MERGED_SOURCE),
    'gemma4-12b': ('gemma4-12b', GEMMA_SOURCE),
}
if MODEL_PRESET not in model_presets:
    raise ValueError('Unknown MODEL_PRESET; a new architecture needs an audited registry/artifact first')
MODEL_KEY, GENERATOR_SOURCE = model_presets[MODEL_PRESET]

if PROFILE not in {'optimized','default','correction','hybrid','concurrency2','concurrency4','prefix','context20k'}:
    raise ValueError('Unknown PROFILE')
if not RUN_TAG or Path(RUN_TAG).name != RUN_TAG or RUN_TAG in {'.','..'}:
    raise ValueError('RUN_TAG must be a single directory name')
SAVE = STORAGE / 'runs/review' / RUN_TAG
if not RESUME and SAVE.exists() and any(SAVE.iterdir()):
    raise ValueError('RUN_TAG already exists. Choose a new tag, or RESUME=True with identical inputs.')
SAVE.mkdir(parents=True,exist_ok=True)
def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
manifest = json.loads(SOURCE_MANIFEST.read_text(encoding='utf-8'))
source_hash = digest(SOURCE_ZIP)
if source_hash != manifest['sha256']:
    raise ValueError('Source ZIP does not match its manifest')
WORK = Path('/content/datamate-review') / ('source-' + source_hash[:12])
WORK.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(SOURCE_ZIP) as archive:
    for entry in archive.infolist():
        target = (WORK / entry.filename).resolve()
        if ('\\\\' in entry.filename or ':' in entry.filename or not target.is_relative_to(WORK.resolve())
                or stat.S_ISLNK(entry.external_attr >> 16)):
            raise ValueError('Unsafe source ZIP path')
    archive.extractall(WORK)
for name, expected in manifest['files'].items():
    if digest(WORK/name) != expected:
        raise ValueError('Source file mismatch: ' + name)
expected_python = {name: value for name,value in manifest['files'].items() if name.startswith('src/') and name.endswith('.py')}
actual_python = {p.relative_to(WORK).as_posix():digest(p) for p in (WORK/'src').rglob('*.py')}
if actual_python != expected_python:
    raise ValueError('Stray Python source in scratch workspace; restart the Colab session')
for name, module in list(sys.modules.items()):
    if name == 'datamate' or name.startswith('datamate.') or name == 'tools.colab_review':
        location = getattr(module,'__file__',None)
        if location and not Path(location).resolve().is_relative_to(WORK):
            raise RuntimeError('Different source already imported. Restart the Colab session.')
os.chdir(WORK)
sys.path[:0] = [str(WORK/'src'),str(WORK)]
from tools.colab_review import write_json, stage_split, diagnostics, export_submission
receipt = {'source_sha256':source_hash,'freeze_sha256':digest(FROZEN_ENV),'profile':PROFILE,'model_preset':MODEL_PRESET}
receipt_path = SAVE/'source.json'
if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
    raise ValueError('Source/environment/profile changed. Use a new RUN_TAG.')
write_json(receipt_path,receipt)
print('SOURCE:',WORK,'\\nSAVE:',SAVE,flush=True)
'''

INSTALL = '''# Run this cell on Colab GPU only. Keep all versions from the known-good freeze.
runtime_key = hashlib.sha256(FROZEN_ENV.read_bytes()+str(sys.version_info[:2]).encode()).hexdigest()[:16]
runtime_dir = Path('/content/datamate-review-envs')/runtime_key
RUNTIME_PYTHON = EXISTING_RUNTIME_PYTHON or str(runtime_dir/'bin/python')
RUNTIME_ENV = {k:v for k,v in os.environ.items() if not k.startswith('DATAMATE_')}
RUNTIME_ENV.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
    PYTHONPATH=str(WORK/'src')+os.pathsep+str(WORK),VIRTUAL_ENV=str(Path(RUNTIME_PYTHON).parent.parent),
    PATH=str(Path(RUNTIME_PYTHON).parent)+os.pathsep+os.environ.get('PATH',''),
    VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0',HF_DATASETS_OFFLINE='0')
RUNTIME_ENV.pop('PYTHONHOME',None)
if not EXISTING_RUNTIME_PYTHON:
    if not Path(RUNTIME_PYTHON).is_file():
        subprocess.run([sys.executable,'-m','venv','--without-pip',str(runtime_dir)],check=True)
    if 'include-system-site-packages = false' not in (runtime_dir/'pyvenv.cfg').read_text().lower():
        raise RuntimeError('Inference venv must be isolated')
    subprocess.run([sys.executable,'-m','pip','--python',RUNTIME_PYTHON,'install','pip',
        '-r',str(FROZEN_ENV),'-c',str(FROZEN_ENV),'-e',str(WORK)+'[llm,vllm,rag]'],
        cwd=WORK,env=RUNTIME_ENV,check=True)
# Existing runtime: do not pip install or mutate it. Both modes must pass the same version gate.
version_check = """import importlib.metadata as m, json, pathlib, re, sys
expected={}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if '==' in line and not line.startswith('#'):
        name,version=line.strip().split('==',1); expected[name]=version
mismatch={}
for name,version in expected.items():
    try: actual=m.version(name)
    except m.PackageNotFoundError: actual=None
    if actual!=version: mismatch[name]={'expected':version,'actual':actual}
print(json.dumps(mismatch,indent=2))
if mismatch: raise SystemExit('Runtime differs from successful Public1000 freeze; no automatic upgrades/downgrades')
"""
subprocess.run([RUNTIME_PYTHON,'-c',version_check,str(FROZEN_ENV)],env=RUNTIME_ENV,check=True)
check = subprocess.run([RUNTIME_PYTHON,'-m','pip','check'],env=RUNTIME_ENV,capture_output=True,text=True)
(SAVE/'pip-check.txt').write_text(check.stdout+check.stderr,encoding='utf-8')
print(check.stdout+check.stderr)
check.check_returncode()
freeze = subprocess.run([RUNTIME_PYTHON,'-m','pip','freeze'],env=RUNTIME_ENV,capture_output=True,text=True,check=True)
(SAVE/'runtime-freeze.txt').write_text(freeze.stdout,encoding='utf-8')
gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version',
    '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
print(gpu.stdout)
(SAVE/'gpu-preflight.txt').write_text(gpu.stdout,encoding='utf-8')
first = [part.strip() for part in gpu.stdout.splitlines()[0].split(',')]
if min(int(first[1]),int(first[2])) < 39*1024:
    raise RuntimeError('Use a free A100 40GB or larger; profile not validated on smaller cards')
import shutil
for executable in ('ninja','nvcc','c++'):
    location = shutil.which(executable,path=RUNTIME_ENV['PATH'])
    if not location: raise RuntimeError('Missing executable: '+executable)
    subprocess.run([location,'--version'],env=RUNTIME_ENV,check=True)
OFFLINE_ENV = {**RUNTIME_ENV,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
print('Same frozen inference libraries; ready to stage input/models.')
'''

PREPARE = '''# Source helpers copy files only; the GPU backend verifies every model before loading.
from tools.colab_model_cache import prepare_local_model
from datamate.models.registry import inventory
# Validate the full inference parameter budget before copying large checkpoints.
use_correction = PROFILE in {'optimized','correction'}
extra_models = ['vietnamese-embedding','vietnamese-reranker']
if use_correction: extra_models.append('vietnamese-error-correction')
model_inventory = inventory(MODEL_KEY, extra_keys=extra_models)
SPLIT_ROOT, input_report = stage_split(DATA_SOURCE,Path('/content/datamate-review/data'),kind=SPLIT_KIND,limit=LIMIT)
input_path = SAVE/'input.json'
if input_path.exists() and json.loads(input_path.read_text()) != input_report:
    raise ValueError('Input changed; choose a new RUN_TAG')
write_json(input_path,input_report)
paths = {}
sources = [(MODEL_KEY,GENERATOR_SOURCE),('vietnamese-embedding',EMBEDDING_SOURCE),('vietnamese-reranker',RERANKER_SOURCE)]
if use_correction:
    if not (CORRECTION_SOURCE/'artifact_manifest.json').is_file():
        # Explicit ONLINE preparation before inference, using the existing frozen venv.
        online_env = {**RUNTIME_ENV,'HF_HUB_OFFLINE':'0','TRANSFORMERS_OFFLINE':'0'}
        subprocess.run([RUNTIME_PYTHON,'-m','datamate','stage-model','--model','vietnamese-error-correction',
            '--output',str(CORRECTION_SOURCE)],cwd=WORK,env=online_env,check=True)
    sources.append(('vietnamese-error-correction',CORRECTION_SOURCE))
for key,source in sources:
    if not (source/'artifact_manifest.json').is_file():
        raise FileNotFoundError('Provide the full sealed checkpoint for the selected model: '+str(source))
    if key=='qwen35-9b' and digest(source/'artifact_manifest.json') != '732f292bfb088f8dc90b0fe91e255a60886772292614cf0b1af58219d097b054':
        raise ValueError('Generator is not the exact merged model of the successful Public1000 run')
    paths[key] = prepare_local_model(source,Path('/content/datamate-review/models'),key)
config = json.loads((WORK/'configs/submission.json').read_text(encoding='utf-8'))
RUN_OUTPUT = SAVE/'run'
config.update(split_root=str(SPLIT_ROOT),cache_dir=str(Path('/content/datamate-review/cache')/source_hash[:12]/input_report['fingerprint']),
    result_dir=str(RUN_OUTPUT),model_key=MODEL_KEY,model_dir=str(paths[MODEL_KEY]))
config['vllm']['gpu_memory_utilization'] = 0.75
config['rag'].update(device='cuda:0',embedding_dir=str(paths['vietnamese-embedding']),
    reranker_dir=str(paths['vietnamese-reranker']),index_dir=str(Path(config['cache_dir'])/'retrieval'))
if use_correction:
    config['rag'].update(correction_enabled=True,correction_dir=str(paths['vietnamese-error-correction']))
if PROFILE in {'optimized','hybrid'}:
    config['rag']['hybrid_retrieval'] = True
if PROFILE == 'optimized':
    config['question_workers'] = 2
    config['vllm'].update(max_num_seqs=2,enable_prefix_caching=True,cache_prompt_prefix=True)
    config['recovery_steps'] = 2
if PROFILE in {'concurrency2','concurrency4'}:
    workers = 2 if PROFILE=='concurrency2' else 4
    config['question_workers'] = workers
    config['vllm']['max_num_seqs'] = workers
elif PROFILE=='prefix':
    config['vllm'].update(enable_prefix_caching=True,cache_prompt_prefix=True)
elif PROFILE=='context20k':
    config['max_context_tokens'] = 20480
CONFIG = SAVE/'config.json'
if CONFIG.exists() and json.loads(CONFIG.read_text()) != config:
    raise ValueError('Configuration changed; choose a new RUN_TAG')
write_json(CONFIG,config)
subprocess.run([RUNTIME_PYTHON,'-c',
    'from datamate.config import Settings; import sys; Settings.load(sys.argv[1],use_env=False); print("Config OK")',str(CONFIG)],
    cwd=WORK,env=OFFLINE_ENV,check=True)
print('Questions:',input_report['questions'],'| businesses:',input_report['businesses'],'| profile:',PROFILE)
print('Generator:',MODEL_KEY,'| total inference parameters:',model_inventory['total_parameters'])
print('Inference input contains no answers.jsonl. Output:',RUN_OUTPUT)
'''

RUN = '''# Same CLI and supervisor as Docker; save every completed answer to Drive.
import signal
command = [RUNTIME_PYTHON,'-u','-m','datamate','run','--config',str(CONFIG)]
if RESUME: command.append('--resume')
process = subprocess.Popen(command,cwd=WORK,env=OFFLINE_ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
    text=True,encoding='utf-8',errors='replace',bufsize=1,start_new_session=True)
try:
    with (SAVE/'inference.log').open('a',encoding='utf-8',buffering=1) as stream:
        for line in process.stdout:
            print(line,end='',flush=True)
            stream.write(line)
    status = process.wait()
    write_json(SAVE/'last-process.json',{'returncode':status,'resume':RESUME})
    if status: raise RuntimeError('Inference failed; keep SAVE and inspect inference.log before resuming.')
finally:
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try: process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
print(json.dumps(diagnostics(RUN_OUTPUT),ensure_ascii=False,indent=2))
'''

EXPORT = '''SUBMISSION = SAVE/'submission.jsonl'
report = export_submission(SPLIT_ROOT,RUN_OUTPUT,SUBMISSION)
print(json.dumps(report,ensure_ascii=False,indent=2))
print('OUTPUT:',SUBMISSION)
if SPLIT_KIND == 'public':
    from google.colab import files
    files.download(str(SUBMISSION))
else:
    print('TRAIN smoke output: do not upload this file as public submission.')
'''


def build(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    cells=[]
    for index,(title,code) in enumerate([
        ('# DataMate · Chạy source đã sửa trên Colab\n\nA100, runtime-freeze của Public1000; chọn MODEL_PRESET và PROFILE trong ô1. PROFILE optimized tích hợp correction + hybrid retrieval + concurrency2 + prefix + bounded recovery. PROFILE default là đối chứng cũ. Chưa benchmark GPU bản mới. Cần full sealed generator/RAG. Xem `zlai/huogndan.md`; mặc định TRAIN20, không tự nộp bài.',SETUP),
        ('## 2. Runtime giữ đúng phiên bản\nChỉ cài trong venv riêng khi chưa có runtime; không sửa thư viện kernel Colab. Thiếu version thì dừng, không tự nâng/hạ stack.',INSTALL),
        ('## 3. Chuẩn bị input và model\nGenerator + hai RAG phải có sẵn. Profile correction/optimized chuẩn bị thêm checkpoint correction đã pin bằng bước online riêng trước inference. Không đọc gold. Cache SSD; kết quả trên Drive.',PREPARE),
        ('## 4. Inference\nMột run tag cho đúng source/config/input. Resume chỉ với cùng bộ đó. Chỉ chạy một process cho một RUN_TAG.',RUN),
        ('## 5. Kiểm output và xuất file\nKiểm đủ ID/log/type trước export. Format hợp lệ không chứng minh PA tốt hơn.',EXPORT),
    ]):
        cells.append({'cell_type':'markdown','metadata':{},'id':f'review-help-{index}','source':title.splitlines(True)})
        ast.parse(code)
        cells.append({'cell_type':'code','metadata':{},'id':f'review-code-{index}',
                      'source':code.splitlines(True),'outputs':[],'execution_count':None})
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU',
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    path=root/'notebooks/DataMate_Colab_Review.ipynb'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return path


if __name__=='__main__':
    print(build())
