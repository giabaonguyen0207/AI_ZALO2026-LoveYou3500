"""Generate a dedicated Colab Pro notebook; leave the earlier notebooks intact."""
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    cells=[]
    def cell(kind,identifier,text):
        item={'cell_type':kind,'id':identifier,'metadata':{},'source':text.strip().splitlines(keepends=True)}
        if kind=='code': item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','seed-intro','''
# DataMate · Gemma · Seed 72+100 · Colab Pro QLoRA

Notebook nạp **dataset đã chuẩn bị và kiểm chứng**, không chạy baseline lại trên 2.370 câu và không tạo SQL từ nhãn trên Colab.
Model mặc định Gemma 4 12B như các lượt inference; dùng checkpoint đã pin trong source. LoRA attention-only, QLoRA 4bit là cấu hình thử, chưa có kết quả GPU fine-tune.

Đặt cạnh nhau trên Drive: `datamate-colab-gemma-seed-source.zip`, `seed72_100_gemma_v1.zip`, và Gemma đã tải ở thư mục models. `train.zip` chỉ cần ở bước đánh giá generation tùy chọn. Dùng runtime mới để giải phóng agent inference cũ trước khi train.
Chọn GPU trong Runtime → Change runtime type. Colab Pro không bảo đảm một loại GPU; xem GPU/VRAM thực tế ở ô kiểm tra. Profile 16K mặc định dừng dưới 22 GiB; A100 nếu được cấp là lựa chọn để thử. Ngưỡng này không bảo đảm đủ bộ nhớ: bước smoke dùng **mẫu dài nhất thực tế** để kiểm tra trước pilot.
''')
    cell('code','seed-setup','''
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, sys, subprocess, json, zipfile, runpy

DRIVE_ROOT = Path('/content/drive/MyDrive/DataMate')
SOURCE_ZIP = DRIVE_ROOT / 'datamate-colab-gemma-seed-source.zip'
SEED_ZIP = DRIVE_ROOT / 'seed72_100_gemma_v1.zip'
TRAIN_ZIP = DRIVE_ROOT / 'train.zip'  # Chỉ dùng nếu bật đánh giá generation.
MODEL_KEY = 'gemma4-12b'              # Bộ này chỉ dùng Gemma, như hai lượt inference.
MODEL_SOURCE_DIR = DRIVE_ROOT / 'models' / MODEL_KEY
RUN_TAG = 'seed72-100-v1-gemma-qlora-01'
WORK = Path('/content/datamate-seed')
SAVE = DRIVE_ROOT / 'runs' / RUN_TAG
MAX_STEPS = 5                      # Pilot 5 optimizer steps. -1: một epoch theo config.
RESUME_CHECKPOINT = None            # Ví dụ str(SAVE/'adapter/checkpoint-3').
MIN_VRAM_GIB = 22                   # Policy cho profile 16K, không phải kết quả đo fit.
RUN_EVALUATION = False              # Không tự tiêu GPU cho merge/benchmark khi Run all.
TRANSFORMERS_VERSION = '5.22.0'      # Phiên bản đã ghi nhận ở lượt inference Colab trước.

assert MODEL_KEY == 'gemma4-12b'
WORK.mkdir(parents=True,exist_ok=True)
SAVE.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(SOURCE_ZIP) as archive:
    for name in archive.namelist():
        target=(WORK/name).resolve()
        if not target.is_relative_to(WORK.resolve()):
            raise ValueError('Source ZIP chứa đường dẫn không an toàn')
    archive.extractall(WORK)
os.chdir(WORK)
os.environ['HF_HOME']=str(WORK/'.cache/huggingface')
os.environ['PIP_CACHE_DIR']=str(WORK/'.cache/pip')
print('Source:',WORK,'| lưu lâu dài:',SAVE)
''')
    cell('markdown','seed-install-help','''
## Cài thư viện

Giữ PyTorch/CUDA do Colab cấp. Cài PEFT, datasets, bitsandbytes và Transformer theo source. Nếu cần restart runtime sau khi pip thay thư viện đã import, restart rồi chạy lại từ đầu; model và checkpoint trên Drive vẫn còn. Ô dưới không tải trọng số model.
''')
    cell('code','seed-install','''
import torch, importlib
constraints=WORK/'colab-constraints.txt'
constraints.write_text('torch=='+torch.__version__+'\\n')
subprocess.run([sys.executable,'-m','pip','install','-r','requirements-colab.txt',
                'transformers=='+TRANSFORMERS_VERSION,'-c',str(constraints)],cwd=WORK,check=True)
subprocess.run([sys.executable,'-m','pip','install','-e','.', '--no-deps'],cwd=WORK,check=True)
if str(WORK/'src') not in sys.path: sys.path.insert(0,str(WORK/'src'))
importlib.invalidate_caches()
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
(SAVE/'pip-freeze.txt').write_text(freeze,encoding='utf-8')

def cli(*args,env=None):
    subprocess.run([sys.executable,'-m','datamate',*[str(a) for a in args]],cwd=WORK,check=True,env=env)
''')
    cell('markdown','seed-data-help','''
## Kiểm dataset và GPU trước khi tải model

Kiểm hash toàn bộ ZIP, file và phân chia doanh nghiệp. Dataset không chứa public/private test. `validation.jsonl` rỗng vì không dựng gold action trên holdout; đánh giá bằng generation ở cuối. 18 câu validation seed đã từng được xem khi debug, nên đây là đánh giá phát triển.
''')
    cell('code','seed-data-preflight','''
from datamate.training.colab import unpack_seed_archive, make_seed_training_config, gpu_preflight
from datamate.training.seed import verify_seed_dataset

DATASET_DIR=unpack_seed_archive(SEED_ZIP,WORK/'datasets/seed72_100_gemma_v1',
    WORK/'configs/gemma_seed72_100_release.json',WORK/'configs/training_sources.json')
manifest=verify_seed_dataset(DATASET_DIR)
data_report=json.loads((DATASET_DIR/'report.json').read_text(encoding='utf-8'))
print(json.dumps(data_report,ensure_ascii=False,indent=2))
gpu=gpu_preflight(torch,MIN_VRAM_GIB)
(SAVE/'gpu-preflight.json').write_text(json.dumps(gpu,indent=2),encoding='utf-8')
print(json.dumps(gpu,indent=2))
''')
    cell('markdown','seed-model-help','''
## Dùng model đã có trên Drive, chép sang ổ tạm

Chỉ khi chưa có artifact manifest, ô này mới chuẩn bị checkpoint online theo revision đã pin. Các bước train/merge sau dùng file local và chế độ offline. Không nạp lại agent inference cũ cùng lúc với training.
''')
    cell('code','seed-model','''
if not (MODEL_SOURCE_DIR/'artifact_manifest.json').is_file():
    online_env={**os.environ,'HF_HUB_OFFLINE':'0','TRANSFORMERS_OFFLINE':'0'}
    cli('stage-model','--model',MODEL_KEY,'--output',MODEL_SOURCE_DIR,env=online_env)
cache_helper=runpy.run_path(str(WORK/'tools/colab_model_cache.py'))
MODEL_DIR=cache_helper['prepare_local_model'](MODEL_SOURCE_DIR,Path('/content/models'),MODEL_KEY)
print('Base model trên ổ tạm:',MODEL_DIR)
''')
    cell('markdown','seed-smoke-help','''
## Smoke: một optimizer step trên mẫu dài nhất

Smoke adapter được lưu riêng và không dùng làm điểm bắt đầu cho pilot. Bước này kiểm tra bộ nhớ với sequence dài nhất thay vì tình cờ chọn mẫu ngắn. Nếu thiếu VRAM, dừng và chọn GPU/profile khác; không cắt rulebook hoặc lặng lẽ bỏ mẫu. Khi đổi cấu hình sau một lượt lỗi, dùng RUN_TAG mới để giữ bằng chứng lượt cũ.
''')
    cell('code','seed-smoke','''
smoke_dir=SAVE/'smoke-longest'
smoke=make_seed_training_config(MODEL_KEY,MODEL_DIR,DATASET_DIR,smoke_dir,
    smoke=True,min_vram_gib=MIN_VRAM_GIB)
smoke['cache_dir']=str(WORK/'.cache')
smoke_path=SAVE/'smoke-config.json'
smoke_path.write_text(json.dumps(smoke,indent=2),encoding='utf-8')
if not (smoke_dir/'training_report.json').is_file():
    cli('train','--config',smoke_path)
smoke_report=json.loads((smoke_dir/'training_report.json').read_text(encoding='utf-8'))
assert smoke_report['training_manifest_sha256']==__import__('hashlib').sha256((DATASET_DIR/'manifest.json').read_bytes()).hexdigest()
print({key:smoke_report.get(key) for key in ('train_sequences','longest_sequence_tokens',
    'peak_cuda_bytes','cuda_device','elapsed_training_process_seconds','smoke_longest_only')})
''')
    cell('markdown','seed-train-help','''
## Pilot QLoRA và resume

Mặc định 5 optimizer steps, batch 1, tích lũy gradient 8; đây là lượt thử để đo tốc độ/bộ nhớ, không phải khuyến nghị train nhiều epoch trên bộ nhỏ. Đặt MAX_STEPS=-1 để chạy một epoch khi đã xem pilot. Checkpoint lưu mỗi step trên Drive, giữ hai checkpoint gần nhất. Khi resume, giữ RUN_TAG/model/dataset/cấu hình; chỉ tăng giới hạn bước khi cần.
''')
    cell('code','seed-train','''
config=make_seed_training_config(MODEL_KEY,MODEL_DIR,DATASET_DIR,SAVE/'adapter',
    max_steps=MAX_STEPS,min_vram_gib=MIN_VRAM_GIB)
config['cache_dir']=str(WORK/'.cache')
config_path=SAVE/'training-config.json'
config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
args=['train','--config',config_path]
if RESUME_CHECKPOINT: args+=['--resume',RESUME_CHECKPOINT]
cli(*args)
report=json.loads((SAVE/'adapter/training_report.json').read_text(encoding='utf-8'))
print({key:report.get(key) for key in ('metrics','train_sequences','tokens_per_epoch',
    'longest_sequence_tokens','peak_cuda_bytes','elapsed_training_process_seconds')})
''')
    cell('markdown','seed-eval-help','''
## Tùy chọn: merge và so base/fine-tuned trên 18 câu validation

Chỉ chạy khi RUN_EVALUATION=True. Cần train.zip gốc đã có trên Drive. Merge dùng trọng số không nén nên cần thêm RAM/VRAM và dung lượng Drive; QLoRA không biến file merged thành model 4bit. Ô này dùng công cụ merge/evaluate hiện có, chưa được chạy GPU trong lần chuẩn bị dataset.
Đánh giá cùng quantization 4bit cho base và fine-tuned. PA, câu sai/bỏ qua, type và thời gian quan trọng hơn training loss. Khi mở rộng, thay validation_seed_ids.json bằng validation_ids.json để đánh giá đủ 630 câu giữ ngoài train.
''')
    cell('code','seed-eval','''
if RUN_EVALUATION:
    archives=runpy.run_path(str(WORK/'tools/colab_archives.py'))
    TRAIN_ROOT=archives['prepare_train_archive'](TRAIN_ZIP,WORK,WORK/'configs/training_sources.json')
    MERGED_DIR=SAVE/'merged-model'
    if not (MERGED_DIR/'artifact_manifest.json').is_file():
        cli('merge','--model',MODEL_KEY,'--base',MODEL_DIR,'--adapter',SAVE/'adapter','--output',MERGED_DIR)
    paths=[]
    for label,model_dir in [('base',MODEL_DIR),('finetuned',MERGED_DIR)]:
        cfg={'backend':'transformers','model_key':MODEL_KEY,'model_dir':str(model_dir),
             'split_root':str(TRAIN_ROOT),'cache_dir':str(WORK/'.cache/datamate'),
             'result_dir':str(SAVE/('eval-'+label)),'quantization':'4bit','mode':'single',
             'max_context_tokens':16384,'max_new_tokens':768,'max_steps':6,
             'question_timeout_seconds':90,'schema_mode':'code_first','adaptive_preparation':False,
             'require_plan':True,'auto_execute_plan':True}
        path=SAVE/('eval-'+label+'.json');path.write_text(json.dumps(cfg,indent=2),encoding='utf-8');paths.append(path)
    cli('compare','--configs',*paths,'--ids',DATASET_DIR/'validation_seed_ids.json',
        '--references',TRAIN_ROOT/'answers.jsonl','--output',SAVE/'comparison')
    print((SAVE/'comparison/comparison.json').read_text(encoding='utf-8'))
else:
    print('Đã bỏ qua merge/benchmark theo RUN_EVALUATION=False. Adapter và checkpoint ở',SAVE/'adapter')
''')
    cell('markdown','seed-finish','''
Giữ `adapter/`, `training_report.json`, `run_identity.json`, config, pip-freeze và comparison trên Drive. Source model, license, revision và dataset hash được lưu để truy nguyên. Chưa có kết quả fine-tune/PA được xác nhận trước khi notebook thực sự chạy.
''')
    notebook={'cells':cells,'metadata':{'colab':{'provenance':[]},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                      'language_info':{'name':'python'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
    path=root/'notebooks/training/DataMate_Colab_Seed72_100_QLoRA.ipynb'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for item in cells:
        if item['cell_type']=='code': compile(''.join(item['source']),item['id'],'exec')
    print(path)


if __name__=='__main__': main()
