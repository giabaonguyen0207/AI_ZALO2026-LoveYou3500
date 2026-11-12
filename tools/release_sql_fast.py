"""Seal v1.3 training kernels/resume changes; retain exact v1.2 SFT examples."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from datamate.models.artifacts import sha256
from datamate.training.sql_dataset import unpack_sql_archive
from build_sql_colab_fast import build


def encoded(value): return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
def digest(data): return hashlib.sha256(data).hexdigest()


def main():
    packages=ROOT/'outputs/packages'
    parent=json.loads((packages/'qwen35-9b-sql-sft-v1.2.release.json').read_text(encoding='utf-8'))
    for name in ('source_archive','dataset_archive'):
        assert sha256(packages/parent[name])==parent[name+'_sha256']
    with zipfile.ZipFile(packages/parent['source_archive']) as z:
        source={name:z.read(name) for name in z.namelist()}
    with zipfile.ZipFile(packages/parent['dataset_archive']) as z:
        data={name:z.read(name) for name in z.namelist()}
    manifest=json.loads(data.pop('manifest.json'))
    assert all(digest(data[name])==value for name,value in manifest['files'].items())
    allowed={'training/finetune.py','training/checkpoints.py','training/kernels.py'}
    for name,value in manifest['runtime_source_hashes'].items():
        assert digest(source['src/datamate/'+name])==value
        if name not in allowed: assert sha256(ROOT/'src/datamate'/name)==value, name
    provenance={'date':'2026-11-10','parent_manifest_sha256':parent['dataset_manifest_sha256'],
        'parent_source_archive_sha256':parent['source_archive_sha256'],
        'changes':['verified FLA training dispatch and explicit Triton conv adapter',
                   'full-state checksummed periodic checkpoints with automatic resume and incomplete-save fallback',
                   'same-runtime duplicate-run lock and completed-report reuse',
                   'local SSD staging; per-step timing; persistent logs and per-invocation environment'],
        'baseline_user_observation':{'gpu':'A100-SXM4-40GB','optimizer_steps':5,'sequences':40,
                                     'train_runtime_seconds':2601.4365,'fallback_warnings_observed':True},
        'all_training_examples_byte_identical_to_parent':True,'sql_runtime_changed':False,
        'sql_replay_repeated':False,'new_gpu_speedup_measured_at_packaging':False,
        'cpu_resume_verified_on_tiny_model_and_peft_adapter':True,
        'builder_sha256':sha256(Path(__file__)),'runtime_changed_files':{}}
    for name in sorted(allowed):
        content=(ROOT/'src/datamate'/name).read_bytes()
        provenance['runtime_changed_files'][name]={'before':manifest['runtime_source_hashes'].get(name),'after':digest(content)}
        source['src/datamate/'+name]=content
        manifest['runtime_source_hashes'][name]=digest(content)
    for name in ('tools/build_sql_colab.py','tools/build_sql_colab_fast.py','tools/release_sql_fast.py',
                 'configs/sql-qwen35-fast-requirements.txt','configs/sql-qwen35-9b-qlora.json','configs/sql-qwen35-9b-lora.json',
                 'tests/test_training_checkpoints.py','tests/test_training_kernels.py','tests/test_sql_colab_fast.py'):
        source[name]=(ROOT/name).read_bytes()
    data['training_compatibility.json']=encoded(provenance)
    data['README.md']+='\n\n## v1.3 — kernels và resume\nMọi mẫu học/tokenizer/context giữ nguyên từ v1.2. Cài các phiên bản GPU đã pin, giữ Torch/Triton của Colab; kiểm forward/backward kernel trước khi train. Lưu full Trainer checkpoint trên Drive ở bước 1, mỗi 5 bước và ngưỡng thời gian tại ranh giới optimizer step; giữ 3 bản. Chỉ resume checkpoint có marker/checksum khớp run. Không đảm bảo khôi phục phần chưa checkpoint. Chỉ một Colab server/RUN_TAG. Chưa đo speedup GPU mới ở thời điểm đóng gói.\n'.encode('utf-8')
    manifest.update(release='qwen35-9b-sql-sft-v1.3',created_date='2026-11-10',
                    training_compatibility=provenance,files={name:digest(value) for name,value in data.items()})
    data['manifest.json']=encoded(manifest)
    data_name='qwen35-9b-sql-sft-v1.3.zip'
    source_name='datamate-qwen35-9b-sql-source-v1.3.zip'
    notebook=ROOT/'notebooks/training/DataMate_Colab_Qwen35_9B_SQL_SFT_v1_3.ipynb'
    release_path=packages/'qwen35-9b-sql-sft-v1.3.release.json'
    assert not any(p.exists() for p in (packages/data_name,packages/source_name,notebook,release_path)), 'Do not overwrite release'
    with zipfile.ZipFile(packages/parent['dataset_archive']) as z:
        for name in ('train.jsonl','validation.jsonl','inventory.jsonl','repair_queue.jsonl','train_ids.json',
                     'validation_ids.json','tokenization_report.json'):
            assert data[name]==z.read(name)
    with zipfile.ZipFile(packages/data_name,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,value in sorted(data.items()): z.writestr(name,value)
    release={'dataset_archive':data_name,'dataset_archive_sha256':sha256(packages/data_name),
             'dataset_manifest_sha256':digest(data['manifest.json']),'target_model':parent['target_model']}
    source['configs/qwen35-9b-sql-release.json']=encoded(release)
    source['training_compatibility.json']=encoded(provenance)
    with zipfile.ZipFile(packages/source_name,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,value in sorted(source.items()): z.writestr(name,value)
    build(notebook,sha256(packages/source_name),source_name,data_name)
    release.update(source_archive=source_name,source_archive_sha256=sha256(packages/source_name),
                   notebook_sha256=sha256(notebook),counts=parent['counts'],token_statistics=parent['token_statistics'],
                   training_compatibility=provenance)
    release_path.write_bytes(encoded(release))
    out=ROOT/'outputs/runs/review/sql-sft-qwen35-9b-20261110-speed'
    out.mkdir(parents=True,exist_ok=True)
    unpack_sql_archive(packages/data_name,out/'dataset-v1.3',release_path)
    for archive in (packages/data_name,packages/source_name):
        with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    print(json.dumps({name:release[name] for name in ('source_archive','source_archive_sha256','dataset_archive','counts')},indent=2))


if __name__=='__main__': main()
