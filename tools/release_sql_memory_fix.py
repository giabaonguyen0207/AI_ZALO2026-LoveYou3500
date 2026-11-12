"""Release training-only memory/logging fixes with unchanged v1.1 examples."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from datamate.models.artifacts import sha256
from datamate.training.sql_dataset import unpack_sql_archive
from build_sql_colab import build


def digest(data): return hashlib.sha256(data).hexdigest()
def encoded(obj): return (json.dumps(obj, ensure_ascii=False, indent=2)+'\n').encode('utf-8')


def main():
    packages = ROOT/'outputs/packages'
    parent = json.loads((packages/'qwen35-9b-sql-sft-v1.1.release.json').read_text(encoding='utf-8'))
    for name in ('source_archive','dataset_archive'):
        assert sha256(packages/parent[name]) == parent[name+'_sha256']
    with zipfile.ZipFile(packages/parent['source_archive']) as z:
        source = {name:z.read(name) for name in z.namelist()}
    with zipfile.ZipFile(packages/parent['dataset_archive']) as z:
        data = {name:z.read(name) for name in z.namelist()}
    manifest = json.loads(data.pop('manifest.json'))
    assert all(digest(data[name]) == value for name,value in manifest['files'].items())
    allowed = {'training/finetune.py','training/memory.py'}
    for name,value in manifest['runtime_source_hashes'].items():
        assert digest(source['src/datamate/'+name]) == value
        if name not in allowed: assert sha256(ROOT/'src/datamate'/name) == value
    provenance = {'date':'2026-11-10','parent_manifest_sha256':parent['dataset_manifest_sha256'],
        'parent_source_archive_sha256':parent['source_archive_sha256'],
        'reason':'Observed CUDA OOM on L4 22.03 GiB during longest-sequence forward',
        'changes':['restore only frozen input/output weights to original BF16 compute dtype after PEFT preparation',
                   'compute logits only from predecessor of first supervised label, preserving the entire prompt and causal loss',
                   'stream notebook child stdout/stderr to visible output and persistent logs',
                   'preserve failed smoke directories and retry separately'],
        'all_training_examples_byte_identical_to_parent':True,'sql_runtime_changed':False,
        'sql_replay_repeated':False,'gpu_fit_verified_at_packaging':False,
        'builder_sha256':sha256(Path(__file__)),
        'runtime_changed_files':{}}
    for name in sorted(allowed):
        content=(ROOT/'src/datamate'/name).read_bytes()
        provenance['runtime_changed_files'][name]={'before':manifest['runtime_source_hashes'].get(name),'after':digest(content)}
        source['src/datamate/'+name]=content
        manifest['runtime_source_hashes'][name]=digest(content)
    source['tools/build_sql_colab.py']=(ROOT/'tools/build_sql_colab.py').read_bytes()
    source['tools/release_sql_memory_fix.py']=Path(__file__).read_bytes()
    data['training_compatibility.json']=encoded(provenance)
    data['README.md']+='\n\n## v1.2 — sửa bộ nhớ L4\nGiữ nguyên mọi mẫu/tokenizer/context từ v1.1. Tối ưu weights IO frozen và suffix logits trong trainer, giữ nguyên loss trên assistant. Notebook ghi log stdout/stderr và giữ riêng các lần smoke lỗi. GPU fit cần kiểm bằng smoke thực tế; xem training_compatibility.json.\n'.encode('utf-8')
    manifest.update(release='qwen35-9b-sql-sft-v1.2',created_date='2026-11-10',training_compatibility=provenance,
                    files={name:digest(value) for name,value in data.items()})
    data['manifest.json']=encoded(manifest)
    data_name='qwen35-9b-sql-sft-v1.2.zip'
    source_name='datamate-qwen35-9b-sql-source-v1.2.zip'
    notebook=ROOT/'notebooks/training/DataMate_Colab_Qwen35_9B_SQL_SFT_v1_2.ipynb'
    release_path=packages/'qwen35-9b-sql-sft-v1.2.release.json'
    assert not any(path.exists() for path in (packages/data_name,packages/source_name,notebook,release_path)), 'Do not overwrite releases'
    with zipfile.ZipFile(packages/parent['dataset_archive']) as z:
        for name in ('train.jsonl','validation.jsonl','inventory.jsonl','repair_queue.jsonl',
                     'train_ids.json','validation_ids.json','tokenization_report.json'):
            assert data[name] == z.read(name)
    with zipfile.ZipFile(packages/data_name,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,value in sorted(data.items()): z.writestr(name,value)
    release={'dataset_archive':data_name,'dataset_archive_sha256':sha256(packages/data_name),
             'dataset_manifest_sha256':digest(data['manifest.json']),'target_model':parent['target_model']}
    source['configs/qwen35-9b-sql-release.json']=encoded(release)
    source['training_compatibility.json']=encoded(provenance)
    with zipfile.ZipFile(packages/source_name,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,value in sorted(source.items()): z.writestr(name,value)
    build(notebook,sha256(packages/source_name),source_name,data_name)
    nb=json.loads(notebook.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        cell['source']=[s.replace('qwen35-9b-sql-v1-pilot-01','qwen35-9b-sql-v1.2-pilot-01') for s in cell['source']]
    notebook.write_bytes(encoded(nb))
    release.update(source_archive=source_name,source_archive_sha256=sha256(packages/source_name),
                   notebook_sha256=sha256(notebook),counts=parent['counts'],token_statistics=parent['token_statistics'],
                   training_compatibility=provenance)
    release_path.write_bytes(encoded(release))
    out=ROOT/'outputs/runs/review/sql-sft-qwen35-9b-20261110-l4-fix'
    out.mkdir(parents=True,exist_ok=True)
    unpack_sql_archive(packages/data_name,out/'dataset-v1.2',release_path)
    for path in (packages/data_name,packages/source_name):
        with zipfile.ZipFile(path) as z: assert z.testzip() is None
    print(json.dumps({k:release[k] for k in ('source_archive','source_archive_sha256','dataset_archive','counts')},indent=2))


if __name__=='__main__': main()
