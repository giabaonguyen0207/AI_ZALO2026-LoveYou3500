"""Package the user's verified full adapter evaluation; never load full weights."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from datamate.data.split import read_jsonl
from datamate.models.artifacts import sha256
from datamate.training.checkpoints import completed_report,verify_checkpoint
from tools.colab_archives import verify_train_directory
from tools.sql100 import select_questions,verify_benchmark
from tools.build_sql100_colab import build


def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
def digest(value):return hashlib.sha256(value).hexdigest()


def main():
    packages=ROOT/'outputs/packages'
    parent=json.loads((packages/'qwen35-9b-sql-sft-v1.3.release.json').read_text(encoding='utf-8'))
    assert sha256(packages/parent['source_archive'])==parent['source_archive_sha256']
    assert sha256(packages/parent['dataset_archive'])==parent['dataset_archive_sha256']
    run=ROOT/'outputs/runs/Bảo/qwen35-9b-sql-v1.3-full-01'
    config=json.loads((run/'training_config.json').read_text(encoding='utf-8'))
    report=completed_report(run/'adapter',config)
    assert report['global_step']==298 and report['metrics']['epoch']==1.0
    verify_checkpoint(run/'adapter/checkpoint-298',report['run_identity'])
    root=ROOT/'train/train'
    provenance=verify_train_directory(root,ROOT/'configs/training_sources.json')
    with zipfile.ZipFile(packages/parent['dataset_archive']) as z:
        dataset_manifest=json.loads(z.read('manifest.json'))
        validation_ids=json.loads(z.read('validation_ids.json'))
        train_ids=json.loads(z.read('train_ids.json'))
    assert provenance['fingerprint']==dataset_manifest['source_fingerprint']
    assert not set(validation_ids)&set(train_ids) and len(validation_ids)==630
    questions=read_jsonl(root/'questions.jsonl')
    selected=select_questions(questions,validation_ids,100,42)
    ids=[q['id'] for q in selected];businesses={q['business_id'] for q in selected}
    assert businesses<=set(dataset_manifest['validation_businesses'])
    references={r['id']:r for r in read_jsonl(root/'answers.jsonl')}
    files={'input/questions.jsonl':''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in selected).encode('utf-8'),
           'question_ids.json':encoded(ids),
           'evaluation/references.jsonl':''.join(json.dumps(references[i],ensure_ascii=False)+'\n' for i in ids).encode('utf-8')}
    for name in ('rulebook.md','calendar.csv'):files['input/'+name]=(root/name).read_bytes()
    for business in sorted(businesses):
        for p in sorted((root/'data'/business).rglob('*')):
            if p.is_file():
                assert p.resolve().is_relative_to((root/'data'/business).resolve())
                files['input/'+p.relative_to(root).as_posix()]=p.read_bytes()
    manifest={'benchmark_id':'qwen35-sql-v1.3-validation100-seed42','date':'2026-11-10','sample_size':100,
        'seed':42,'sampling':'Proportional business quotas (largest remainder), SHA256(seed:id) ranking; no label/outcome selection',
        'validation_pool_size':630,'business_counts':dict(sorted(collections.Counter(q['business_id'] for q in selected).items())),
        'reference_type_counts':dict(collections.Counter(references[i]['type'] for i in ids)),
        'source_fingerprint':provenance['fingerprint'],'parent_dataset_manifest_sha256':parent['dataset_manifest_sha256'],
        'development_validation_already_inspected':True,'train_overlap_count':len(set(ids)&set(train_ids)),
        'files':{name:digest(content) for name,content in files.items()}}
    files['benchmark_manifest.json']=encoded(manifest)
    source_name='datamate-qwen35-sql-validation100-source-v1.zip'
    data_name='qwen35-sql-validation100-data-v1.zip'
    notebook=ROOT/'notebooks/training/DataMate_Colab_Qwen35_SQL_Validation100_v1.ipynb'
    release_path=packages/'qwen35-sql-validation100-v1.release.json'
    assert not any(p.exists() for p in (packages/source_name,packages/data_name,notebook,release_path)), 'Do not overwrite release'
    with zipfile.ZipFile(packages/data_name,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in sorted(files.items()):z.writestr(name,content)
    adapter_files={name:sha256(run/'adapter'/name) for name in
        ('adapter_config.json','adapter_model.safetensors','training_report.json','tokenizer.json','tokenizer_config.json','chat_template.jinja')}
    release={'benchmark_id':manifest['benchmark_id'],'sample_size':100,'data_archive':data_name,
             'data_sha256':sha256(packages/data_name),'benchmark_manifest_sha256':digest(files['benchmark_manifest.json']),
             'adapter_files':adapter_files,'base_manifest_sha256':report['run_identity']['base_manifest_sha256'],
             'parent_source_archive_sha256':parent['source_archive_sha256'],
             'inference_runtime_unchanged_from_training_release':True,'gpu_comparison_run':False}
    with zipfile.ZipFile(packages/parent['source_archive']) as z:
        source={name:z.read(name) for name in z.namelist()}
    for name,content in source.items():
        if name.startswith('src/datamate/') and name.endswith('.py'):
            assert (ROOT/name).read_bytes()==content, 'Workspace runtime differs from frozen training source: '+name
    for name,expected in dataset_manifest['runtime_source_hashes'].items():
        assert digest(source['src/datamate/'+name])==expected
    for name in ('tools/sql100.py','tools/build_sql100_colab.py','tools/release_sql100.py','tests/test_sql100.py',
                 'docs/sql-validation100-colab.md'):
        source[name]=(ROOT/name).read_bytes()
    source['configs/sql100-release.json']=encoded(release)
    with zipfile.ZipFile(packages/source_name,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in sorted(source.items()):z.writestr(name,content)
    build(notebook,sha256(packages/source_name),source_name,data_name)
    release.update(source_archive=source_name,source_sha256=sha256(packages/source_name),
                   notebook_sha256=sha256(notebook),business_counts=manifest['business_counts'],
                   reference_type_counts=manifest['reference_type_counts'])
    release_path.write_bytes(encoded(release))
    out=ROOT/'outputs/runs/review/sql-validation100-20261110';out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(packages/data_name) as z:z.extractall(out/'benchmark')
    verify_benchmark(out/'benchmark')
    for p in (packages/source_name,packages/data_name):
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
    (out/'artifact-verification.json').write_bytes(encoded({
        'received_date':'2026-11-10','source':str(run.relative_to(ROOT)),
        'full_adapter_checksums_verified':True,'checkpoint298_full_state_verified':True,
        'checkpoint295_missing_file':'optimizer.pt (downloaded copy only; unused for inference)',
        'benchmark_size':100,'train_overlap_count':0,'validation_pool_size':630,
        'gpu_merge_run_locally':False,'gpu_inference_run_locally':False,
        'release':release}))
    print(json.dumps(release,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
