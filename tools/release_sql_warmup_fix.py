"""Derive SQL SFT v1.1 from sealed v1 for the single HF warmup API fix.

No relabeling/replay: refuse any other runtime change, preserve every v1
training example byte, and retain the original generation lineage explicitly.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from datamate.models.artifacts import sha256
from datamate.training.sql_dataset import unpack_sql_archive


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    packages = ROOT / 'outputs/packages'
    parent = json.loads((packages / 'qwen35-9b-sql-sft-v1.release.json').read_text(encoding='utf-8'))
    for key in ('source_archive', 'dataset_archive'):
        assert sha256(packages / parent[key]) == parent[key + '_sha256']
    with zipfile.ZipFile(packages / parent['source_archive']) as archive:
        source = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(packages / parent['dataset_archive']) as archive:
        data = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(data.pop('manifest.json'))
    assert all(digest(data[name]) == value for name, value in manifest['files'].items())
    name = 'src/datamate/training/finetune.py'
    before, after = b'warmup_ratio=0.03', b'warmup_steps=0.03'
    assert source[name].count(before) == 1
    fixed = source[name].replace(before, after)
    assert (ROOT / name).read_bytes() == fixed, 'Only the reviewed one-keyword trainer fix is allowed'
    for module, value in manifest['runtime_source_hashes'].items():
        path = ROOT / 'src/datamate' / module
        assert digest(source['src/datamate/' + module]) == value
        if module != 'training/finetune.py':
            assert sha256(path) == value, 'Unrelated runtime change requires separate review/release'
    source[name] = fixed
    provenance = {
        'date': '2026-11-10', 'fix': 'warmup_ratio=0.03 -> warmup_steps=0.03',
        'reason': 'Transformers 5 TrainingArguments rejects removed warmup_ratio keyword',
        'warmup_ratio_preserved': 0.03,
        'parent_manifest_sha256': parent['dataset_manifest_sha256'],
        'parent_source_archive_sha256': parent['source_archive_sha256'],
        'runtime_changed_files': {'training/finetune.py': {
            'before': manifest['runtime_source_hashes']['training/finetune.py'], 'after': digest(fixed)}},
        'all_training_examples_byte_identical_to_parent': True,
        'sql_runtime_changed': False, 'sql_replay_repeated': False,
        'gpu_training_performed_for_fix_verification': False,
        'cpu_api_regression_test_transformers': '5.22.2',
        'test': 'tests/test_training_arguments_compat.py',
        'builder_sha256': sha256(Path(__file__)),
    }
    data['training_compatibility.json'] = json.dumps(provenance, ensure_ascii=False, indent=2).encode('utf-8')
    data['README.md'] += ('\n\n## v1.1 — 10/11/2026\n'
        'Sửa API warmup trong trainer cho Transformers 5. Mọi mẫu train/validation, token report và ID giữ nguyên byte từ v1. '
        'Manifest cập nhật hash đúng file trainer đã sửa và giữ nguồn generation/replay v1; không replay SQL mới. '
        'Dùng source ZIP và notebook v1.1 cùng dataset này. Xem training_compatibility.json.\n').encode('utf-8')
    manifest.update(release='qwen35-9b-sql-sft-v1.1', created_date='2026-11-10',
                    training_compatibility=provenance,
                    files={key: digest(value) for key, value in data.items()})
    manifest['runtime_source_hashes']['training/finetune.py'] = digest(fixed)
    data['manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    data_name = 'qwen35-9b-sql-sft-v1.1.zip'
    source_name = 'datamate-qwen35-9b-sql-source-v1.1.zip'
    notebook = ROOT / 'notebooks/training/DataMate_Colab_Qwen35_9B_SQL_SFT_v1_1.ipynb'
    release_path = packages / 'qwen35-9b-sql-sft-v1.1.release.json'
    assert not any(p.exists() for p in (packages/data_name, packages/source_name, notebook, release_path)), 'Do not overwrite a release'
    for key in ('train.jsonl', 'validation.jsonl', 'tokenization_report.json', 'inventory.jsonl',
                'repair_queue.jsonl', 'train_ids.json', 'validation_ids.json'):
        assert digest(data[key]) == manifest['training_compatibility'].get('parent_files', {}).get(key, manifest['files'][key])
        # Compare to sealed parent, independently of the updated manifest.
        with zipfile.ZipFile(packages / parent['dataset_archive']) as archive:
            assert data[key] == archive.read(key)
    with zipfile.ZipFile(packages/data_name, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in sorted(data.items()): archive.writestr(key, value)
    release = {
        'dataset_archive': data_name, 'dataset_archive_sha256': sha256(packages/data_name),
        'dataset_manifest_sha256': digest(data['manifest.json']), 'target_model': parent['target_model'],
    }
    source['configs/qwen35-9b-sql-release.json'] = json.dumps(release, ensure_ascii=False, indent=2).encode('utf-8')
    source['training_compatibility.json'] = data['training_compatibility.json']
    with zipfile.ZipFile(packages/source_name, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for key, value in sorted(source.items()): archive.writestr(key, value)
    # Use exactly the parent's notebook builder, with only new artifact names/hash/tag.
    namespace = {'__name__': 'sealed_notebook_builder'}
    exec(compile(source['tools/build_sql_colab.py'], '<sealed notebook builder>', 'exec'), namespace)
    namespace['build'](notebook, sha256(packages/source_name), source_name, data_name)
    nb = json.loads(notebook.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        text = ''.join(cell['source']).replace('qwen35-9b-sql-v1-pilot-01', 'qwen35-9b-sql-v1.1-pilot-01')
        cell['source'] = text.splitlines(True)
        if cell['cell_type'] == 'code': ast.parse(text)
    notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    release.update(source_archive=source_name, source_archive_sha256=sha256(packages/source_name),
                   notebook_sha256=sha256(notebook), counts=parent['counts'],
                   token_statistics=parent['token_statistics'], training_compatibility=provenance)
    release_path.write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding='utf-8')
    output = ROOT / 'outputs/runs/review/sql-sft-qwen35-9b-20261110-compat'
    output.mkdir(parents=True, exist_ok=True)
    config = output/'release.json'
    config.write_text(json.dumps(release), encoding='utf-8')
    unpack_sql_archive(packages/data_name, output/'dataset-v1.1', config)
    for path in (packages/data_name, packages/source_name):
        with zipfile.ZipFile(path) as archive: assert archive.testzip() is None
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
