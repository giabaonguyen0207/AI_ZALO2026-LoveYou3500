"""Derive the L4 orchestration release without changing frozen inputs/runtime."""
import ast
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from datamate.models.artifacts import sha256
from tools.build_sql100_colab import build


def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def main(*, selectable_quantization=False, mtp_export_fix=False, default_quantization='4bit'):
    if mtp_export_fix:selectable_quantization=True
    packages=ROOT/'outputs/packages'
    parent=json.loads((packages/'qwen35-sql-validation100-v1.release.json').read_text(encoding='utf-8'))
    assert sha256(packages/parent['source_archive'])==parent['source_sha256']
    assert sha256(packages/parent['data_archive'])==parent['data_sha256']
    with zipfile.ZipFile(packages/parent['source_archive']) as z:
        source={name:z.read(name) for name in z.namelist()}
    release=json.loads(source['configs/sql100-release.json'])
    release.update(profile='l4_nf4_cpu_merge',inference_quantization='4bit',minimum_vram_gib_policy=22,
                   cpu_merge_minimum_available_ram_gib_policy=32,
                   parent_evaluation_source_sha256=parent['source_sha256'],gpu_comparison_run=False,
                   data_byte_identical_to_v1=True,inference_runtime_unchanged_from_v1=True)
    for name in ('tools/build_sql100_colab.py','tools/merge_sql100_cpu.py','tools/merge_sql_export.py','tools/release_sql100_l4.py',
                 'tests/test_sql100_l4.py','docs/sql-validation100-l4-colab.md'):
        source[name]=(ROOT/name).read_bytes()
    if selectable_quantization:
        release.update(profile='selectable_quantization_cpu_merge',inference_quantization='selectable',
                       default_quantization=default_quantization,quantization_options=['none','8bit','4bit'])
        for name in ('tests/test_sql100_quant.py','docs/sql-validation100-quant-colab.md'):
            source[name]=(ROOT/name).read_bytes()
    if mtp_export_fix:
        release.update(auxiliary_export_fix='Preserve original missing mtp.* tensors; active keys/shapes and parameter count remain strict')
        for name in ('tools/merge_sql_export.py','tests/test_merge_sql_export.py','docs/sql100-mtp-export-fix.md'):
            source[name]=(ROOT/name).read_bytes()
    source['configs/sql100-release.json']=encoded(release)
    version='1.3' if mtp_export_fix else ('1.2' if selectable_quantization else '1.1')
    source_name=f'datamate-qwen35-sql-validation100-source-l4-v{version}.zip'
    path=packages/source_name
    notebook=ROOT/('notebooks/training/DataMate_Colab_Qwen35_SQL_Validation100_L4_v'+version.replace('.','_')+'.ipynb')
    release_path=packages/f'qwen35-sql-validation100-l4-v{version}.release.json'
    assert not any(p.exists() for p in (path,notebook,release_path)), 'Do not overwrite release'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in sorted(source.items()):z.writestr(name,content)
    build(notebook,sha256(path),source_name,parent['data_archive'],profile='l4',
          selectable_quantization=selectable_quantization,default_quantization=default_quantization)
    for c in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    release.update(source_archive=source_name,source_sha256=sha256(path),notebook_sha256=sha256(notebook),
                   business_counts=parent['business_counts'],reference_type_counts=parent['reference_type_counts'])
    release_path.write_bytes(encoded(release))
    with zipfile.ZipFile(path) as new,zipfile.ZipFile(packages/parent['source_archive']) as old:
        assert new.testzip() is None
        for name in old.namelist():
            if name.startswith('src/datamate/') or name=='tools/sql100.py':
                assert old.read(name)==new.read(name),name
    print(json.dumps({k:release[k] for k in ('source_archive','source_sha256','data_archive','data_sha256','profile')},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--selectable-quantization',action='store_true')
    parser.add_argument('--mtp-export-fix',action='store_true')
    parser.add_argument('--default-quantization',choices=['none','8bit','4bit'],default='4bit')
    args=parser.parse_args()
    main(selectable_quantization=args.selectable_quantization,mtp_export_fix=args.mtp_export_fix,
         default_quantization=args.default_quantization)
