"""Package a verified Trainer LoRA snapshot; merge only on the Colab host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import uuid
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'src'),str(PROJECT)]
from datamate.domain import DataMateError, json_text
from datamate.models.artifacts import sha256, verify_artifact
from datamate.models.registry import get_model
from datamate.training.checkpoints import verify_checkpoint, atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local(root, name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise DataMateError('Unsafe checkpoint path: '+name)
    result = root.joinpath(*path.parts)
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise DataMateError('Checkpoint symlink/path outside package: '+name)
    return result


def package_checkpoint(checkpoint, output, *, model_key='qwen35-9b', reported_total_epochs=None):
    checkpoint,output = Path(checkpoint).resolve(),Path(output).resolve()
    if output.exists() or output.is_relative_to(checkpoint):
        raise DataMateError('Use a new ZIP path outside the original checkpoint')
    marker = read(checkpoint/'checkpoint_complete.json')
    verify_checkpoint(checkpoint,marker['run_identity'])  # Hashes only; never unpickle/load weights.
    identity = marker['run_identity']
    spec = get_model(model_key)
    if identity.get('training',{}).get('model_key') != model_key:
        raise DataMateError('Checkpoint model_key differs from requested base')
    if read(checkpoint/'adapter_config.json').get('peft_type') != 'LORA':
        raise DataMateError('Expected a LoRA checkpoint')
    if not all((checkpoint/name).is_file() for name in ('tokenizer.json','tokenizer_config.json')):
        raise DataMateError('Checkpoint lacks its tokenizer; do not substitute a different tokenizer')
    state = read(checkpoint/'trainer_state.json')
    source = {'name':checkpoint.name,'global_step':marker['global_step'],'stage_epoch':state.get('epoch'),
              'marker_sha256':sha256(checkpoint/'checkpoint_complete.json')}
    retained = [name for name in marker['files'] if name in {
        'adapter_config.json','adapter_model.safetensors','trainer_state.json','README.md',
        'tokenizer.json','tokenizer_config.json','chat_template.jinja','special_tokens_map.json',
        'added_tokens.json','vocab.json','merges.txt','sentencepiece.bpe.model','tokenizer.model'}]
    report = {'kind':'trainer_checkpoint_snapshot','base_source':spec.to_dict(),'run_identity':identity,
              'source_checkpoint':source,'training_finalization_verified':False,
              'training_config':identity['training'],'training_manifest_sha256':identity['dataset_manifest_sha256'],
              'reported_total_epochs':reported_total_epochs,'total_epochs_independently_verified':False,
              'adapter_files':{name:marker['files'][name] for name in retained}}
    # Existing exporter accepts this explicit snapshot report; it is not a claim
    # that Trainer completed/finalized the parent or continuation run.
    report_bytes = json_text(report,indent=2).encode('utf-8')
    import hashlib
    files = {'adapter/'+name:marker['files'][name] for name in retained}
    files['adapter/training_report.json'] = hashlib.sha256(report_bytes).hexdigest()
    files['source_checkpoint_complete.json'] = source['marker_sha256']
    manifest = {'format_version':1,'model_key':model_key,'base_source':spec.to_dict(),
                'source_global_step':marker['global_step'],'source_stage_epoch':state.get('epoch'),
                'expected_base_manifest_sha256':identity['base_manifest_sha256'],
                'reported_total_epochs':reported_total_epochs,'total_epochs_independently_verified':False,'files':files}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name+'.tmp-'+uuid.uuid4().hex[:12])
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in retained:
            archive.write(checkpoint/name,'adapter/'+name)
        archive.write(checkpoint/'checkpoint_complete.json','source_checkpoint_complete.json')
        archive.writestr('adapter/training_report.json',report_bytes)
        archive.writestr('checkpoint_inference.json',json_text(manifest,indent=2))
    verify_checkpoint(checkpoint,identity)  # Refuse a source changing while packaging.
    temporary.replace(output)
    atomic_json(output.with_suffix(output.suffix+'.manifest.json'),{**manifest,'archive_sha256':sha256(output)})
    return manifest


def verify_package(root):
    root = Path(root).resolve()
    manifest = read(root/'checkpoint_inference.json')
    if manifest.get('format_version')!=1 or manifest.get('base_source')!=get_model(manifest['model_key']).to_dict():
        raise DataMateError('Checkpoint package source/format differs from registry')
    expected = set(manifest['files'])|{'checkpoint_inference.json'}
    if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} != expected:
        raise DataMateError('Checkpoint package contains missing/extra files')
    for name,digest in manifest['files'].items():
        if sha256(local(root,name))!=digest:
            raise DataMateError('Checkpoint package checksum mismatch: '+name)
    marker = read(root/'source_checkpoint_complete.json')
    if marker['run_identity']['base_manifest_sha256']!=manifest['expected_base_manifest_sha256']:
        raise DataMateError('Checkpoint base identity mismatch')
    for name,digest in marker['files'].items():
        exported = 'adapter/'+name
        if exported in manifest['files'] and digest!=manifest['files'][exported]:
            raise DataMateError('Package differs from original checkpoint marker: '+name)
    return manifest


def extract_package(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as packet:
        manifest_bytes = packet.read('checkpoint_inference.json')
        manifest = json.loads(manifest_bytes)
        names = set(manifest['files'])|{'checkpoint_inference.json'}
        if set(packet.namelist())!=names or len(packet.namelist())!=len(names):
            raise DataMateError('Unexpected/duplicate ZIP entries')
        if destination.exists() and any(destination.iterdir()):
            verify_package(destination)
            if (destination/'checkpoint_inference.json').read_bytes()!=manifest_bytes:
                raise DataMateError('Extraction folder contains another checkpoint; choose a new folder')
            return destination
        for info in packet.infolist():
            target = local(destination,info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise DataMateError('ZIP symlink is not allowed')
            target.parent.mkdir(parents=True,exist_ok=True)
            with packet.open(info) as src,target.open('xb') as dst:
                shutil.copyfileobj(src,dst,length=4*1024**2)
    verify_package(destination)
    return destination


def merge_package(root, base, output):
    root,base,output = Path(root).resolve(),Path(base).resolve(),Path(output).resolve()
    manifest = verify_package(root)
    key = manifest['model_key']
    report_hash = sha256(root/'adapter/training_report.json')
    if output.exists() and any(output.iterdir()):
        sealed = verify_artifact(output,key)
        lineage = sealed.get('lineage') or {}
        if (lineage.get('training_report_sha256')!=report_hash
                or lineage.get('base_manifest_sha256')!=manifest['expected_base_manifest_sha256']):
            raise DataMateError('Merged output belongs to another checkpoint/base; use a new output')
        return {'output':str(output),'reused':True}
    verify_artifact(base,key)
    if sha256(base/'artifact_manifest.json')!=manifest['expected_base_manifest_sha256']:
        raise DataMateError('Base manifest differs from the exact base used in training; select the original sealed base on Drive')
    from tools.merge_sql_export import merge_verified
    result = merge_verified(key,base,root/'adapter',output)
    verify_artifact(output,key)
    return {**result,'reused':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    package = sub.add_parser('package')
    package.add_argument('--checkpoint',required=True,type=Path)
    package.add_argument('--output',required=True,type=Path)
    package.add_argument('--reported-total-epochs',type=float)
    unpack = sub.add_parser('extract')
    unpack.add_argument('--archive',required=True,type=Path)
    unpack.add_argument('--output',required=True,type=Path)
    merge = sub.add_parser('merge',help='Run in a fresh Colab CPU child with sufficient High RAM')
    for flag in ('package-root','base','output'):
        merge.add_argument('--'+flag,required=True,type=Path)
    args = parser.parse_args()
    if args.command=='package':
        result = package_checkpoint(args.checkpoint,args.output,reported_total_epochs=args.reported_total_epochs)
    elif args.command=='extract':
        result = {'extracted':str(extract_package(args.archive,args.output))}
    else:
        if not (args.output/'artifact_manifest.json').is_file():
            from tools.merge_sql100_cpu import cpu_resources,check_cpu_ram
            resources = cpu_resources()
            check_cpu_ram(get_model(verify_package(args.package_root)['model_key']).parameters,resources['available_ram_bytes'])
        result = merge_package(args.package_root,args.base,args.output)
    print(json_text(result,indent=2))


if __name__=='__main__':
    main()
