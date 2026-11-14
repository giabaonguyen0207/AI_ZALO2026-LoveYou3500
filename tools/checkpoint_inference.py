"""Package a verified LoRA snapshot or completed adapter; merge on Colab only."""
from __future__ import annotations

import argparse
import json
import math
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
from datamate.training.checkpoints import verify_checkpoint, atomic_json, completed_report, build_run_identity

INFERENCE_FILES = {
    'adapter_config.json','adapter_model.safetensors','trainer_state.json','README.md',
    'tokenizer.json','tokenizer_config.json','chat_template.jinja','special_tokens_map.json',
    'added_tokens.json','vocab.json','merges.txt','sentencepiece.bpe.model','tokenizer.model'}


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
    retained = [name for name in marker['files'] if name in INFERENCE_FILES]
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


def _completed_adapter(run,model_key):
    run = Path(run).resolve()
    adapter = run if (run/'training_report.json').is_file() else run/'adapter'
    try:
        report = read(adapter/'training_report.json')
        config = report['training_config']
        completed_report(adapter,config)  # Hash the saved adapter, never load it.
        identity = report['run_identity']
        # A continuation's initial_adapter is a relocatable path. Its immutable
        # initial_adapter_files must remain in the normalized training identity.
        normalized = build_run_identity({k:v for k,v in config.items() if k!='initial_adapter'},
                                        identity['base_manifest_sha256'],identity['dataset_manifest_sha256'])
        if (identity.get('version')!=normalized['version'] or identity['training']!=normalized['training']
                or report['training_manifest_sha256']!=identity['dataset_manifest_sha256']):
            raise DataMateError('Training config/report differs from normalized run identity')
        if (report['base_source']!=get_model(model_key).to_dict()
                or config.get('model_key')!=model_key
                or report['run_identity']['training'].get('model_key')!=model_key):
            raise DataMateError('Completed adapter differs from selected model/revision')
        if (adapter/'run_identity.json').is_file() and read(adapter/'run_identity.json')!=report['run_identity']:
            raise DataMateError('Adapter run_identity differs from completed report')
        config_path = adapter.parent/'training_config.json'
        if config_path.is_file() and read(config_path)!=config:
            raise DataMateError('Run config differs from completed training report')
        if (config.get('smoke_longest_only') or config.get('max_steps',-1)>0
                or read(adapter/'adapter_config.json').get('peft_type')!='LORA'):
            raise DataMateError('Select the completed full-epoch LoRA adapter, not smoke/pilot')
        if set(report['adapter_files'])!={'adapter_config.json','adapter_model.safetensors'}:
            raise DataMateError('Completed report must pin config and safetensors adapter')
        _epoch_entry(report,sha256(adapter/'training_report.json'))
        if not all((adapter/n).is_file() for n in ('tokenizer.json','tokenizer_config.json')):
            raise DataMateError('Completed adapter lacks its saved tokenizer')
        return adapter,report
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise DataMateError('Invalid completed adapter/report: '+str(run)) from exc


def _epoch_entry(report,digest):
    requested = report['training_config']['epochs']
    observed = report['metrics']['epoch']
    if (type(requested) not in (int,float) or type(observed) not in (int,float)
            or not math.isfinite(requested) or not math.isfinite(observed)
            or requested<=0 or observed+1e-8<requested
            or type(report['global_step']) is not int or report['global_step']<1):
        raise DataMateError('Report does not show the requested epoch run finished')
    return {'training_report_sha256':digest,'global_step':report['global_step'],
            'requested_epochs':requested,'observed_epoch':observed}


def _completed_summary(report,report_hash,parent=None,parent_hash=None):
    entry = _epoch_entry(report,report_hash)
    chain = [entry]
    verified = parent is not None
    if verified:
        expected = report['training_config'].get('initial_adapter_files',{})
        actual = {**parent['adapter_files'],'training_report.json':parent_hash}
        if (set(expected)!={'adapter_config.json','adapter_model.safetensors','training_report.json'}
                or expected!=actual or report.get('initial_adapter',{}).get('adapter_files')!=expected
                or report['run_identity']['training'].get('initial_adapter_files')!=expected
                or report['initial_adapter'].get('parent_global_step')!=parent['global_step']
                or parent['base_source']!=report['base_source']
                or parent['run_identity']['base_manifest_sha256']!=report['run_identity']['base_manifest_sha256']):
            raise DataMateError('Parent adapter/report does not match continuation lineage')
        if parent['training_config'].get('initial_adapter_files'):
            raise DataMateError('Parent itself is a continuation; supply the full chain before claiming total epochs')
        chain.insert(0,_epoch_entry(parent,parent_hash))
    return {'source_kind':'completed_adapter','training_finalization_verified':True,
            'source_global_step':report['global_step'],'source_stage_epoch':entry['observed_epoch'],
            'expected_base_manifest_sha256':report['run_identity']['base_manifest_sha256'],
            'source_training_report_sha256':report_hash,'verified_epoch_chain':chain,
            'total_epochs_independently_verified':verified,
            'verified_requested_total_epochs':sum(e['requested_epochs'] for e in chain) if verified else None}


def package_completed_adapter(run,output,*,parent_adapter=None,model_key='qwen35-9b'):
    """Keep the original final report, optionally verify the epoch-1 parent."""
    run,output = Path(run).resolve(),Path(output).resolve()
    if output.exists() or output.is_relative_to(run):
        raise DataMateError('Use a new ZIP path outside the original training run')
    adapter,report = _completed_adapter(run,model_key)
    parent_path,parent = _completed_adapter(parent_adapter,model_key) if parent_adapter is not None else (None,None)
    sources = {'adapter/'+n:local(adapter,n) for n in sorted(INFERENCE_FILES|{'training_report.json','run_identity.json'})
               if (adapter/n).is_file()}
    if parent_path is not None:
        sources['provenance/parent_training_report.json'] = parent_path/'training_report.json'
    hashes = {name:sha256(path) for name,path in sources.items()}
    summary = _completed_summary(report,hashes['adapter/training_report.json'],parent,
                                hashes.get('provenance/parent_training_report.json'))
    manifest = {'format_version':1,'model_key':model_key,'base_source':get_model(model_key).to_dict(),
                **summary,'files':hashes}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name+'.tmp-'+uuid.uuid4().hex[:12])
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as packet:
        for name,path in sources.items(): packet.write(path,name)
        packet.writestr('checkpoint_inference.json',json_text(manifest,indent=2))
    if hashes!={name:sha256(path) for name,path in sources.items()}:
        raise DataMateError('Completed source changed while packaging; keep temporary ZIP for diagnosis')
    _completed_adapter(run,model_key)
    if parent_path is not None: _completed_adapter(parent_path,model_key)
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
    if manifest.get('source_kind')=='completed_adapter':
        _,report = _completed_adapter(root/'adapter',manifest['model_key'])
        parent_path = root/'provenance/parent_training_report.json'
        parent = read(parent_path) if parent_path.is_file() else None
        summary = _completed_summary(report,sha256(root/'adapter/training_report.json'),parent,
                                     sha256(parent_path) if parent is not None else None)
        if any(manifest.get(k)!=v for k,v in summary.items()):
            raise DataMateError('Completed adapter manifest differs from original training reports')
        return manifest
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
        temporary = destination.with_name('.'+destination.name+'.extract-'+uuid.uuid4().hex[:12])
        temporary.mkdir(parents=True)
        for info in packet.infolist():
            target = local(temporary,info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise DataMateError('ZIP symlink is not allowed')
            target.parent.mkdir(parents=True,exist_ok=True)
            with packet.open(info) as src,target.open('xb') as dst:
                shutil.copyfileobj(src,dst,length=4*1024**2)
    verify_package(temporary)
    if destination.is_dir():
        destination.rmdir()  # Only an empty explicit output may be replaced.
    temporary.rename(destination)
    return destination


def merge_package(root, base, output):
    root,base,output = Path(root).resolve(),Path(base).resolve(),Path(output).resolve()
    manifest = verify_package(root)
    key = manifest['model_key']
    report_hash = sha256(root/'adapter/training_report.json')
    package_hash = sha256(root/'checkpoint_inference.json')
    if output.exists() and any(output.iterdir()):
        sealed = verify_artifact(output,key)
        lineage = sealed.get('lineage') or {}
        if (lineage.get('training_report_sha256')!=report_hash
                or lineage.get('base_manifest_sha256')!=manifest['expected_base_manifest_sha256']
                or lineage.get('merged_lora') is not True
                or lineage.get('inference_package_sha256')!=package_hash):
            raise DataMateError('Merged output belongs to another/unpinned package or base; use a new output')
        return {'output':str(output),'reused':True}
    verify_artifact(base,key)
    if sha256(base/'artifact_manifest.json')!=manifest['expected_base_manifest_sha256']:
        raise DataMateError('Base manifest differs from the exact base used in training; select the original sealed base on Drive')
    from tools.merge_sql_export import merge_verified
    result = merge_verified(key,base,root/'adapter',output,inference_package_sha256=package_hash)
    verify_artifact(output,key)
    return {**result,'reused':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    package = sub.add_parser('package')
    package.add_argument('--checkpoint',required=True,type=Path)
    package.add_argument('--output',required=True,type=Path)
    package.add_argument('--reported-total-epochs',type=float)
    completed = sub.add_parser('package-adapter',help='Package a completed training run or its adapter folder')
    completed.add_argument('--run',required=True,type=Path)
    completed.add_argument('--parent-adapter',type=Path)
    completed.add_argument('--output',required=True,type=Path)
    unpack = sub.add_parser('extract')
    unpack.add_argument('--archive',required=True,type=Path)
    unpack.add_argument('--output',required=True,type=Path)
    merge = sub.add_parser('merge',help='Run in a fresh Colab CPU child with sufficient High RAM')
    for flag in ('package-root','base','output'):
        merge.add_argument('--'+flag,required=True,type=Path)
    args = parser.parse_args()
    if args.command=='package':
        result = package_checkpoint(args.checkpoint,args.output,reported_total_epochs=args.reported_total_epochs)
    elif args.command=='package-adapter':
        result = package_completed_adapter(args.run,args.output,parent_adapter=args.parent_adapter)
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
