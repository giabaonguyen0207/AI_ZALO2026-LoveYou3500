"""Export a merged model while preserving unloaded auxiliary checkpoint tensors."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import struct
import uuid

from datamate.domain import DataMateError
from datamate.models import artifacts
from datamate.training.checkpoints import atomic_json


def tensor_inventory(root):
    root=Path(root).resolve();result={}
    for path in sorted(root.glob('*.safetensors')):
        if path.is_symlink():raise DataMateError('Export không nhận symlink trọng số')
        with path.open('rb') as handle:
            size=struct.unpack('<Q',handle.read(8))[0]
            if size>100_000_000:raise DataMateError('Header quá lớn')
            header=json.loads(handle.read(size))
        for name,meta in header.items():
            if name=='__metadata__':continue
            if name in result:raise DataMateError('Tensor trùng: '+name)
            result[name]={'file':path.name,'shape':meta['shape'],'dtype':meta['dtype'],
                          'bytes':meta['data_offsets'][1]-meta['data_offsets'][0]}
    if not result:raise DataMateError('Thiếu tensor checkpoint')
    return result


def preserve_auxiliary_tensors(base, output):
    """Only missing mtp.* may be restored; all active names/shapes must match."""
    base,output=Path(base),Path(output)
    original=tensor_inventory(base);exported=tensor_inventory(output)
    missing=set(original)-set(exported);extra=set(exported)-set(original)
    if extra or any(not name.startswith('mtp.') for name in missing):
        raise DataMateError('Export khác tensor active của base: missing='+str(sorted(missing))+' extra='+str(sorted(extra)))
    for name in exported:
        if exported[name]['shape']!=original[name]['shape']:
            raise DataMateError('Export sai shape tensor: '+name)
    receipt={'preserved_tensor_names':sorted(missing),
             'preserved_parameters':sum(math.prod(original[n]['shape']) for n in missing)}
    if not missing:return receipt
    from safetensors import safe_open
    from safetensors.torch import save_file
    tensors={}
    for file in sorted({original[n]['file'] for n in missing}):
        with safe_open(base/file,framework='pt',device='cpu') as handle:
            for name in sorted(missing):
                if original[name]['file']==file:tensors[name]=handle.get_tensor(name)
    # No synthetic padding and no count relaxation: copy original values/dtypes.
    temporary=output/('preserved-mtp.tmp-'+uuid.uuid4().hex)
    save_file(tensors,temporary,metadata={'format':'pt'})
    temporary.replace(output/'model-preserved-mtp.safetensors')
    # A lone model.safetensors takes precedence over the index in HF loading.
    lone=output/'model.safetensors'
    if lone.exists():
        target=output/'model-exported.safetensors'
        if target.exists():raise DataMateError('Export đã có shard model-exported')
        lone.rename(target)
    complete=tensor_inventory(output)
    if set(complete)!=set(original):raise DataMateError('Export chưa đủ tensor sau khôi phục MTP')
    atomic_json(output/'model.safetensors.index.json',{
        'metadata':{'total_size':sum(t['bytes'] for t in complete.values())},
        'weight_map':{n:complete[n]['file'] for n in sorted(complete)}})
    return receipt


def merge_verified(model_key, base, adapter, output):
    from datamate.config import configure_offline
    configure_offline()
    base,adapter,output=Path(base),Path(adapter),Path(output)
    artifacts.verify_artifact(base,model_key)
    spec=artifacts.get_model(model_key)
    if output.exists() and any(output.iterdir()):raise DataMateError('Merge output phải là thư mục mới')
    report=json.loads((adapter/'training_report.json').read_text(encoding='utf-8'))
    if report['base_source']!=spec.to_dict():raise DataMateError('Adapter khác base revision')
    import torch
    from transformers import AutoModelForImageTextToText,AutoTokenizer
    from peft import PeftModel
    model=AutoModelForImageTextToText.from_pretrained(base,local_files_only=True,trust_remote_code=False,
                                                    dtype=torch.bfloat16,device_map={'':'cpu'})
    model=PeftModel.from_pretrained(model,adapter,local_files_only=True).merge_and_unload()
    model.save_pretrained(output,safe_serialization=True,max_shard_size='4GB')
    AutoTokenizer.from_pretrained(adapter,local_files_only=True).save_pretrained(output)
    for path in base.glob('LICENSE*'):shutil.copy2(path,output/path.name)
    shutil.copy2(adapter/'training_report.json',output/'training_report.json')
    receipt=preserve_auxiliary_tensors(base,output)
    atomic_json(output/'auxiliary_export.json',receipt)
    manifest=artifacts.seal_artifact(output,spec,{
        'merged_lora':True,'training_report_sha256':artifacts.sha256(output/'training_report.json'),
        'base_manifest_sha256':artifacts.sha256(base/'artifact_manifest.json'),'auxiliary_export':receipt})
    return {'output':str(output.resolve()),'tensor_parameters':manifest['tensor_parameters'],'auxiliary_export':receipt}
