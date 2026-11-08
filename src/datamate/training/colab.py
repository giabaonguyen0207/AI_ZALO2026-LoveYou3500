"""Small Colab helpers for an already curated seed, not data generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import zipfile

from ..agent.runner import Agent
from ..domain import DataMateError
from ..models.artifacts import sha256
from .seed import verify_seed_dataset


def unpack_seed_archive(archive,destination,release_path,sources_path):
    archive,destination=Path(archive),Path(destination)
    release=json.loads(Path(release_path).read_text(encoding='utf-8'))
    if sha256(archive)!=release['archive_sha256']:
        raise DataMateError('Seed ZIP khác bản đã chuẩn bị; kiểm tra file upload/source ZIP cùng phiên bản')
    with zipfile.ZipFile(archive) as handle:
        names=handle.namelist()
        if len(names)!=len(set(names)) or 'manifest.json' not in names:
            raise DataMateError('Seed ZIP trùng tên hoặc thiếu manifest')
        manifest_bytes=handle.read('manifest.json')
        if hashlib.sha256(manifest_bytes).hexdigest()!=release['dataset_manifest_sha256']:
            raise DataMateError('Manifest seed không khớp release')
        manifest=json.loads(manifest_bytes)
        allowed=json.loads(Path(sources_path).read_text(encoding='utf-8'))['sources']
        if not any(r['fingerprint']==manifest.get('source_fingerprint') and r['role']=='competition_training' for r in allowed):
            raise DataMateError('Seed không thuộc nguồn TRAIN đã đăng ký')
        if manifest.get('target_pipeline_version')!=Agent.pipeline_version:
            raise DataMateError('Source code không cùng giao thức agent với dataset; thay source ZIP')
        source_root=Path(__file__).resolve().parents[1]
        for name,digest in manifest.get('runtime_source_hashes',{}).items():
            source=(source_root/name).resolve()
            if not source.is_relative_to(source_root) or not source.is_file() or sha256(source)!=digest:
                raise DataMateError('Source ZIP khác code đã chuẩn bị seed: '+name+'; dùng gói source cùng release và restart runtime')
        if set(names)!=set(manifest['files'])|{'manifest.json'}:
            raise DataMateError('Seed ZIP có file thừa/thiếu so với manifest')
        if sum(item.file_size for item in handle.infolist())>512*1024**2:
            raise DataMateError('Seed ZIP giải nén vượt giới hạn 512 MiB')
        for item in handle.infolist():
            name=item.filename
            if name!=Path(name).name or '\\' in name or ':' in name or name in {'.','..'} or stat.S_ISLNK(item.external_attr>>16):
                raise DataMateError('Đường dẫn seed ZIP không an toàn')
            if name!='manifest.json' and hashlib.sha256(handle.read(name)).hexdigest()!=manifest['files'][name]:
                raise DataMateError('File seed ZIP khác hash: '+name)
        if destination.exists() and any(destination.iterdir()):
            if not (destination/'manifest.json').is_file() or sha256(destination/'manifest.json')!=release['dataset_manifest_sha256']:
                raise DataMateError('Thư mục seed cũ khác release; dùng thư mục/tag mới')
        else:
            destination.mkdir(parents=True,exist_ok=True)
            handle.extractall(destination)
    verify_seed_dataset(destination)
    return destination


def make_seed_training_config(model_key,base_dir,dataset_dir,output_dir,*,smoke=False,max_steps=5,min_vram_gib=22):
    manifest=verify_seed_dataset(dataset_dir)
    if model_key!='gemma4-12b' or model_key!=manifest['target_model_key']:
        raise DataMateError('Notebook seed này chỉ fine-tune Gemma 4 12B')
    if max_steps!=-1 and (type(max_steps) is not int or max_steps<1):
        raise DataMateError('max_steps cần -1 hoặc số nguyên dương')
    # Attention-only adapters are the initial memory-conscious trial; they are
    # not claimed to be optimal. The trainer checks the actual Gemma modules.
    suffixes=['q_proj','k_proj','v_proj','o_proj']
    return {'model_key':model_key,'base_dir':str(base_dir),'dataset_dir':str(dataset_dir),
            'output_dir':str(output_dir),'method':'qlora','rank':16,'alpha':32,'dropout':0.05,
            'target_suffixes':suffixes,'max_length':manifest['max_length'],'batch_size':1,
            'gradient_accumulation_steps':1 if smoke else 8,'epochs':1,
            'max_steps':1 if smoke else max_steps,'learning_rate':5e-5,
            'save_steps':1,'seed':42,'reject_overlength':True,
            'smoke_longest_only':bool(smoke),'min_vram_gib':min_vram_gib}


def gpu_preflight(torch_module,min_vram_gib=22):
    if not torch_module.cuda.is_available():
        raise DataMateError('Chọn GPU runtime của Colab trước khi train')
    total=torch_module.cuda.get_device_properties(0).total_memory/1024**3
    if total<min_vram_gib:
        raise DataMateError(f'GPU hiện có {total:.1f} GiB, dưới ngưỡng {min_vram_gib} GiB của profile 16K. Chọn GPU lớn hơn nếu Colab cấp; không tự cắt prompt để ép chạy.')
    return {'gpu':torch_module.cuda.get_device_name(0),'total_vram_gib':round(total,2),
            'bf16_supported':bool(torch_module.cuda.is_bf16_supported()),
            'torch':torch_module.__version__,'cuda':torch_module.version.cuda,
            'minimum_vram_policy_gib':min_vram_gib,
            'memory_fit_verified':False,'next_step':'one optimizer step on the longest actual sequence'}
