"""Run the existing BF16 merge on a Colab CPU child, preserving the parent's GPU."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from datamate.domain import DataMateError


def check_cpu_ram(parameters, available_bytes):
    # BF16 base plus adapter, deserialization/merge buffers and allocator margin.
    required_gib=max(32,math.ceil(parameters*2*1.75/1024**3))
    if available_bytes<required_gib*1024**3:
        raise DataMateError(f'CPU merge cần ít nhất {required_gib} GiB RAM hệ thống còn trống; '
                            f'hiện có {available_bytes/1024**3:.1f} GiB. Chọn High RAM rồi chạy lại. '
                            'Đây là RAM hệ thống, không phải VRAM L4.')
    return required_gib


def effective_ram_bytes(host_available, cgroup_root=Path('/sys/fs/cgroup')):
    available=host_available
    for limit_name,usage_name in [('memory.max','memory.current'),
                                  ('memory/memory.limit_in_bytes','memory/memory.usage_in_bytes')]:
        try:
            limit=(cgroup_root/limit_name).read_text().strip()
            if limit!='max':
                usage=int((cgroup_root/usage_name).read_text().strip())
                available=min(available,max(0,int(limit)-usage))
        except (OSError,ValueError):pass
    return available


def cpu_resources():
    if 'torch' in sys.modules:
        raise DataMateError('CPU merge phải chạy trong process Python mới, không gọi trong kernel đã nạp Torch')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['ACCELERATE_USE_CPU']='true'
    import torch
    import psutil
    if torch.cuda.is_available():raise DataMateError('CPU merge không được nhìn thấy CUDA GPU')
    return {'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
            'accelerate_use_cpu':os.environ['ACCELERATE_USE_CPU'],
            'cuda_available':torch.cuda.is_available(),
            'available_ram_bytes':effective_ram_bytes(psutil.virtual_memory().available)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',action='store_true',help='Report CPU isolation without loading weights')
    for name in ('base','adapter','output'):parser.add_argument('--'+name)
    args=parser.parse_args()
    if not args.probe and not all((args.base,args.adapter,args.output)):
        parser.error('merge requires --base, --adapter, --output')
    resources=cpu_resources()
    if args.probe:
        print(json.dumps(resources));return
    manifest=json.loads((Path(args.base)/'artifact_manifest.json').read_text(encoding='utf-8'))
    required=check_cpu_ram(manifest['tensor_parameters'],resources['available_ram_bytes'])
    print(json.dumps({'mode':'cpu_bf16_merge','required_available_ram_gib_policy':required,**resources}),flush=True)
    # CUDA is hidden before Torch/Accelerate/PEFT; exporter also explicitly uses CPU.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from tools.merge_sql_export import merge_verified
    result=merge_verified('qwen35-9b',args.base,args.adapter,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
