"""Re-run both frozen SQL training stages into NEW outputs; never change evidence.

This standalone helper is also copied into training_code/reproduce.py. A fresh
stage-1 parent is pinned for stage 2; original reports are never relabelled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(file):
    value = hashlib.sha256()
    with Path(file).open('rb') as handle:
        while block := handle.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def prepare_config(evidence, base, work, stage):
    evidence, base, work = (Path(p).resolve() for p in (evidence, base, work))
    if stage not in ('stage1', 'stage2'):
        raise ValueError('Choose stage1 or stage2')
    if work.is_relative_to(evidence) or evidence.is_relative_to(work):
        raise ValueError('Reproduction output must be separate from frozen evidence')
    config = json.loads((evidence/'configs'/f'{stage}.json').read_text(encoding='utf-8'))
    target = work/stage/'adapter'
    if target.exists():
        raise ValueError('Reproduction does not overwrite an existing adapter')
    config.update(base_dir=str(base), dataset_dir=str(evidence/'data'/stage), output_dir=str(target))
    if stage == 'stage2':
        parent = work/'stage1/adapter'
        config['initial_adapter'] = str(parent)
        config['initial_adapter_files'] = {name:digest(parent/name) for name in (
            'adapter_config.json','adapter_model.safetensors','training_report.json')}
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True, help='Verified local public base, never the merged model')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--stage', choices=('stage1','stage2'), required=True)
    args = parser.parse_args()
    root = args.evidence.resolve()
    manifest = json.loads((root/'data_manifest.json').read_text(encoding='utf-8'))
    for name, info in manifest['files'].items():
        file = (root/name).resolve()
        if not file.is_relative_to(root) or digest(file) != info['sha256']:
            raise ValueError('Frozen training evidence differs: '+name)
    config = prepare_config(root,args.base,args.work,args.stage)
    config_path = args.work.resolve()/f'{args.stage}.json'
    config_path.parent.mkdir(parents=True,exist_ok=True)
    with config_path.open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(config,handle,ensure_ascii=False,indent=2)
    source = root/'source'/args.stage
    env = {**os.environ, 'PYTHONPATH':str(source/'src'),
           'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    subprocess.run([sys.executable,'-m','datamate','train','--config',str(config_path)],
                   env=env,cwd=source,check=True)


if __name__ == '__main__':
    main()
