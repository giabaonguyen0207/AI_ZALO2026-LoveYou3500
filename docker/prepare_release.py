"""Run INSIDE the selected Linux amd64 base before offline Docker build.

Downloads dependencies during preparation only. All model inputs must already
be sealed. It never substitutes base Qwen for the fine-tuned merged checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from datamate.config import Settings
from tools.package_solution import stage
from tools.runtime_artifacts import prepare_bundle, prepare_wheels, verify_bundle
from tools.training_evidence import with_dependencies, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-image', required=True, help='Exact digest used to run this preparation container')
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    parser.add_argument('--merged-model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        parser.error('Run inside the selected Linux amd64 base image')
    output = args.output.resolve()
    if (output/'context').exists() or (output/'training-evidence').exists():
        parser.error('Choose a new output directory; existing releases are preserved')
    evidence = verify(args.evidence)
    settings = Settings.load(ROOT/'configs/submission.json',use_env=False)
    settings.model_dir = str(args.merged_model or args.models/'qwen35-9b-stage2-comparison-merged')
    settings.rag.embedding_dir = str(args.models/'vietnamese-embedding')
    settings.rag.reranker_dir = str(args.models/'vietnamese-reranker')
    output.mkdir(parents=True,exist_ok=True)
    bundle = prepare_bundle(output/'artifacts',ROOT,settings,sys.executable,
                            requirements=ROOT/'configs/docker-runtime-constraints.txt',install=True)
    verify_bundle(output/'artifacts/bundle.json')
    # Training uses a separate offline environment; original Torch/Transformers
    # versions differ from inference and must not overwrite the runtime stack.
    wheels, _, _ = prepare_wheels(output/'training-artifacts',ROOT,'llm',sys.executable,
        requirements=args.evidence/'requirements.txt',
        extra_index_urls=['https://download.pytorch.org/whl/cu130'])
    with_dependencies(args.evidence,wheels,output/'training-evidence')
    models = {key:output/'artifacts'/item['path'] for key,item in bundle['models'].items()}
    report = stage(ROOT,output/'artifacts'/bundle['config'],output/'context',
                   output/'artifacts'/bundle['wheelhouse'],models,
                   base_image=args.base_image,training_evidence=output/'training-evidence',release=True)
    check = subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,check=True)
    (output/'pip-check.txt').write_text(check.stdout+check.stderr,encoding='utf-8')
    freeze = subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (output/'runtime-freeze.txt').write_text(freeze.stdout,encoding='utf-8')
    print(json.dumps({'context':str(output/'context'), 'parameters':report['inventory']['total_parameters'],
                      'training_files':len(evidence['files']), 'image_built':False,
                      'gpu_rehearsal_verified':False},indent=2))


if __name__ == '__main__':
    main()
