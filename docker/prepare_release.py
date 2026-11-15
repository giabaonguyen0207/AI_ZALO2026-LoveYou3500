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
from datamate.models.artifacts import verify_artifact, sha256
from datamate.models.registry import model_paths
from tools.package_solution import stage, verify_training_link
from tools.runtime_artifacts import prepare_bundle, prepare_wheels, verify_bundle
from tools.training_evidence import with_dependencies, verify
from tools.frozen_runtime import prepare_requirements, check_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-image', required=True, help='Exact digest used to run this preparation container')
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    parser.add_argument('--merged-model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-freeze', type=Path,
                        help='Reuse exact inference versions from a successful run; never resolve newer versions')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/submission.json')
    args = parser.parse_args()
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        parser.error('Run inside the selected Linux amd64 base image')
    output = args.output.resolve()
    if (output/'context').exists() or (output/'training-evidence').exists():
        parser.error('Choose a new output directory; existing releases are preserved')
    evidence = verify(args.evidence)
    settings = Settings.load(args.config,use_env=False)
    settings.model_dir = str(args.merged_model or args.models/'qwen35-9b-stage2-comparison-merged')
    settings.rag.embedding_dir = str(args.models/'vietnamese-embedding')
    settings.rag.reranker_dir = str(args.models/'vietnamese-reranker')
    if settings.rag.correction_enabled:
        settings.rag.correction_dir = str(args.models/settings.rag.correction_key)
    # Fail before any dependency/model download or installation. The general
    # bundle helper is also used for public-base preparation and allows download;
    # a submission of this fine-tuned system must never take that fallback.
    for key,folder in model_paths(settings).items():
        path=Path(folder)
        if not (path/'artifact_manifest.json').is_file():
            raise ValueError('Missing sealed local model/checkpoint: '+str(path))
        manifest=verify_artifact(path,key)
        verify_training_link(evidence,key,path)
        if key==settings.model_key and not (manifest.get('lineage') or {}).get('merged_lora'):
            raise ValueError('Generator must be the verified fine-tuned merged model')
    output.mkdir(parents=True,exist_ok=True)
    requirements = ROOT/'configs/docker-runtime-constraints.txt'
    if args.runtime_freeze:
        requirements = prepare_requirements(args.runtime_freeze, ROOT, output/'inference-frozen.requirements.txt')
    bundle = prepare_bundle(output/'artifacts',ROOT,settings,sys.executable,
                            requirements=requirements,install=True)
    verify_bundle(output/'artifacts/bundle.json')
    if args.runtime_freeze:
        check_runtime(args.runtime_freeze, skip_project=True)
    # Training uses a separate offline environment; original Torch/Transformers
    # versions differ from inference and must not overwrite the runtime stack.
    training_evidence = args.evidence
    if not (args.evidence/'wheelhouse').is_dir():
        wheels, _, _ = prepare_wheels(output/'training-artifacts',ROOT,'llm',sys.executable,
            requirements=args.evidence/'requirements.txt',
            extra_index_urls=['https://download.pytorch.org/whl/cu130'])
        training_evidence = output/'training-evidence'
        with_dependencies(args.evidence,wheels,training_evidence)
    models = {key:output/'artifacts'/item['path'] for key,item in bundle['models'].items()}
    report = stage(ROOT,output/'artifacts'/bundle['config'],output/'context',
                   output/'artifacts'/bundle['wheelhouse'],models,
                   base_image=args.base_image,training_evidence=training_evidence,release=True)
    check = subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,check=True)
    (output/'pip-check.txt').write_text(check.stdout+check.stderr,encoding='utf-8')
    freeze = subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (output/'runtime-freeze.txt').write_text(freeze.stdout,encoding='utf-8')
    print(json.dumps({'context':str(output/'context'), 'parameters':report['inventory']['total_parameters'],
                      'training_files':len(evidence['files']), 'image_built':False,
                      'gpu_rehearsal_verified':False},indent=2))


if __name__ == '__main__':
    main()
