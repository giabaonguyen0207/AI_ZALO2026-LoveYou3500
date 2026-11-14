"""Stage a minimal offline Docker build context; does not build or submit an image."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'src')]
from datamate.config import Settings
from datamate.models.artifacts import sha256, verify_artifact
from datamate.models.registry import inventory_from_settings, relocate_models, inference_extras
from tools.runtime_artifacts import verify_bundle, verify_wheels
from tools.training_evidence import verify as verify_training_evidence


def stage(root, config, output, wheelhouse, model_sources, *, base_image=None,
          training_evidence=None, release=False):
    root, output, wheelhouse = map(lambda p: Path(p).resolve(), (root, output, wheelhouse))
    settings = Settings.load(config)
    specs = inventory_from_settings(settings)
    evidence = None
    if release and training_evidence is None:
        raise ValueError('Release requires verified training evidence, data and offline dependencies')
    if training_evidence is not None:
        training_evidence = Path(training_evidence).resolve()
        evidence = verify_training_evidence(training_evidence)
        if output == training_evidence or output.is_relative_to(training_evidence):
            raise ValueError('Build output must be outside training evidence')
    if release and not re.fullmatch(r'.+@sha256:[0-9a-f]{64}', base_image or ''):
        raise ValueError('Release requires a digest-pinned base image')
    if not wheelhouse.is_dir() or not list(wheelhouse.glob('*.whl')):
        raise ValueError('Prepare a Linux/GPU-compatible wheelhouse before staging')
    wheel_manifest = None
    if (wheelhouse / 'wheelhouse_manifest.json').exists():
        wheel_manifest = verify_wheels(wheelhouse)
        expected_extra = inference_extras(settings)
        if wheel_manifest['identity']['extra'] != expected_extra:
            raise ValueError('Wheelhouse backend does not match solution config')
    if release and (not wheel_manifest or wheel_manifest['identity']['profile']['system'] != 'linux'
                    or wheel_manifest['identity']['profile']['machine'] not in ('x86_64', 'AMD64')):
        raise ValueError('Release requires a verified Linux amd64 wheelhouse')
    if release:
        training_wheels = verify_wheels(training_evidence / 'wheelhouse')
        if training_wheels['identity']['profile'] != wheel_manifest['identity']['profile']:
            raise ValueError('Training offline wheels must match the image Python/platform')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory; existing releases are not overwritten')
    manifests = {}
    for model in specs['models']:
        key = model['key']
        if key not in model_sources:
            raise ValueError('Missing --model-source ' + key + '=PATH')
        source = Path(model_sources[key]).resolve()
        if source == output or output.is_relative_to(source):
            raise ValueError('Build output must be outside the source checkpoint')
        manifests[key] = verify_artifact(source, key)
        if release and evidence.get('inference_artifacts', {}).get(key) != sha256(source / 'artifact_manifest.json'):
            raise ValueError('Training evidence is not linked to the selected inference model: ' + key)
    if wheelhouse == output or output.is_relative_to(wheelhouse):
        raise ValueError('Build output must be outside wheelhouse')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / 'src', output / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    shutil.copytree(wheelhouse, output / 'wheelhouse')
    if wheel_manifest:
        verify_wheels(output / 'wheelhouse', wheel_manifest['identity'])
    (output / 'docker').mkdir()
    shutil.copy2(root / 'docker/Dockerfile.competition', output / 'docker/Dockerfile.competition')
    shutil.copy2(root / 'docker/install_runtime.py', output / 'docker/install_runtime.py')
    for name in ('pyproject.toml', 'README.md', 'predict.sh'):
        shutil.copy2(root / name, output / name)
    # This context is already curated. The repository dockerignore excludes data
    # and would silently drop required training assets from the final image.
    (output / '.dockerignore').write_text('**/__pycache__/\n**/*.pyc\n.git/\n', encoding='utf-8')
    if wheel_manifest:
        shutil.copy2(wheelhouse / 'requirements.resolved.txt', output / 'requirements.txt')
    else:
        (output / 'requirements.txt').write_text('# Development-only unsealed wheelhouse\n', encoding='utf-8')
    (output / 'configs').mkdir()
    settings.split_root, settings.result_dir, settings.cache_dir = '/data', '/result', '/work/datamate'
    relocate_models(settings, {key: '/code/models/' + key for key in manifests})
    if settings.rag.mode != 'none':
        settings.rag.index_dir = '/work/datamate/retrieval'
    (output / 'configs/submission.json').write_text(json.dumps(settings.to_dict(), indent=2) + '\n', encoding='utf-8')
    shutil.copy2(root / 'configs/training_sources.json', output / 'configs/training_sources.json')
    for key in manifests:
        shutil.copytree(Path(model_sources[key]), output / 'models' / key,
                        ignore=shutil.ignore_patterns('.cache'))
        verify_artifact(output / 'models' / key, key)
    if training_evidence is not None:
        shutil.copytree(training_evidence, output / 'training_code')
        verify_training_evidence(output / 'training_code')
    else:
        (output / 'training_code').mkdir()
        (output / 'training_code/README.md').write_text('Development context: training evidence absent. Do not submit.\n', encoding='utf-8')
    files = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    report = {'inventory': specs, 'files': files, 'image_built': False,
              'wheelhouse_verified': wheel_manifest is not None,
              'inference_extra': inference_extras(settings),
              'base_image': base_image, 'base_image_policy': 'team_selected',
              'release_context': release, 'training_evidence_verified': evidence is not None,
              'training_manifest_sha256': sha256(output / 'training_code/data_manifest.json') if evidence else None,
              'base_image_gpu_verified': False, 'full_gpu_inference_performed': False}
    (output / 'solution_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--bundle', type=Path, help='Verified artifact bundle: resolve config/models/wheelhouse automatically')
    parser.add_argument('--output', type=Path, default=root / 'outputs/packages/docker-context')
    parser.add_argument('--wheelhouse', type=Path)
    parser.add_argument('--model-source', action='append', default=[], metavar='KEY=PATH')
    parser.add_argument('--training-evidence', type=Path, help='Sealed folder with data_manifest.json and offline training wheelhouse')
    parser.add_argument('--development', action='store_true', help='Allow incomplete fixture context; never submit this image')
    parser.add_argument('--base-image', default=os.environ.get('DATAMATE_BASE_IMAGE'),
                        help='Team-selected base tag/digest, recorded for release traceability')
    args = parser.parse_args()
    models = {}
    for value in args.model_source:
        key, sep, path = value.partition('=')
        if not sep or not key or not path or key in models:
            parser.error('Each --model-source must be a unique KEY=PATH')
        models[key] = Path(path)
    config, wheelhouse = args.config, args.wheelhouse
    if args.bundle:
        bundle_path = args.bundle.resolve()
        bundle = verify_bundle(bundle_path)
        config = config or bundle_path.parent / bundle['config']
        wheelhouse = wheelhouse or bundle_path.parent / bundle['wheelhouse']
        models = {**{key: bundle_path.parent / item['path'] for key, item in bundle['models'].items()}, **models}
    if wheelhouse is None:
        parser.error('Provide --bundle or --wheelhouse')
    report = stage(root, config or root / 'configs/submission.json', args.output, wheelhouse, models,
                   base_image=args.base_image, training_evidence=args.training_evidence, release=not args.development)
    print(json.dumps({'context': str(args.output), 'total_parameters': report['inventory']['total_parameters'],
                      'image_built': False}))


if __name__ == '__main__':
    main()
