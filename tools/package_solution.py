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


def verify_training_link(evidence, key, source):
    """Team-trained artifacts need lineage evidence; pinned public correction does not."""
    expected = evidence.get('inference_artifacts', {}).get(key)
    if expected is None and key == 'vietnamese-error-correction':
        # This exact public checkpoint is independently pinned, bounded and
        # weight-hash verified by verify_artifact; it was not trained by the team.
        verify_artifact(source, key)
        return
    if expected != sha256(Path(source)/'artifact_manifest.json'):
        raise ValueError('Training evidence is not linked to the selected inference model: ' + key)


def image_readme(settings, specs, evidence, base_image, release):
    lines=['# DataMate submission', '',
        'Status: '+('Release context; GPU/driver rehearsal is still required.' if release else 'DEVELOPMENT ONLY — do not submit.'), '',
        '## Approach', '',
        'Modular Python agent with Adapter + Strategy. Vietnamese embedding/reranker retrieve rulebook, README and schema context for the current business. The local generator proposes plans/SQL; read-only tools execute and validate calculations, then project the required answer type. Time is resolved from as_of_date and the mounted split rulebook. Routing IDs do not decide answers.', '',
        'When enabled, the pinned public Vietnamese correction checkpoint supplies an advisory query variant; the original question remains the source of evidence. It is used unmodified, not trained by the team. Its parameters count toward the inference cap (conservative 400M bound; actual staged tensor count is in its manifest). Hybrid retrieval fuses lexical and dense candidates before reranking.', '',
        '## Runtime', '',
        'All inference assets are local. Network must be disabled. Input /data is read-only; scratch/cache /work; output /result/submission.jsonl and one-line per-question records in /result/logs. Additional debug files are in /result/debug.', '',
        'Exact scoring command inside the container: `bash /code/predict.sh` (no arguments).', '',
        'The selected generator/backend is '+settings.backend+'; retrieval is '+settings.rag.mode+'. '+
        'Resolved settings: [configs/submission.json](configs/submission.json). Installed dependency pins: [requirements.txt](requirements.txt).', '',
        'Candidate base: `'+str(base_image)+'`. This file does not certify driver, GPU memory or six-hour runtime compatibility.', '',
        '## Inference models', '',
        '| Key | Repository | Revision | Public revision date | License | Parameters | Local files |',
        '|---|---|---|---|---|---:|---|']
    for model in specs['models']:
        key=model['key']
        lines.append('| '+ ' | '.join(str(v) for v in [key,model['repo_id'],model['revision'],
            model['revision_date'],model['license'],model['parameters'],'models/'+key+'/'])+' |')
    lines.extend(['', 'Combined inference parameters: '+str(specs['total_parameters'])+'. Each model directory includes its artifact_manifest.json, file SHA256 checksums, source revision and fine-tune lineage where applicable.', '',
        '## Training and data', '',
        'Fine-tuning/data generation were performed before this release. Frozen source, generation tools, original TRAIN, configs, logs, reports and verification adapters are under training_code/. Public/private test are not training inputs.', '',
        '[training_code/README.md](training_code/README.md) explains the two stages, measured hardware/runtime, exact provenance and reproducibility limits. [training_code/data_manifest.json](training_code/data_manifest.json) enumerates local paths, sizes, SHA256, record counts, data splits and revision-pinned public URLs for omitted training-only base assets.', ''])
    for item in (evidence or {}).get('training_stages',[]):
        files=evidence['files']
        lines.append('- '+item['name']+': train `'+item['train']+'` ('+str(files[item['train']].get('records'))+'), validation `'+item['validation']+'` ('+str(files[item['validation']].get('records'))+').')
    lines.extend(['', 'Training dependencies are bundled offline in training_code/wheelhouse and installed into a separate environment by train.sh; they do not replace inference dependencies.', '',
        'Exact retraining command, after staging the declared public base locally: `bash /code/training_code/train.sh /work/verified-public-base /work/new-training-reproduction`.', '',
        'It trains stage1 then stage2 with the new parent hashes and new outputs, preserving frozen evidence. Seeds/configs are recorded; CUDA/QLoRA/kernel execution may be nondeterministic. The original parent requires its exact original base manifest for direct continuation.', '',
        'Generation source snapshots: training_code/source/stage1/tools/ and training_code/source/stage2/tools/. Original generation reports/inventories and frozen source instructions are included alongside the datasets. Training verification is separate from the six-hour inference limit.', '',
        'To regenerate proposals into NEW scratch outputs (not overwrite the frozen datasets), use the training environment prepared by train.sh, then:', '',
        '```bash',
        'cd /code/training_code',
        'cp -a source/stage1 /work/regen-stage1',
        'mkdir -p /work/regen-stage1/train',
        'cp -a data_generation/original_train /work/regen-stage1/train/train',
        'cd /work/regen-stage1',
        'python tools/prepare_sql_training.py --output /work/sql-pass --cache-dir /work/sql-catalogs',
        'python tools/release_sql_training.py --pass-dir /work/sql-pass --output /work/sql-regenerated --tokenizer /work/verified-public-base --packages /work/regenerated-packages',
        'cd /code/training_code/source/stage2',
        'python tools/prepare_sql_stage2.py --parent /code/training_code/data/stage1 --split /code/training_code/data_generation/original_train --sources configs/training_sources.json --cache /work/stage2-catalogs --tokenizer /work/verified-public-base --output /work/stage2-regenerated --limit 300',
        '```', '',
        'Use the frozen data/ manifests for the submitted checkpoint; regenerated datasets are new artifacts. Tokenizer/software versions and nondeterministic timing metadata can change hashes. Do not edit frozen manifests to force a match.', '',
        '## Execution logs', '',
        'Per-question JSON records contain id, answer, elapsed_ms, raw thinking (call order) or null, and actual executed SQL/helper functions, arguments and results/errors. No generated reasoning is summarized or fabricated.', '',
        'Build-context checks and model hashes are recorded in [solution_manifest.json](solution_manifest.json). They are not evidence of a completed GPU rehearsal.', ''])
    return '\n'.join(lines)


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
        if release:
            verify_training_link(evidence,key,source)
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
    for name in ('pyproject.toml', 'predict.sh'):
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
    (output/'README.md').write_text(image_readme(settings,specs,evidence,base_image,release),encoding='utf-8',newline='\n')
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
