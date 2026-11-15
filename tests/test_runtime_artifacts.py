import json
from pathlib import Path
import shutil
import struct
from types import SimpleNamespace
import zipfile

import pytest

from datamate.config import Settings
from datamate.domain import DataMateError
from datamate.models.artifacts import seal_artifact, sha256, verify_artifact
from datamate.models.registry import MODELS, ModelSpec
from tools import runtime_artifacts as artifacts
from tools.install_runtime import install_bundle


def model(path, spec, *, lineage=None):
    path.mkdir(parents=True)
    header = json.dumps({'tensor': {'shape': [3], 'dtype': 'BF16', 'data_offsets': [0, 6]}}).encode()
    (path / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 6)
    (path / 'config.json').write_text('{}')
    seal_artifact(path, spec, lineage)


def wheel(directory, name, version):
    path = directory / f'{name}-{version}-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(f'{name}-{version}.dist-info/METADATA', f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
    return path


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    spec = ModelSpec('fixture-cache', 'test/model', 'a' * 40, 3, '2026-01-01T00:00:00Z')
    monkeypatch.setitem(MODELS, spec.key, spec)
    profile = {'python': [3, 13], 'system': 'linux', 'platform': 'linux-x86_64',
               'machine': 'x86_64', 'implementation': 'CPython'}
    monkeypatch.setattr(artifacts, 'python_profile', lambda python: profile)
    calls, installed = [], {}

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if '-c' in command:
            if 'PackageNotFoundError' in command[command.index('-c') + 1]:
                return SimpleNamespace(stdout=json.dumps({name: installed.get(name) for name in json.loads(command[-1])}))
            return SimpleNamespace(stdout='1\n')
        if 'download' in command:
            target = Path(command[command.index('--dest')+1])
            wheel(target, 'fixturelib', '1.0')
            wheel(target, 'setuptools', '77.0')
            wheel(target, 'wheel', '0.45')
        if 'install' in command and '--require-hashes' in command:
            target = Path(command[command.index('--find-links')+1])
            manifest = json.loads((target / 'wheelhouse_manifest.json').read_text())
            installed.update({p['name']: p['version'] for p in manifest['packages'].values()})
        if 'stage-model' in command:
            target = Path(command[command.index('--output')+1])
            model(target, spec)
        return SimpleNamespace(stdout='')

    monkeypatch.setattr(artifacts, '_run', run)
    monkeypatch.setattr('tools.install_runtime._run', run)
    project = Path(__file__).resolve().parents[1]
    root = tmp_path / 'artifacts'
    settings = Settings(backend='transformers', model_key=spec.key, model_dir=str(tmp_path / 'missing-model'),
                        split_root=str(tmp_path / 'data'), cache_dir=str(tmp_path / 'cache'))
    return SimpleNamespace(root=root, project=project, settings=settings, spec=spec, calls=calls,
                           installed=installed, profile=profile)


def prepare(runtime, **kwargs):
    return artifacts.prepare_bundle(runtime.root, runtime.project, runtime.settings, 'target-python', install=True, **kwargs)


def test_online_preparation_downloads_once_then_reuses_exact_models_and_libraries(runtime):
    first = prepare(runtime)
    assert first['preparation']['wheelhouse'] == 'downloaded'
    assert first['preparation']['models'][runtime.spec.key] == 'downloaded'
    assert first['preparation']['libraries']['status'] == 'installed'
    assert first['gpu_smoke_verified'] is False
    assert len([c for c, _ in runtime.calls if 'download' in c]) == 1
    assert len([c for c, _ in runtime.calls if 'stage-model' in c]) == 1
    second = prepare(runtime)
    assert second['preparation']['wheelhouse'] == second['preparation']['libraries']['status'] == 'hit'
    assert second['preparation']['models'][runtime.spec.key] == 'hit'
    assert len([c for c, _ in runtime.calls if 'download' in c or 'stage-model' in c]) == 2
    artifacts.verify_bundle(runtime.root / 'bundle.json')


def test_corrupt_cache_is_not_resealed_or_overwritten_and_online_repair_uses_new_folder(runtime):
    first = prepare(runtime)
    original = runtime.root / first['models'][runtime.spec.key]['path']
    weights = original / 'model.safetensors'
    broken = weights.read_bytes()[:-1] + b'X'
    weights.write_bytes(broken)
    with pytest.raises(DataMateError, match='Hash'):
        artifacts.verify_bundle(runtime.root / 'bundle.json')
    second = prepare(runtime)
    replacement = runtime.root / second['models'][runtime.spec.key]['path']
    assert replacement != original and weights.read_bytes() == broken
    verify_artifact(replacement, runtime.spec.key)


def test_wrong_model_candidate_is_preserved_and_rejected_without_base_substitution(runtime, tmp_path, monkeypatch):
    other = ModelSpec('other-cache', 'test/other', 'b' * 40, 3, '2026-01-01T00:00:00Z')
    monkeypatch.setitem(MODELS, other.key, other)
    source = tmp_path / 'other'
    model(source, other)
    original = sha256(source / 'artifact_manifest.json')
    runtime.settings.model_dir = str(source)
    with pytest.raises(DataMateError, match='manifest'):
        prepare(runtime)
    assert not any('stage-model' in command for command, _ in runtime.calls)
    assert sha256(source / 'artifact_manifest.json') == original


def test_valid_external_model_is_copied_without_download_and_derived_corruption_does_not_fall_back(runtime, tmp_path):
    source = tmp_path / 'fine-tuned'
    model(source, runtime.spec, lineage={'training': 'fixture'})
    runtime.settings.model_dir = str(source)
    bundle = prepare(runtime)
    assert bundle['preparation']['models'][runtime.spec.key] == 'copied'
    assert not any('stage-model' in c for c, _ in runtime.calls)
    weights = source / 'model.safetensors'
    weights.write_bytes(weights.read_bytes()[:-1] + b'X')
    runtime.settings.model_dir = str(source)
    with pytest.raises(DataMateError, match='fine-tune'):
        prepare(runtime)
    assert not any('stage-model' in c for c, _ in runtime.calls)


def test_offline_verification_never_launches_download_and_rejects_corrupt_wheels(runtime):
    bundle = prepare(runtime)
    before = len(runtime.calls)
    artifacts.verify_bundle(runtime.root / 'bundle.json')
    assert len(runtime.calls) == before
    wheelhouse = runtime.root / bundle['wheelhouse']
    target = next(wheelhouse.glob('*.whl'))
    target.write_bytes(target.read_bytes() + b'corrupt')
    with pytest.raises(DataMateError, match='hash'):
        artifacts.verify_bundle(runtime.root / 'bundle.json')
    assert len(runtime.calls) == before


def test_offline_install_repairs_wrong_versions_and_relocates_model_paths_after_copy(runtime, tmp_path):
    bundle = prepare(runtime)
    moved = tmp_path / 'moved-bundle'
    shutil.copytree(runtime.root, moved)
    runtime.installed['fixturelib'] = 'wrong-version'
    output = tmp_path / 'local-config.json'
    result = install_bundle(moved / 'bundle.json', 'target-python', runtime.project, output)
    assert result['libraries']['changed']['fixturelib']['installed'] == 'wrong-version'
    assert runtime.installed['fixturelib'] == '1.0'
    settings = Settings.load(output, use_env=False)
    assert Path(settings.model_dir).is_relative_to(moved)
    offline_installs = [c for c, _ in runtime.calls if 'install' in c]
    assert offline_installs and all('--no-index' in c for c in offline_installs)
    assert artifacts.verify_bundle(moved / 'bundle.json')['config'] == bundle['config']


def test_check_only_does_not_install_and_platform_or_lock_changes_do_not_reuse_wrong_wheels(runtime, tmp_path):
    bundle = prepare(runtime)
    directory = runtime.root / bundle['wheelhouse']
    before = len([c for c, _ in runtime.calls if 'install' in c])
    runtime.installed['fixturelib'] = 'wrong'
    with pytest.raises(DataMateError, match='version'):
        artifacts.install_libraries(directory, 'target-python', check_only=True)
    assert len([c for c, _ in runtime.calls if 'install' in c]) == before
    runtime.profile['python'] = [3, 12]
    with pytest.raises(DataMateError, match='platform'):
        artifacts.install_libraries(directory, 'target-python')


def test_lock_include_change_invalidates_wheel_cache(runtime, tmp_path):
    lock = tmp_path / 'requirements.txt'
    child = tmp_path / 'pins.txt'
    lock.write_text('-r pins.txt\n')
    child.write_text('fixturelib==1.0\n')
    first = prepare(runtime, requirements=lock)
    child.write_text('fixturelib==2.0\n')
    second = prepare(runtime, requirements=lock)
    assert first['wheelhouse'] != second['wheelhouse']


def test_vllm_online_preparation_rejects_windows_before_pip_download(runtime):
    runtime.settings.backend = 'vllm'
    runtime.profile['system'] = 'win32'
    with pytest.raises(DataMateError, match='Linux'):
        prepare(runtime)
    assert not runtime.calls


def test_docker_build_installer_verifies_before_offline_pip_and_rejects_corruption(runtime, monkeypatch):
    import importlib.util
    loader = importlib.util.spec_from_file_location('image_runtime_installer', runtime.project / 'docker/install_runtime.py')
    image = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(image)
    bundle = prepare(runtime)
    directory = runtime.root / bundle['wheelhouse']
    solution = runtime.root / 'solution_manifest.json'
    solution.write_text('{"inference_extra":"llm"}')
    monkeypatch.setattr(image, 'profile', lambda: runtime.profile)
    commands = []
    monkeypatch.setattr(image.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    result = image.install(directory, solution)
    assert result == {'extra': 'llm', 'wheelhouse_verified': True}
    assert all('--no-index' in c for c in commands if 'install' in c)
    assert any('--require-hashes' in c for c in commands)
    commands.clear()
    path = next(directory.glob('*.whl'))
    path.write_bytes(path.read_bytes() + b'bad')
    with pytest.raises(ValueError, match='hash'):
        image.install(directory, solution)
    assert commands == []


def test_packaging_checks_sealed_wheelhouse_and_records_verification(runtime, tmp_path):
    from tools.package_solution import stage
    bundle = prepare(runtime)
    config = runtime.root / bundle['config']
    directory = runtime.root / bundle['wheelhouse']
    source = runtime.root / bundle['models'][runtime.spec.key]['path']
    report = stage(runtime.project, config, tmp_path / 'context', directory, {runtime.spec.key: source})
    assert report['wheelhouse_verified'] is True
    assert (tmp_path / 'context/docker/install_runtime.py').is_file()
    lock = directory / 'requirements.resolved.txt'
    lock.write_text('changed')
    with pytest.raises(DataMateError, match='requirements'):
        stage(runtime.project, config, tmp_path / 'bad-context', directory, {runtime.spec.key: source})
    assert not (tmp_path / 'bad-context').exists()


def test_package_cli_resolves_bundle_without_hardcoded_model_or_wheel_paths(runtime, tmp_path, monkeypatch, capsys):
    from tools.package_solution import main
    import sys
    prepare(runtime)
    output = tmp_path / 'bundle-context'
    monkeypatch.setattr(sys, 'argv', ['package_solution', '--development', '--bundle', str(runtime.root / 'bundle.json'), '--output', str(output)])
    main()
    assert json.loads(capsys.readouterr().out)['context'] == str(output)
    manifest = json.loads((output / 'solution_manifest.json').read_text(encoding='utf-8'))
    assert manifest['wheelhouse_verified'] is True
    assert (output / 'models' / runtime.spec.key / 'artifact_manifest.json').is_file()


def test_release_refuses_missing_training_evidence_before_staging(runtime, tmp_path):
    from tools.package_solution import stage
    bundle = prepare(runtime)
    source = runtime.root / bundle['models'][runtime.spec.key]['path']
    output = tmp_path / 'release'
    with pytest.raises(ValueError, match='training evidence'):
        stage(runtime.project, runtime.root / bundle['config'], output,
              runtime.root / bundle['wheelhouse'], {runtime.spec.key:source}, release=True,
              base_image='vllm/vllm-openai@sha256:'+'1'*64)
    assert not output.exists()


def test_context_relocates_every_inference_path_under_fixed_code_root(runtime, tmp_path):
    from tools.package_solution import stage
    bundle = prepare(runtime)
    source = runtime.root / bundle['models'][runtime.spec.key]['path']
    output = tmp_path / 'context'
    stage(runtime.project, runtime.root / bundle['config'], output,
          runtime.root / bundle['wheelhouse'], {runtime.spec.key:source})
    config = json.loads((output/'configs/submission.json').read_text(encoding='utf-8'))
    assert config['model_dir'] == '/code/models/' + runtime.spec.key
    assert (output/'requirements.txt').read_bytes() == (runtime.root / bundle['wheelhouse'] / 'requirements.resolved.txt').read_bytes()


def test_release_includes_sealed_training_data_and_offline_training_wheels(runtime, tmp_path):
    from test_training_evidence import evidence_spec
    from tools.training_evidence import assemble, with_dependencies, verify
    from tools.package_solution import stage
    bundle = prepare(runtime)
    source = runtime.root / bundle['models'][runtime.spec.key]['path']
    wheelhouse = runtime.root / bundle['wheelhouse']
    spec = evidence_spec(tmp_path)
    spec['inference_artifacts'] = {runtime.spec.key:sha256(source/'artifact_manifest.json')}
    assemble(spec,tmp_path/'evidence-source')
    evidence = tmp_path/'evidence-offline'
    with_dependencies(tmp_path/'evidence-source',wheelhouse,evidence)
    output = tmp_path/'release-context'
    report = stage(runtime.project,runtime.root/bundle['config'],output,wheelhouse,
                   {runtime.spec.key:source},training_evidence=evidence,release=True,
                   base_image='team/base@sha256:'+'1'*64)
    assert report['training_evidence_verified'] is True and report['release_context'] is True
    assert verify(output/'training_code') == verify(evidence)
    assert 'data/' not in (output/'.dockerignore').read_text()
    readme=(output/'README.md').read_text(encoding='utf-8')
    assert '/code/predict.sh' in readme and runtime.spec.revision in readme
    assert 'data/train.jsonl' in readme and 'data_manifest.json' in readme
    assert 'docs/' not in readme


def test_release_preparation_never_downloads_when_merged_input_is_missing(runtime,tmp_path,monkeypatch):
    import importlib.util
    import sys
    path=runtime.project/'docker/prepare_release.py'
    loader=importlib.util.spec_from_file_location('release_prepare_test',path)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    monkeypatch.setattr(module.sys,'platform','linux')
    monkeypatch.setattr(module.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(module,'verify',lambda _: {'files':{},'inference_artifacts':{}})
    monkeypatch.setattr(module,'prepare_bundle',lambda *a,**kw:pytest.fail('download/preparation started before checking local models'))
    monkeypatch.setattr(sys,'argv',['prepare_release','--base-image','team/base@sha256:'+'1'*64,
        '--evidence',str(tmp_path/'evidence'),'--models',str(tmp_path/'missing-models'),'--output',str(tmp_path/'out')])
    with pytest.raises((ValueError,DataMateError),match='model|checkpoint|merged'):
        module.main()
