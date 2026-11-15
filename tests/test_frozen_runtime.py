from pathlib import Path

import pytest

from tools.frozen_runtime import read_freeze, prepare_requirements, check_runtime


def test_freeze_preserves_all_library_pins_but_builds_project_locally(tmp_path):
    root = Path(__file__).resolve().parents[1]
    freeze = tmp_path/'freeze.txt'
    freeze.write_text((root/'configs/docker-runtime-constraints.txt').read_text()
                      + '\ndatamate-agent==0.2.0\nexample_lib==1.2.3\n', encoding='utf-8')
    path = prepare_requirements(freeze,root,tmp_path/'requirements.txt')
    pins = read_freeze(path)
    assert pins['example-lib']=='1.2.3' and 'datamate-agent' not in pins
    assert check_runtime(path,version_lookup=pins.__getitem__)['versions_match']
    with pytest.raises(ValueError,match='Runtime differs'):
        check_runtime(path,version_lookup=lambda name:'0.0.0')


@pytest.mark.parametrize('text',['vllm>=0.33','-e /old/source','torch @ https://example.invalid/a.whl',
                                'vllm==0.33.0\nvllm==0.34.0'])
def test_unpinned_or_conflicting_freeze_stops_instead_of_changing_stack(tmp_path,text):
    path=tmp_path/'freeze.txt'
    path.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError):read_freeze(path)


def test_docker_stack_cannot_silently_override_freeze(tmp_path):
    root=Path(__file__).resolve().parents[1]
    freeze=tmp_path/'freeze.txt'
    freeze.write_text((root/'configs/docker-runtime-constraints.txt').read_text().replace('vllm==0.33.0','vllm==0.34.0'))
    with pytest.raises(ValueError,match='Docker pin'):
        prepare_requirements(freeze,root,tmp_path/'requirements.txt')
