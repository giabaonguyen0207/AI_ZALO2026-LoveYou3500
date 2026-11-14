import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('broken',[False,True])
def test_install_and_pip_check_target_private_runtime(tmp_path,monkeypatch,capsys,broken):
    from tools.create_finetuned_compare_colab import INSTALL
    work=tmp_path/'source';work.mkdir();(work/'pyproject.toml').write_text('[project]\nname="fixture"')
    save=tmp_path/'results';save.mkdir()
    runtime=tmp_path/'runtime'
    monkeypatch.setenv('DATAMATE_RUNTIME_DIR',str(runtime))
    inherited_path = os.environ.get('PATH', '')
    calls=[]
    conflict='vllm requires torch==2.14.0, but you have torch 2.13.0.'
    def command(args,**kwargs):
        calls.append(args)
        if 'venv' in args:
            assert '--without-pip' in args
            python=runtime/'bin/python';python.parent.mkdir(parents=True);python.touch()
            (runtime/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
        if 'install' in args:
            assert '--python' in args
            assert args[args.index('--python')+1]==str(runtime/'bin/python')
            assert '.[llm,vllm,rag]' in ' '.join(args)
        if args[-1]=='check':
            assert args[0]==str(runtime/'bin/python')
            return SimpleNamespace(returncode=1 if broken else 0,stdout=conflict if broken else 'No broken requirements found.',stderr='')
        return SimpleNamespace(returncode=0,stdout='fixture versions',stderr='')
    state={'WORK':work,'SAVE':save,'Path':Path,'os':os,'sys':sys,'hashlib':hashlib,
           'subprocess':SimpleNamespace(run=command)}
    if broken:
        with pytest.raises(RuntimeError,match='torch 2.13.0'):
            exec(compile(INSTALL,'install','exec'),state)
        assert conflict in capsys.readouterr().out
        assert conflict in (save/'pip-check.txt').read_text()
    else:
        exec(compile(INSTALL,'install','exec'),state)
        assert state['RUNTIME_PYTHON']==str(runtime/'bin/python')
        assert state['RUNTIME_ENV']['PYTHONPATH'].startswith(str(work/'src'))
        assert state['RUNTIME_ENV']['PYTHONNOUSERSITE']=='1'
        assert state['RUNTIME_ENV']['VIRTUAL_ENV']==str(runtime)
        assert state['RUNTIME_ENV']['PATH']==str(runtime/'bin')+os.pathsep+inherited_path
        assert os.environ.get('PATH','')==inherited_path
    assert any('install' in args for args in calls)


def test_parent_pip_can_check_venv_without_ensurepip_or_downloads(tmp_path):
    envdir=tmp_path/'isolated'
    # The test runner may expose the editable project's egg-info via PYTHONPATH.
    # This probe measures an empty isolated interpreter, not inherited project metadata.
    clean_env={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PYTHONHOME'}}
    subprocess.run([sys.executable,'-m','venv','--without-pip',str(envdir)],env=clean_env,check=True,capture_output=True)
    python=envdir/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    result=subprocess.run([sys.executable,'-m','pip','--python',str(python),'check'],env=clean_env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    probe=subprocess.run([str(python),'-c','import sys; assert sys.prefix != sys.base_prefix'],env=clean_env,capture_output=True,text=True)
    assert probe.returncode==0,probe.stderr
