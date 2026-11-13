import json
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest

from tools import kaggle_workflow as workflow
from tools.prepare_kaggle_benchmark import freeze_baseline
from tools.package_colab import source_path_allowed
from test_kaggle_runtime_benchmark import comparison_fixture


def test_frozen_source_uses_same_policy_without_changing_old_source_bytes(tmp_path):
    project=Path(__file__).resolve().parents[1]
    exists=subprocess.run(['git','rev-parse','--verify','main'],cwd=project,capture_output=True)
    if exists.returncode:
        pytest.skip('main reference unavailable')
    output=tmp_path/'baseline.zip'
    commit=freeze_baseline(project,'main',output)
    with zipfile.ZipFile(output) as packet:
        names=packet.namelist()
        assert all(source_path_allowed(name) for name in names)
        assert not any(name.startswith(('docs/references/','docs/history/','docs/research/')) for name in names)
        assert not any(name.endswith(('.zip','.pyc')) or '__pycache__' in name for name in names)
        original=subprocess.run(['git','show',commit+':src/datamate/agent/runner.py'],cwd=project,check=True,capture_output=True).stdout
        assert packet.read('src/datamate/agent/runner.py')==original
        assert json.loads(packet.comment)['git_commit']==commit


def test_main_failure_still_emits_known_status_log_and_capsule(tmp_path,monkeypatch):
    def fail(run,config):
        run.update('simulated_model_load')
        run.details.mkdir(parents=True)
        (run.details/'partial.txt').write_text('partial')
        raise RuntimeError('simulated OOM')
    monkeypatch.setattr(workflow,'benchmark',fail)
    with pytest.raises(RuntimeError,match='OOM'):
        workflow.entrypoint('benchmark',arm='torch',phase='main',output_root=tmp_path/'output',work_root=tmp_path/'scratch')
    output=tmp_path/'output'
    status=json.loads((output/'torch_main_status.json').read_text())
    assert status['state']=='failed' and status['failed_stage']=='simulated_model_load'
    assert 'OOM' in (output/'torch_main_run.log').read_text()
    with zipfile.ZipFile(output/'torch_main_result.zip') as packet:
        assert json.loads(packet.read('torch_main_status.json'))['state']=='failed'
        assert 'details/torch_main/partial.txt' in packet.namelist()


def test_benchmark_downloads_model_to_scratch_without_saving_weights_in_output(tmp_path,monkeypatch):
    run=workflow.SavedRun('torch_pilot',output_root=tmp_path/'out',work_root=tmp_path/'scratch')
    def fake_stage(self,command,**kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==''
        assert kwargs['env']['HF_HUB_OFFLINE']=='0'
        assert command[command.index('--model')+1]=='qwen35-9b'
        target=Path(command[command.index('--output')+1])
        target.mkdir(parents=True)
        (target/'artifact_manifest.json').write_text('{}')
    monkeypatch.setattr(workflow.SavedRun,'call',fake_stage)
    target=workflow.download_model(run,{},sys.executable)
    assert target==tmp_path/'scratch/models/qwen35-9b'
    assert (target/'artifact_manifest.json').is_file()
    assert not list((tmp_path/'out').rglob('*.safetensors'))
    assert not (tmp_path/'out/qwen35-9b').exists()


@pytest.mark.parametrize('fail_download,limited',[(False,False),(False,True),(True,False)])
def test_one_kit_input_installs_and_downloads_before_timed_pipeline(tmp_path,monkeypatch,fail_download,limited):
    upload=tmp_path/'input/kit'
    (upload/'baseline/configs').mkdir(parents=True)
    workflow.save(upload/'baseline/configs/qwen35-9b.json',{})
    workflow.save(upload/'kaggle_kit.json',{'baseline_commit':'cc2ea9842d88a84fa6f9982d0dad3dbf6686ce76',
                  'source_archives':{'baseline':{'sha256':'old-source'}}})
    workflow.save(upload/'benchmark/benchmark_manifest.json',{'files':{}})
    (upload/'benchmark/dataset').mkdir()
    (upload/'benchmark/dataset/questions.jsonl').write_text('{"id":"q1"}\n')
    stages=[]
    def environment(run,*,gpu):
        assert gpu
        stages.append('install_libraries')
        return sys.executable
    def call(self,command,**kwargs):
        stage=kwargs['stage']; stages.append(stage)
        if stage=='download_and_verify_model':
            if fail_download:raise RuntimeError('simulated download failure')
            target=Path(command[command.index('--output')+1])
            workflow.save(target/'artifact_manifest.json',{})
            assert kwargs['env']['HF_HUB_OFFLINE']=='0'
        elif stage=='measure_actual_context_without_weights':
            workflow.save(Path(command[command.index('--output')+1]),{'ready':True,'suggested_context_tokens':16384})
        elif stage=='run_pilot_questions':
            assert kwargs['env']['HF_HUB_OFFLINE']==kwargs['env']['TRANSFORMERS_OFFLINE']=='1'
            workflow.save(self.details/'benchmark_metrics.json',{'valid_complete_run':not limited,'execution_complete':True,
                          'measurement_warnings':['generation_timeouts'] if limited else [],'written':4,
                          'answer_types':{'number':4},'inference_seconds':12})
    monkeypatch.setattr(workflow,'environment',environment)
    monkeypatch.setattr(workflow.SavedRun,'call',call)
    options=dict(arm='torch',phase='pilot',kit_root=upload,input_root=tmp_path/'input',
                 output_root=tmp_path/'out',work_root=tmp_path/'scratch')
    if fail_download:
        with pytest.raises(RuntimeError,match='download failure'):
            workflow.entrypoint('benchmark',**options)
        assert 'run_pilot_questions' not in stages
        assert json.loads((tmp_path/'out/torch_pilot_status.json').read_text())['state']=='failed'
    else:
        status=workflow.entrypoint('benchmark',**options)
        assert status['state']==('needs_review' if limited else 'passed') and status['inference_seconds']==12
        assert status['questions']==4 and status['execution_complete']
        assert stages.index('install_libraries')<stages.index('download_and_verify_model')<stages.index('run_pilot_questions')
        assert stages.index('measure_actual_context_without_weights')<stages.index('run_pilot_questions')
    with zipfile.ZipFile(tmp_path/'out/torch_pilot_result.zip') as packet:
        assert not any('/models/' in name or name.endswith('.safetensors') for name in packet.namelist())


def test_explicit_checkpoint_is_verified_without_downloading_or_copying(tmp_path,monkeypatch):
    run=workflow.SavedRun('vllm_pilot',output_root=tmp_path/'out',work_root=tmp_path/'scratch')
    target=tmp_path/'existing-model'
    workflow.save(target/'artifact_manifest.json',{})
    calls=[]
    monkeypatch.setattr(run,'call',lambda command,**kwargs:calls.append((command,kwargs)))
    assert workflow.download_model(run,{'model_dir':str(target)},sys.executable)==target
    assert len(calls)==1 and calls[0][1]['stage']=='verify_explicit_model'
    assert not (tmp_path/'scratch/models').exists()


def test_environment_repairs_partial_venv_without_ensurepip_or_host_pip(tmp_path,monkeypatch):
    run=workflow.SavedRun('model_prepare',output_root=tmp_path/'out',work_root=tmp_path/'scratch',archive=False)
    target=run.work/'venv'/('Scripts/python.exe' if workflow.os.name=='nt' else 'bin/python')
    target.parent.mkdir(parents=True)
    target.write_text('incomplete environment left by failed ensurepip')
    calls=[]
    def call(command,**kwargs):
        calls.append((command,kwargs['stage']))
        if kwargs['stage']=='check_python_installer':
            raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(run,'call',call)
    monkeypatch.setattr(workflow.urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(b"print('installer fixture')\n"))
    result=workflow.environment(run,gpu=False)
    assert result==target
    assert '--without-pip' in calls[0][0]
    assert all('-m ensurepip' not in ' '.join(str(x) for x in command) for command,_ in calls)
    installs=[command for command,_ in calls if 'install' in command or any(str(x).endswith('get-pip.py') for x in command)]
    assert installs and all(command[0]==target for command in installs)
    assert any(stage=='install_pip_in_isolated_environment' for _,stage in calls)
    assert any('huggingface-hub>=1.5,<2' in command for command,_ in calls)


def test_environment_with_working_pip_never_downloads_the_bootstrap(tmp_path,monkeypatch):
    run=workflow.SavedRun('model_prepare',output_root=tmp_path/'out',work_root=tmp_path/'scratch',archive=False)
    calls=[]
    monkeypatch.setattr(run,'call',lambda command,**kwargs:calls.append((command,kwargs['stage'])))
    monkeypatch.setattr(workflow.urllib.request,'urlopen',lambda *a,**kw:pytest.fail('Unneeded installer download'))
    workflow.environment(run,gpu=False)
    assert '--without-pip' in calls[0][0]
    assert not any(stage=='install_pip_in_isolated_environment' for _,stage in calls)


def test_gpu_environment_installs_the_shared_dependency_specification_online(tmp_path,monkeypatch):
    run=workflow.SavedRun('torch_pilot',output_root=tmp_path/'out',work_root=tmp_path/'scratch')
    calls=[]
    monkeypatch.setattr(run,'call',lambda command,**kwargs:calls.append((command,kwargs['stage'])))
    workflow.environment(run,gpu=True)
    install=next(command for command,stage in calls if stage=='install_libraries')
    assert str(workflow.PROJECT)+'[llm,vllm]' in install and 'psutil' in install
    assert '--no-cache-dir' in install and '--no-index' not in install
    assert any(stage=='check_libraries' for _,stage in calls)
    assert any(stage=='record_library_versions' for _,stage in calls)


def test_invalid_python_environment_does_not_install_into_host(tmp_path,monkeypatch):
    run=workflow.SavedRun('model_prepare',output_root=tmp_path/'out',work_root=tmp_path/'scratch',archive=False)
    def call(command,**kwargs):
        if kwargs['stage']=='verify_isolated_python':
            raise subprocess.CalledProcessError(1,command)
        assert kwargs['stage']=='create_python_environment_without_pip'
    monkeypatch.setattr(run,'call',call)
    monkeypatch.setattr(workflow.urllib.request,'urlopen',lambda *a,**kw:pytest.fail('Must stop before installer download'))
    with pytest.raises(subprocess.CalledProcessError):
        workflow.environment(run,gpu=False)


def result_capsule(root,arm,state='passed',*,limited=False,failed_stage=None):
    source=comparison_fixture(root/f'{arm}-evidence','transformers' if arm=='torch' else 'vllm')
    if limited:
        metrics=json.loads((source/'benchmark_metrics.json').read_text())
        metrics.update(valid_complete_run=False,written=1,expected_questions=1,timeout_calls=1)
        workflow.save(source/'benchmark_metrics.json',metrics)
        (source/'model_calls.jsonl').write_text(json.dumps({'id':'q1','start_seconds':1,'messages_sha256':'same',
                'stop_reason':'timeout_aborted','error':'GenerationTimeoutError: deadline'})+'\n')
        (source/'submission.jsonl').write_text(json.dumps({'id':'q1','type':'abstain','reason':'deadline'})+'\n')
    packet=root/f'{arm}_main_result.zip'
    with zipfile.ZipFile(packet,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f'{arm}_main_status.json',json.dumps({'state':state,'questions':1,'failed_stage':failed_stage}))
        for path in source.iterdir():
            archive.write(path,Path('details')/(arm+'_main')/path.name)
    return packet


def test_cpu_comparison_reads_our_named_capsules_and_writes_readable_files(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    result_capsule(inputs,'torch');result_capsule(inputs,'vllm')
    status=workflow.entrypoint('compare',input_root=inputs,output_root=tmp_path/'output',work_root=tmp_path/'scratch')
    assert status['state']=='passed'
    output=tmp_path/'output'
    assert json.loads((output/'comparison.json').read_text())['inference_speedup']==1
    assert 'TF/vLLM = 1.000' in (output/'comparison.txt').read_text(encoding='utf-8')
    assert 'Transformers,vLLM' in (output/'comparison.csv').read_text(encoding='utf-8')
    with zipfile.ZipFile(output/'comparison_result.zip') as archive:
        assert {'comparison.json','comparison.csv','comparison.txt'} <= set(archive.namelist())


def test_failed_main_receipt_is_rejected_even_if_old_metrics_were_valid(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    result_capsule(inputs,'torch',state='failed');result_capsule(inputs,'vllm')
    with pytest.raises(ValueError,match='did not pass'):
        workflow.entrypoint('compare',input_root=inputs,output_root=tmp_path/'out',work_root=tmp_path/'scratch')
    assert json.loads((tmp_path/'out/comparison_status.json').read_text())['state']=='failed'
    assert (tmp_path/'out/comparison_result.zip').is_file()


@pytest.mark.parametrize('legacy',[False,True])
def test_completed_deadline_results_produce_diagnostics_without_gpu_rerun(tmp_path,legacy):
    inputs=tmp_path/'inputs';inputs.mkdir()
    result_capsule(inputs,'torch')
    result_capsule(inputs,'vllm',state='failed' if legacy else 'needs_review',limited=True,
                   failed_stage='run_main_questions' if legacy else None)
    status=workflow.entrypoint('compare',input_root=inputs,output_root=tmp_path/'out',work_root=tmp_path/'scratch')
    report=json.loads((tmp_path/'out/comparison.json').read_text())
    assert status['state']=='needs_review' and not status['paired_comparison_valid']
    assert report['execution_complete_both'] and report['inference_speedup'] is None
    assert report['observed_inference_time_ratio']==1
    assert report['per_question'][0]['difference_kind']=='deadline_affected_difference'
    text=(tmp_path/'out/comparison.txt').read_text(encoding='utf-8')
    assert 'pipeline' in text and 'deadline' in text and 'Tỷ số thời gian quan sát' in text
    assert (tmp_path/'out/comparison_result.zip').is_file()


def test_comparison_also_accepts_our_results_after_dataset_auto_extraction(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    for arm in ('torch','vllm'):
        packet=result_capsule(inputs,arm)
        with zipfile.ZipFile(packet) as archive:
            archive.extractall(inputs)
        packet.unlink()
    status=workflow.entrypoint('compare',input_root=inputs,output_root=tmp_path/'out',work_root=tmp_path/'scratch')
    assert status['state']=='passed' and (tmp_path/'out/comparison.csv').is_file()


def test_upload_kit_has_separate_source_directories_not_nested_archives(tmp_path):
    from tools.package_kaggle_kit import package
    current=tmp_path/'current.zip';baseline=tmp_path/'baseline.zip';benchmark=tmp_path/'benchmark.zip'
    for path,content in [(current,b'new source'),(baseline,b'old source'),(benchmark,b'question fixture')]:
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('same-name.txt',content)
            if path==baseline:archive.comment=json.dumps({'git_commit':'a'*40}).encode()
    output=tmp_path/'upload.zip'
    report=package(current,baseline,benchmark,output)
    with zipfile.ZipFile(output) as archive:
        assert archive.read('current/same-name.txt')==b'new source'
        assert archive.read('baseline/same-name.txt')==b'old source'
        assert not any(n.endswith('.zip') for n in archive.namelist())
        for name,sha in report['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==sha


@pytest.mark.parametrize('shape',['extracted','archive'])
def test_copyable_bootstrap_works_in_a_fresh_namespace_for_both_input_shapes(tmp_path,shape):
    from tools.create_kaggle_benchmark_notebook import build
    root=Path(__file__).resolve().parents[1]
    inputs=tmp_path/'inputs';inputs.mkdir()
    kit=tmp_path/'fake-kit'
    source=kit/'current/tools/kaggle_workflow.py';source.parent.mkdir(parents=True)
    source.write_text("def entrypoint(task, **config):\n    return {'state':'passed','observed_task':task,'kit_root':config['kit_root']}\n")
    manifest={'format':'datamate-kaggle-kit-v1','source_archives':{'current':{'sha256':'b'*64}},
              'files':{'current/tools/kaggle_workflow.py':hashlib.sha256(source.read_bytes()).hexdigest()}}
    (kit/'kaggle_kit.json').write_text(json.dumps(manifest))
    if shape=='archive':
        with zipfile.ZipFile(inputs/'kaggle-runtime-kit.zip','w') as archive:
            for path in kit.rglob('*'):
                if path.is_file():archive.write(path,path.relative_to(kit))
    else:
        import shutil
        shutil.copytree(kit,inputs/'mounted-kit')
    code=''.join(next(c for c in build(root,'compare')['cells'] if c['cell_type']=='code')['source'])
    options={'input_root':str(inputs),'output_root':str(tmp_path/'out'),'bootstrap_root':str(tmp_path/'bootstrap')}
    code=code.replace('OPTIONS = {}','OPTIONS = '+repr(options),1)
    namespace={}
    exec(compile(code,'fresh-kaggle-cpu-cell','exec'),namespace)
    assert namespace['STATUS']['state']=='passed' and namespace['STATUS']['observed_task']=='compare'
    assert Path(namespace['STATUS']['kit_root'],'kaggle_kit.json').is_file()


def test_watchdog_stops_a_silent_subprocess_without_waiting_for_console_output(tmp_path,monkeypatch):
    def silent(run,config):
        run.call([sys.executable,'-c','import time; time.sleep(30)'],stage='silent_child',timeout=.2)
    monkeypatch.setattr(workflow,'benchmark',silent)
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        workflow.entrypoint('benchmark',arm='vllm',phase='pilot',output_root=tmp_path/'out',work_root=tmp_path/'scratch')
    assert time.monotonic()-started<10
    assert json.loads((tmp_path/'out/vllm_pilot_status.json').read_text())['state']=='failed'
    assert (tmp_path/'out/vllm_pilot_result.zip').is_file()


def test_two_notebooks_are_generated_without_a_prepare_notebook():
    from tools.create_kaggle_benchmark_notebook import build
    root=Path(__file__).resolve().parents[1]
    assert not (root/'notebooks/DataMate_Kaggle_Prepare_Model.ipynb').exists()
    for kind,name,count in [('benchmark','DataMate_Kaggle_Runtime_Benchmark.ipynb',2),
                            ('compare','DataMate_Kaggle_Compare.ipynb',1)]:
        notebook=json.loads((root/'notebooks'/name).read_text(encoding='utf-8'))
        assert notebook==build(root,kind)
        code=[c for c in notebook['cells'] if c['cell_type']=='code']
        assert len(code)==count
        for cell in code:
            compile(''.join(cell['source']),kind,'exec')
            assert cell['outputs']==[]
