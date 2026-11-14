"""Self-contained Kaggle stages with explicit saved files, including failure receipts.

This module orchestrates existing tools in fresh processes; it does not implement
or change the agent loop. Importing it needs only the Python standard library.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile

PROJECT = Path(__file__).resolve().parents[1]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def unique_input(root, name):
    found = list(Path(root).rglob(name))
    if len(found) != 1:
        raise ValueError(f'Add Input must contain exactly one {name}; found: {found}')
    return found[0]


def unpack(archive, destination):
    destination = Path(destination).resolve()
    marker = destination / '.archive-sha256'
    expected = digest(archive)
    if destination.exists():
        if marker.is_file() and marker.read_text() == expected:
            return destination
        raise ValueError('Extraction folder contains another input version: ' + str(destination))
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as packet:
        for member in packet.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError('Unsafe ZIP path')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlink ZIP entry is not supported')
        packet.extractall(destination)
    marker.write_text(expected)
    return destination


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


class SavedRun:
    def __init__(self, stem, *, output_root, work_root, archive=True):
        self.stem = stem
        self.output = Path(output_root).resolve()
        self.work = Path(work_root).resolve() / stem
        self.details = self.output / 'details' / stem
        self.status_path = self.output / f'{stem}_status.json'
        self.log_path = self.output / f'{stem}_run.log'
        self.archive_path = self.output / f'{stem}_result.zip' if archive else None
        self.output.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.status = {'state': 'running', 'stage': 'starting', 'artifact_stem': stem,
                       'status_file': self.status_path.name, 'log_file': self.log_path.name,
                       'archive_file': self.archive_path.name if self.archive_path else None}
        self.started = time.monotonic()
        self.active_process = None
        self.additional_files = []

    def update(self, stage, **fields):
        self.status.update(stage=stage, elapsed_seconds=round(time.monotonic()-self.started, 2), **fields)
        save(self.status_path, self.status)
        line = f'[{self.stem}] {stage}'
        print(line, flush=True)
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    def call(self, command, *, stage, env=None, timeout=1800, cwd=None):
        self.update(stage)
        child_env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING':'utf-8', **(env or {})}
        with self.log_path.open('a', encoding='utf-8') as log, self.log_path.open('r', encoding='utf-8', errors='replace') as reader:
            log.flush()
            reader.seek(0, 2)
            process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=child_env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name == 'posix')
            self.active_process = process
            deadline = time.monotonic() + timeout
            try:
                while True:
                    try:
                        code = process.wait(timeout=min(5, max(.1, deadline-time.monotonic())))
                    except subprocess.TimeoutExpired:
                        chunk = reader.read(65536)
                        if chunk:
                            print(chunk, end='', flush=True)
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f'{stage} exceeded {timeout} seconds; process group stopped')
                        continue
                    for chunk in iter(lambda: reader.read(65536), ''):
                        print(chunk, end='', flush=True)
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                    return
            finally:
                stop_process(process)
                self.active_process = None

    def finalize(self):
        self.status['elapsed_seconds'] = round(time.monotonic()-self.started, 2)
        save(self.status_path, self.status)
        if self.archive_path:
            temporary = self.archive_path.with_suffix('.zip.tmp')
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as packet:
                paths = [self.status_path, self.log_path, *self.additional_files]
                paths.extend(p for p in self.details.rglob('*') if p.is_file())
                for path in sorted(set(paths)):
                    if path.is_file():
                        packet.write(path, path.relative_to(self.output))
            temporary.replace(self.archive_path)
        print('Saved status:', self.status_path, flush=True)
        print('Saved log:', self.log_path, flush=True)
        if self.archive_path:
            print('Saved result:', self.archive_path, flush=True)


def bootstrap_pip(run, python):
    """Use PyPA's supported installer when the base interpreter cannot ensurepip."""
    run.update('download_python_installer')
    url = os.environ.get('DATAMATE_GET_PIP_URL', 'https://bootstrap.pypa.io/get-pip.py')
    if not url.startswith('https://'):
        raise ValueError('The pip bootstrap URL must use HTTPS')
    target = run.work/'get-pip.py'
    temporary = target.with_suffix('.py.tmp')
    limit = 8*1024*1024
    size = 0
    try:
        with urllib.request.urlopen(url,timeout=60) as response, temporary.open('wb') as handle:
            while block := response.read(1024*1024):
                size += len(block)
                if size > limit:
                    raise ValueError('Unexpectedly large pip bootstrap script')
                handle.write(block)
        if not size:
            raise ValueError('Empty pip bootstrap response')
        compile(temporary.read_bytes(),str(target),'exec')
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    run.update('bootstrap_pip_without_ensurepip', pip_bootstrap_url=url, pip_bootstrap_sha256=digest(target))
    run.call([python,target,'--disable-pip-version-check'],stage='install_pip_in_isolated_environment',timeout=300)


def environment(run, *, gpu):
    python = run.work / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    # Failed ensurepip may leave a Python executable in a partial environment.
    run.call([sys.executable, '-m', 'venv', '--without-pip', python.parent.parent],
             stage='create_python_environment_without_pip', timeout=120)
    expected = str(python.parent.parent.resolve())
    run.call([python,'-c','import os,sys; assert os.path.normcase(os.path.realpath(sys.prefix)) == '
              + repr(os.path.normcase(expected)) + ', "Python is not using the requested isolated environment"'],
             stage='verify_isolated_python',timeout=30)
    try:
        run.call([python,'-m','pip','--version'],stage='check_python_installer',timeout=30)
    except subprocess.CalledProcessError:
        bootstrap_pip(run,python)
    run.call([python, '-m', 'pip', 'install', '--upgrade', 'pip'], stage='prepare_pip', timeout=300)
    # Both arms resolve the same dependency specification; comparison checks versions.
    packages = ['-e', str(PROJECT) + '[llm,vllm]', 'psutil'] if gpu else ['huggingface-hub>=1.5,<2']
    run.call([python, '-m', 'pip', 'install', '--no-cache-dir', *packages], stage='install_libraries', timeout=1800)
    if gpu:
        run.call([python, '-m', 'pip', 'check'], stage='check_libraries', timeout=120)
        run.call([python, '-m', 'pip', 'freeze'], stage='record_library_versions', timeout=120)
    return python


def download_model(run, config, python):
    key = config.get('model_key', 'qwen35-9b')
    # Download into temporary storage, never into the saved benchmark Output.
    # A shared scratch checkpoint can be reused by pilot/main in one session.
    if not key or Path(key).name != key or key in {'.', '..'}:
        raise ValueError('Invalid model key')
    override = config.get('model_dir')
    target = Path(override).resolve() if override else run.work.parent/'models'/key
    env = {'PYTHONPATH': str(PROJECT/'src'), 'CUDA_VISIBLE_DEVICES': '',
           'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0'}
    if override:
        code = 'from datamate.models.artifacts import verify_artifact; import sys; verify_artifact(sys.argv[1],sys.argv[2])'
        run.call([python,'-c',code,target,key],stage='verify_explicit_model',env=env,
                 timeout=config.get('download_timeout',14400))
    else:
        run.call([python,'-m','datamate','stage-model','--model',key,'--output',target],
                 stage='download_and_verify_model',env=env,cwd=PROJECT,
                 timeout=config.get('download_timeout',14400))
    if not (target/'artifact_manifest.json').is_file():
        raise ValueError('Model download/verification did not provide artifact_manifest.json')
    run.update('model_ready', model_dir=str(target), model_manifest_sha256=digest(target/'artifact_manifest.json'))
    return target


GPU_CHECK = """
import torch, transformers, vllm
print('Versions:',torch.__version__,torch.version.cuda,transformers.__version__,vllm.__version__)
print('CUDA architectures:',torch.cuda.get_arch_list())
assert torch.cuda.device_count()==2, 'This profile requires two T4 GPUs'
for index in range(2):
    print(index,torch.cuda.get_device_name(index),torch.cuda.get_device_properties(index).total_memory)
    assert 'T4' in torch.cuda.get_device_name(index), 'Selected hardware is not T4 x2'
    tensor=torch.ones((16,16),device=f'cuda:{index}',dtype=torch.float16)
    result=tensor@tensor
    torch.cuda.synchronize(index)
print('FP16 CUDA check passed; model kernels are checked by the actual run')
"""


def benchmark(run, config):
    arm, phase = config['arm'], config['phase']
    if run.details.exists():
        raise ValueError('This phase already has output in the session. Start a new saved version for a fresh run.')
    inputs = config.get('input_root', '/kaggle/input')
    if config.get('kit_root'):
        upload=Path(config['kit_root'])
        uploaded=json.loads((upload/'kaggle_kit.json').read_text())
        source={'git_commit':uploaded['baseline_commit']}
        baseline=upload/'baseline'
        kit=upload/'benchmark'
        baseline_hash=uploaded['source_archives']['baseline']['sha256']
    else:
        baseline_packet = unique_input(inputs, 'kaggle-baseline-source.zip')
        with zipfile.ZipFile(baseline_packet) as packet:
            try:
                source = json.loads(packet.comment)
            except ValueError as exc:
                raise ValueError('Baseline ZIP uses the old packaging; attach the new upload kit') from exc
        baseline = unpack(baseline_packet, run.work / 'baseline')
        kit = unpack(unique_input(inputs, 'kaggle-runtime24.zip'), run.work / 'dataset-kit')
        baseline_hash=digest(baseline_packet)
    if source.get('git_commit') != 'cc2ea9842d88a84fa6f9982d0dad3dbf6686ce76':
        raise ValueError('Wrong baseline commit')
    manifest = json.loads((kit / 'benchmark_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = (kit / relative).resolve()
        if not path.is_relative_to(kit.resolve()) or digest(path) != expected:
            raise ValueError('Dataset hash mismatch: ' + relative)
    python = environment(run, gpu=True)
    env = {'CUDA_VISIBLE_DEVICES': '0,1', 'TOKENIZERS_PARALLELISM': 'false'}
    env.update(config.get('extra_environment', {}))
    run.call(['nvidia-smi'], stage='record_gpu_hardware', timeout=30)
    run.call([python, '-c', GPU_CHECK], stage='check_two_t4_devices', env=env, timeout=120)
    model = download_model(run, config, python)
    # Download/install end here, before context preflight and the timed driver.
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    common = json.loads((baseline / 'configs/qwen35-9b.json').read_text())
    common.update(model_key=config.get('model_key','qwen35-9b'), model_dir=str(model),
                  split_root=str(kit/'dataset'), cache_dir=str(run.work/'preflight-cache'),
                  result_dir=str(run.work/'preflight-unused'), wall_time_seconds=config.get('wall_seconds',2700))
    if 'question_timeout_seconds' in config:
        value=config['question_timeout_seconds']
        if type(value) not in (int,float) or not 0<value<float('inf'):
            raise ValueError('question_timeout_seconds must be a positive finite number')
        common['question_timeout_seconds']=value
    initial = run.work / 'preflight-config.json'
    save(initial, common)
    all_ids = [json.loads(line)['id'] for line in (kit/'dataset/questions.jsonl').read_text().splitlines() if line.strip()]
    ids_path = run.work / 'all-ids.json'
    save(ids_path, all_ids)
    preflight_path = run.work / 'context-preflight.json'
    checks = run.output/'checks'/run.stem
    checks.mkdir(parents=True, exist_ok=True)
    try:
        run.call([python, PROJECT/'tools/profile_runtime_context.py', '--project-root', baseline,
                  '--config', initial, '--ids', ids_path, '--output', preflight_path,
                  '--context-limit', config.get('context_limit',32768), '--history-reserve',config.get('history_reserve',8192)],
                 stage='measure_actual_context_without_weights', env=env, timeout=900)
    finally:
        if preflight_path.is_file():
            copy = checks/'context-preflight.json'
            shutil.copy2(preflight_path,copy)
            run.additional_files.append(copy)
    preflight = json.loads(preflight_path.read_text())
    if not preflight['ready']:
        raise ValueError('Context preflight failed; no truncation or silent question exclusion')
    common.update(max_context_tokens=preflight['suggested_context_tokens'],
                  backend='transformers' if arm=='torch' else 'vllm', quantization='none',
                  cache_dir=str(run.work/'timed-cache'))
    if arm == 'vllm':
        common.update(question_workers=config.get('workers',2))
        common['vllm'] = {'tensor_parallel_size':2, 'pipeline_parallel_size':1, 'dtype':'float16',
            'quantization':'none', 'load_format':'auto', 'kv_cache_dtype':'auto',
            'gpu_memory_utilization':config.get('gpu_fraction',.88), 'max_num_seqs':config.get('engine_seqs',2),
            'max_num_batched_tokens':config.get('prefill_tokens',2048), 'enable_chunked_prefill':True,
            'enable_prefix_caching':True, 'cache_prompt_prefix':False, 'group_by_business':True,
            'enforce_eager':True, 'cpu_offload_gb':0, 'extra_engine_args':{'disable_custom_all_reduce':True}}
    config_path = run.work / 'timed-config.json'
    save(config_path, common)
    shutil.copy2(config_path,checks/'used-config.json')
    run.additional_files.append(checks/'used-config.json')
    command = [python, PROJECT/'tools/benchmark_runtime.py', '--project-root', baseline if arm=='torch' else PROJECT,
               '--config', config_path, '--ids', kit/f'{phase}_ids.json', '--output', run.details]
    if arm=='torch':
        command += ['--transformers-dtype','float16']
    try:
        run.call(command, stage=f'run_{phase}_questions', env=env, timeout=common['wall_time_seconds']+90)
    finally:
        run.details.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preflight_path, run.details/'context-preflight.json')
        shutil.copy2(config_path, run.details/'used-config.json')
        save(run.details/'source-archives.json', {'current_sha256':config.get('current_zip_sha256'),
             'baseline_sha256':baseline_hash,'baseline_commit':source['git_commit']})
    metrics = json.loads((run.details/'benchmark_metrics.json').read_text())
    run.update('assess_measurement',questions=metrics['written'])
    if not metrics.get('execution_complete',metrics['valid_complete_run']):
        raise ValueError('Pipeline did not complete all selected questions; inspect metrics and log')
    clean=metrics['valid_complete_run'] and any(kind!='abstain' and count for kind,count in metrics['answer_types'].items())
    return {'state':'passed' if clean else 'needs_review','questions':metrics['written'],
            'execution_complete':True,'measurement_warnings':metrics.get('measurement_warnings',[]),
            'context_tokens':common['max_context_tokens'],'quality_verified':False,
            'inference_seconds':metrics['inference_seconds']}


def compare_results(run, config):
    inputs = config.get('input_root','/kaggle/input')
    folders, receipts = [], []
    for arm in ('torch','vllm'):
        packets=list(Path(inputs).rglob(f'{arm}_main_result.zip'))
        if len(packets)==1:
            folder = unpack(packets[0], run.work/arm)
        elif not packets:
            # Also accept our own result structure after Kaggle Dataset extraction.
            status_path=unique_input(inputs,f'{arm}_main_status.json')
            folder=status_path.parent
        else:
            raise ValueError(f'Attach only one main result for {arm}: {packets}')
        status = json.loads((folder/f'{arm}_main_status.json').read_text())
        print(f"{arm}: đọc {folder}; trạng thái={status.get('state')}; số câu={status.get('questions')}",flush=True)
        legacy_completed_failure=(status.get('state')=='failed' and status.get('failed_stage')=='run_main_questions')
        if status.get('state') not in {'passed','needs_review'} and not legacy_completed_failure:
            raise ValueError(f'{arm} main did not pass. Read {arm}_main_status.json and {arm}_main_run.log first')
        result_folder=folder/'details'/f'{arm}_main'
        if not (result_folder/'benchmark_metrics.json').is_file():
            raise ValueError(f'Missing details/{arm}_main/benchmark_metrics.json in saved result')
        folders.append(result_folder)
        receipts.append(status)
    output = run.output/'comparison.json'
    command = [sys.executable, PROJECT/'tools/compare_runtime_benchmarks.py', '--baseline',folders[0],
               '--candidate',folders[1],'--output',output]
    if config.get('references'):
        command += ['--references',config['references']]
    run.call(command, stage='compare_saved_main_results', timeout=120)
    result = json.loads(output.read_text())
    result['run_receipts']=[{'state':r.get('state'),'failed_stage':r.get('failed_stage')} for r in receipts]
    result['gates']['both_receipts_clean']=all(r.get('state')=='passed' for r in receipts)
    if not result['gates']['both_receipts_clean']:
        result.update(paired_comparison_valid=False,inference_speedup=None,cold_speedup=None)
    save(output,result)
    fields = [('inference_seconds','Inference giây'),('load_seconds','Load giây'),
              ('cold_total_seconds','Tổng run_batch giây'),('questions_per_minute','Câu/phút'),
              ('latency_p50_seconds','Latency p50 giây'),('latency_p95_seconds','Latency p95 giây'),
              ('model_calls','Số lượt model'),('input_tokens','Token input'),('output_tokens','Token output'),
              ('cached_tokens','Token cache hit'),('generation_requests_peak','Generate đồng thời cao nhất'),
              ('context_overflow_calls','Lượt vượt context'),('oom_calls','Lượt OOM exception'),('timeout_calls','Lượt timeout'),
              ('possible_time_stop_calls','Lượt có khả năng dừng do thời gian')]
    table_path = run.output/'comparison.csv'
    with table_path.open('w',newline='',encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric','Transformers','vLLM'])
        for key,label in fields:
            writer.writerow([label,result['baseline'].get(key),result['candidate'].get(key)])
    labels={'same_ids_complete':'cùng bộ câu và đủ kết quả', 'same_model_manifest':'cùng checkpoint',
            'expected_backend_pair':'đúng nhánh Transformers/vLLM', 'token_metadata_complete':'đủ số đo token',
            'same_explicit_float16_precision':'cùng FP16', 'same_dataset_hashes':'cùng dữ liệu đầu vào',
            'same_visible_devices':'cùng cấu hình GPU nhìn thấy', 'same_pipeline_settings':'cùng cấu hình agent',
            'same_initial_requests':'cùng prompt đầu', 'same_runtime_libraries':'cùng phiên bản thư viện',
            'same_gpu_type_count':'cùng loại/số GPU', 'both_runs_valid':'hai lượt hoàn tất hợp lệ',
            'nonquantized_primary':'hai bên đều không quantize', 'not_all_abstain':'không phải toàn bộ câu đều bỏ trả lời',
            'no_deadline_limited_questions':'không có câu bị giới hạn thời gian',
            'no_infrastructure_errors':'không có lỗi hạ tầng', 'both_receipts_clean':'status hai lượt không có cảnh báo/lỗi cũ',
            'finite_positive_times':'đủ thời gian đo hợp lệ'}
    text = ['SO SÁNH HAI LƯỢT MAIN', 'Loại phép đo: toàn pipeline, Torch tuần tự và vLLM theo số worker đã cấu hình.',
            'Đây không phải phép đo engine trên cùng chuỗi request cố định.',
            'Đủ điều kiện so sánh không bị giới hạn thời gian: '+('Có' if result['paired_comparison_valid'] else 'Chưa')]
    if result['inference_speedup'] is not None:
        text.append(f"Speedup inference pipeline = TF/vLLM = {result['inference_speedup']:.3f}; >1 là pipeline vLLM nhanh hơn trong lượt này.")
    else:
        text.append('Chưa được kết luận nhanh hơn. Các điều kiện chưa đạt: '+', '.join(labels.get(k,k) for k,v in result['gates'].items() if not v))
    if result['observed_inference_time_ratio'] is not None:
        cold_ratio=result['observed_cold_time_ratio']
        cold_label=f'{cold_ratio:.3f}' if cold_ratio is not None else 'chưa đo'
        text.append(f"Tỷ số thời gian quan sát TF/vLLM: sau load={result['observed_inference_time_ratio']:.3f}; gồm load={cold_label}.")
        text.append('Tỷ số quan sát giữ tất cả câu, kể cả timeout; không dùng nó để tuyên bố engine nhanh hơn hoặc chất lượng bằng nhau.')
    outcomes={'answered':'có đáp án, chưa chấm đúng sai','abstain':'abstain không do deadline',
              'deadline_limited':'bị giới hạn thời gian','infrastructure_error':'lỗi hạ tầng'}
    text.append('Phân loại kết quả Torch/vLLM: '+json.dumps([{outcomes.get(k,k):v for k,v in side.items()}
                for side in result['outcome_counts_baseline_candidate']],ensure_ascii=False))
    text.append(f"Số câu khác type/answer: {len(result['answer_disagreement_ids'])}; agreement tính cả hai bên cùng abstain.")
    differences={'same_answer':'cùng type/answer','missing_result':'thiếu kết quả',
                 'infrastructure_affected_difference':'khác biệt có lỗi hạ tầng',
                 'deadline_affected_difference':'khác biệt có ảnh hưởng deadline',
                 'different_non_abstain_answers':'hai bên trả đáp án khác nhau','answer_vs_abstain':'đáp án so với abstain'}
    for row in result['per_question']:
        left,right=row['baseline'],row['candidate']
        delays=[side.get('first_model_delay_seconds') for side in (left,right)]
        delay_labels=[f'{v:.2f}' if v is not None else 'chưa đo' for v in delays]
        text.append(f"{row['id']}: {differences[row['difference_kind']]}; trước gọi model đầu TF/vLLM={'/'.join(delay_labels)} giây.")
        if row['first_action_difference']:
            delta=row['first_action_difference']
            text.append(f"  Action khác đầu tiên ở lượt {delta['generation_index']}; cùng messages={delta['same_request_messages']}. Xem comparison.json để đọc hai action.")
        feedback=row['first_feedback_difference']
        if feedback:
            text.append(f"  Phản hồi tool khác sau lượt {feedback['after_generation_index']}; chỉ khác elapsed_seconds={feedback['only_elapsed_seconds']}.")
    text.append('Không có nhãn thì chỉ đo timing/agreement, chưa chứng minh độ chính xác.')
    for key,label in fields:
        text.append(f"{label}: Transformers={result['baseline'].get(key)} | vLLM={result['candidate'].get(key)}")
    text_path=run.output/'comparison.txt'
    text_path.write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(text_path.read_text(encoding='utf-8'))
    run.additional_files.extend([output,table_path,text_path])
    return {'state':'passed' if result['paired_comparison_valid'] else 'needs_review',
            'paired_comparison_valid':result['paired_comparison_valid'], 'quality_verified':result['scores_baseline_candidate'] is not None}


def entrypoint(task, **config):
    if task not in {'benchmark','compare'}:
        raise ValueError('Unknown task')
    if task=='benchmark':
        if config.get('arm') not in {'torch','vllm'} or config.get('phase') not in {'pilot','main'}:
            raise ValueError('Use arm=torch/vllm and phase=pilot/main')
        stem=config['arm']+'_'+config['phase']
    else:
        stem='comparison'
    run=SavedRun(stem, output_root=config.get('output_root','/kaggle/working'),
                 work_root=config.get('work_root','/tmp/datamate-kaggle'))
    actions={'benchmark':benchmark,'compare':compare_results}
    try:
        run.update('validate_inputs',task=task)
        fields=actions[task](run,config)
        run.update('finished',**fields)
    except BaseException as exc:
        run.update('failed',state='failed',failed_stage=run.status['stage'],error=f'{type(exc).__name__}: {exc}')
        with run.log_path.open('a',encoding='utf-8') as handle:
            handle.write(traceback.format_exc())
        raise
    finally:
        run.finalize()
    return run.status
