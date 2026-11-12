"""Complete, integrity-checked single-GPU Trainer checkpoints on persistent storage."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid
import contextlib
import functools
import hashlib
import tempfile

from ..domain import DataMateError, json_text
from ..models.artifacts import sha256

MARKER = 'checkpoint_complete.json'


def atomic_json(path, value):
    """Publish metadata only after the temporary file has been fully flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temporary.open('x', encoding='utf-8') as handle:
        handle.write(json_text(value, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _metadata_object(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _preserve_incomplete(path):
    if path.exists():
        path.replace(path.with_name(path.name+'.incomplete-'+uuid.uuid4().hex))


def ensure_training_config(path, config):
    path = Path(path)
    saved = _metadata_object(path) if path.exists() else None
    if saved is not None:
        if saved != config:
            raise DataMateError('Cấu hình thay đổi: dùng RUN_TAG mới')
        return
    # A damaged notebook config is recoverable: prepare_run still checks the
    # checkpoint's embedded identity before allowing another training update.
    _preserve_incomplete(path)
    atomic_json(path, config)


def prepare_run(output, request, identity):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()) and not request:
        raise DataMateError('output_dir đã có; dùng resume hoặc chọn thư mục mới')
    path = root/'run_identity.json'
    saved = _metadata_object(path) if path.exists() else None
    if saved is not None and saved != identity:
        raise DataMateError('Resume khác model/dataset/cấu hình; không trộn checkpoint')
    checkpoint = select_checkpoint(root, request, identity, quarantine=True)
    # Never repair root metadata until a matching checkpoint was verified (or
    # the output was confirmed empty). Corruption cannot silently restart a run.
    if saved is None:
        _preserve_incomplete(path)
        atomic_json(path, identity)
    return checkpoint


@contextlib.contextmanager
def training_lock(output):
    from filelock import FileLock, Timeout
    name=hashlib.sha256(os.path.normcase(str(Path(output).resolve())).encode('utf-8')).hexdigest()
    folder=Path(tempfile.gettempdir())/'datamate-training-locks'
    folder.mkdir(parents=True,exist_ok=True)
    lock=FileLock(folder/(name+'.lock'))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise DataMateError('Lượt train cùng output_dir đang chạy trên runtime này; kết nối lại để xem log') from exc
    try:
        yield
    finally:
        lock.release()


def exclusive_training_run(function):
    @functools.wraps(function)
    def wrapped(config_path, resume=None):
        config=json.loads(Path(config_path).read_text(encoding='utf-8'))
        with training_lock(config['output_dir']):
            return function(config_path,resume)
    return wrapped


def completed_report(output, config):
    root=Path(output).resolve()
    path=root/'training_report.json'
    if not path.is_file(): return None
    try:
        report=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(report,dict): raise ValueError('report must be an object')
        if report['training_config'] != config:
            raise DataMateError('Báo cáo đã hoàn tất nhưng cấu hình khác; dùng RUN_TAG mới')
        files=report.get('adapter_files',{})
        if not isinstance(files,dict): raise ValueError('adapter_files must be an object')
        if 'adapter_model.safetensors' not in files:
            raise DataMateError('Báo cáo cũ chưa có checksum trọng số; kiểm tra bằng source cùng phiên bản')
        for name,digest in files.items():
            file=(root/name).resolve()
            if not file.is_relative_to(root) or not file.is_file() or sha256(file)!=digest:
                raise DataMateError('Adapter đã lưu sai checksum: '+name)
        return report
    except (OSError,ValueError,KeyError) as exc:
        raise DataMateError('Không đọc được báo cáo hoàn tất; giữ run cũ và kiểm tra checkpoint') from exc


def build_run_identity(config, base_sha256, dataset_sha256):
    defaults = {'method':'lora','rank':16,'alpha':32,'dropout':0.05,'max_length':16384,
                'batch_size':1,'gradient_accumulation_steps':8,'epochs':2,'max_steps':-1,
                'learning_rate':0.0001,'optimizer':'adamw_torch','seed':42,
                'smoke_longest_only':False,'save_steps':5,'eval_steps':50,'save_interval_seconds':600,
                'save_total_limit':3,'require_fast_kernels':False,'logging_steps':1}
    resolved = {**defaults, **{k:v for k,v in config.items() if k not in
                              {'base_dir','dataset_dir','output_dir','cache_dir'}}}
    if resolved.get('target_suffixes') is not None:
        resolved['target_suffixes'] = sorted(set(resolved['target_suffixes']))
    return {'version':2,'training':resolved,'base_manifest_sha256':base_sha256,
            'dataset_manifest_sha256':dataset_sha256}


def _files(path):
    required = {'trainer_state.json','optimizer.pt','scheduler.pt','training_args.bin','rng_state.pth'}
    if (path/'adapter_config.json').is_file():
        required.add('adapter_config.json')
        weights = ['adapter_model.safetensors','adapter_model.bin']
    else:
        weights = ['model.safetensors','pytorch_model.bin']
    if not any((path/name).is_file() and (path/name).stat().st_size for name in weights):
        raise DataMateError('Checkpoint thiếu trọng số: '+str(path))
    if not all((path/name).is_file() and (path/name).stat().st_size for name in required):
        raise DataMateError('Checkpoint thiếu Trainer/optimizer/scheduler/RNG state: '+str(path))
    result = {}
    for file in sorted(path.rglob('*')):
        if file.is_symlink(): raise DataMateError('Checkpoint không nhận symlink')
        if file.is_file() and file.name != MARKER and not file.name.startswith(MARKER+'.tmp-'):
            result[file.relative_to(path).as_posix()] = sha256(file)
    return result


def _step(path):
    try:
        step = int(path.name.removeprefix('checkpoint-'))
        state = json.loads((path/'trainer_state.json').read_text(encoding='utf-8'))
        if (not isinstance(state,dict) or type(state.get('global_step')) is not int
                or step < 1 or state['global_step'] != step): raise ValueError('step mismatch')
        return step
    except (OSError, ValueError, KeyError) as exc:
        raise DataMateError('Checkpoint có global_step không hợp lệ: '+str(path)) from exc


def verify_checkpoint(path, identity):
    path = Path(path).resolve()
    try:
        marker = json.loads((path/MARKER).read_text(encoding='utf-8'))
        if not isinstance(marker,dict): raise ValueError('marker must be an object')
        if marker.get('format') != 1 or marker.get('run_identity') != identity:
            raise DataMateError('Checkpoint khác model/dataset/cấu hình: '+str(path))
        if marker.get('global_step') != _step(path) or marker.get('files') != _files(path):
            raise DataMateError('Checkpoint đang ghi dở hoặc sai checksum: '+str(path))
        return marker
    except (OSError, ValueError, KeyError) as exc:
        raise DataMateError('Checkpoint chưa có marker hoàn tất hợp lệ: '+str(path)) from exc


def seal_checkpoint(path, identity):
    path = Path(path).resolve()
    if (path/MARKER).exists(): return verify_checkpoint(path, identity)
    marker = {'format':1,'global_step':_step(path),'run_identity':identity,'files':_files(path)}
    atomic_json(path/MARKER,marker)
    return marker


def select_checkpoint(output, request, identity, *, quarantine=False):
    root = Path(output).resolve()
    if not request: return None
    if request != 'auto':
        path = Path(request).resolve()
        if path.parent != root or path.is_symlink():
            raise DataMateError('Checkpoint resume phải nằm trực tiếp trong output_dir')
        verify_checkpoint(path, identity)
        return path
    candidates = []
    for path in root.glob('checkpoint-*'):
        if path.is_dir() and not path.is_symlink() and path.name.removeprefix('checkpoint-').isdigit():
            candidates.append(path)
    if not candidates:
        if root.exists() and any(root.iterdir()):
            raise DataMateError('Chưa có checkpoint hoàn tất để resume; giữ thư mục cũ và dùng RUN_TAG mới nếu chạy từ đầu')
        return None
    rejected, rejected_paths = [], []
    for path in sorted(candidates, key=lambda p:int(p.name.removeprefix('checkpoint-')), reverse=True):
        try:
            verify_checkpoint(path, identity)
            if rejected: print('Bỏ qua checkpoint chưa hợp lệ:', '; '.join(rejected), flush=True)
            if quarantine:
                for invalid in rejected_paths:
                    destination = root/('incomplete-'+invalid.name+'-'+uuid.uuid4().hex[:12])
                    if invalid.resolve().parent != root or destination.resolve().parent != root:
                        raise DataMateError('Checkpoint quarantine vượt output_dir')
                    invalid.rename(destination)
                    print('Giữ checkpoint ghi dở tại:',destination,flush=True)
            return path
        except DataMateError as exc:
            rejected.append(str(exc))
            rejected_paths.append(path)
    raise DataMateError('Không có checkpoint đầy đủ khớp run hiện tại: '+'; '.join(rejected))


def recover_completed_checkpoint(trainer, path, identity):
    """Finalize an already finished run without Trainer taking an extra step."""
    verify_checkpoint(path, identity)
    from transformers import TrainerState
    state = TrainerState.load_from_json(str(Path(path)/'trainer_state.json'))
    if state.max_steps <= 0 or state.global_step < state.max_steps:
        return None
    # This private API is pinned by the Colab requirements and exercised with
    # a real PEFT adapter in the interruption/resume regression test.
    trainer._load_from_checkpoint(str(path))
    trainer.state = state
    metrics = {'completion_recovered_from_checkpoint':True,
               'global_step':state.global_step, 'epoch':state.epoch}
    for record in reversed(state.log_history):
        if 'loss' in record:
            metrics['last_logged_loss'] = record['loss']
            break
    print('Checkpoint đã đủ số bước; khôi phục adapter/báo cáo, không cập nhật trọng số thêm.',flush=True)
    return metrics


def make_checkpoint_callback(identity, interval_seconds=600):
    from transformers import TrainerCallback
    class CompleteCheckpointCallback(TrainerCallback):
        def __init__(self):
            self.last_save = time.monotonic()
            self.step_started = None

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_started = time.monotonic()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if self.step_started is not None and getattr(args,'output_dir',None):
                import torch
                if torch.cuda.is_available(): torch.cuda.synchronize()
                record = {'global_step':state.global_step,
                          'optimizer_step_seconds':time.monotonic()-self.step_started}
                with (Path(args.output_dir)/'step_timing.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json_text(record)+'\n')
                    handle.flush()
            if state.global_step == 1 or time.monotonic()-self.last_save >= interval_seconds:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            path = Path(args.output_dir)/f'checkpoint-{state.global_step}'
            seal_checkpoint(path, identity)
            self.last_save = time.monotonic()
            print(f'Checkpoint hoàn tất: {path}', flush=True)
            return control
    return CompleteCheckpointCallback()
