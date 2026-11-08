"""Readable execution traces, rendered from recorded output without calling a model."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

from .domain import DataMateError, json_text


def _block(value, language='json'):
    text = value if isinstance(value, str) else json_text(value, indent=2)
    fence = '`' * max(3, max((len(m[0]) + 1 for m in re.finditer(r'`+', text)), default=0))
    return f'{fence}{language}\n{text}\n{fence}\n'


def _model_action(step):
    if 'llm_action' in step:
        return step['llm_action']
    # Older traces stored an action that the validator could later modify.
    # Recover the original only when raw output parses; never execute it here.
    from .agent.protocol import parse_action
    try:
        return parse_action(step.get('raw', ''))
    except DataMateError:
        return None


def render_debug_log(record):
    parts = [f"# Log từng bước — {record.get('id', '?')}\n",
             _block({k: record[k] for k in ('id', 'business_id', 'question', 'as_of_date',
                                           'elapsed_seconds', 'model_calls') if k in record}),
             '## Đáp án cuối của agent\n', _block(record.get('submission'))]
    if record.get('initial_messages'):
        parts.extend(['<details><summary>Prompt ban đầu gửi model</summary>\n',
                      _block(record['initial_messages']), '</details>\n'])
    if not record.get('trace'):
        parts.append('Không có bước thực thi được ghi lại.\n')
    for index, step in enumerate(record.get('trace', []), 1):
        model_action = _model_action(step)
        action = step.get('action') or model_action or {}
        label = action.get('action', 'lỗi trước khi có action hợp lệ')
        if step.get('phase')=='preparation':
            label='chuẩn bị doanh nghiệp / '+step.get('preparation_stage','cache')
        parts.append(f"## Bước {step.get('step', index)} — {label}\n")
        if step.get('input_messages'):
            parts.extend(['<details><summary>Đầu vào bước chuẩn bị gửi model</summary>\n',
                          _block(step['input_messages']), '</details>\n'])
        generation = step.get('generation')
        if generation:
            parts.extend(['### Thông tin lần gọi LLM\n', _block(generation)])
        if 'preparation_elapsed_seconds' in step:
            parts.append('Tổng thời gian chuẩn bị (gồm đọc dữ liệu/cache và model): '+str(step['preparation_elapsed_seconds'])+' giây.\n')
        if 'raw' in step:
            parts.extend(['### LLM trả về — nguyên văn\n', _block(step['raw'], 'text')])
        else:
            parts.append('Bước này không ghi nhận đầu ra LLM.\n')
        if model_action is not None:
            parts.extend(['### Action LLM sinh — trước khi bổ sung tham số\n', _block(model_action)])
            if model_action.get('action') == 'sql':
                parts.extend(['### SQL do LLM sinh\n', _block(model_action['sql'], 'sql'),
                              'Tham số SQL (giữ riêng, không nội suy vào câu lệnh):\n',
                              _block(model_action.get('parameters', []))])
            if model_action.get('action')=='plan' and isinstance(model_action.get('query'),dict):
                parts.extend(['### SQL do LLM sinh trong plan\n', _block(model_action['query'].get('sql'), 'sql'),
                              'Tham số SQL:\n',_block(model_action['query'].get('parameters',[]))])
        if action and action != model_action:
            parts.extend(['### Action đưa vào xử lý — sau parse/bổ sung tham số\n', _block(action)])
        if 'preflight' in step:
            parts.extend(['### Kiểm tra dữ liệu ban đầu\n', _block(step['preflight'])])
        observation = step.get('observation', {})
        if 'execution' in step:
            parts.extend(['### Thực thi tự động sau plan\n',_block(step['execution'])])
        if 'auto_final' in step:
            parts.extend(['### Code chiếu đáp án từ kết quả SQL\n',_block(step['auto_final'])])
        if 'sql' in observation:
            origin = ('do công cụ metric tạo' if action.get('action')=='metric' else
                      'do LLM sinh trong plan' if action.get('action')=='plan' and action.get('query') else
                      'do calculate biên dịch từ plan' if action.get('action') in {'calculate','plan'} else 'được công cụ ghi nhận')
            parts.extend([f'### SQL {origin}\n', _block(observation['sql'], 'sql'),
                          'Tham số SQL:\n', _block(observation.get('parameters', []))])
        if observation:
            parts.extend(['### Kết quả công cụ / phản hồi gửi lại LLM\n', _block(observation)])
        if 'evidence' in step:
            parts.extend(['### Bằng chứng dùng chốt đáp án\n', _block(step['evidence'])])
        if 'sql_validation' in step:
            parts.extend(['### Phạm vi kiểm tra SQL theo kế hoạch\n', _block(step['sql_validation'])])
        if 'error' in step:
            parts.extend(['### Lỗi xử lý\n', _block({k: step[k] for k in ('error', 'error_kind') if k in step})])
        if step.get('accepted'):
            parts.extend(['### Đáp án được chấp nhận tại bước này\n', _block(record.get('submission'))])
        if step.get('loop_stopped'):
            parts.append('Dừng do lặp action/kết quả không có tiến triển.\n')
        if 'type_normalized' in step:
            parts.extend(['### Chuẩn hóa kiểu đáp án\n', _block(step['type_normalized'])])
    return '\n'.join(parts)


def write_debug_log(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(render_debug_log(record), encoding='utf-8')
    temporary.replace(path)
    return path


def export_debug_logs(run_dir, output_dir=None):
    """Export existing JSON logs; leave source records and submission untouched."""
    import json
    run_dir = Path(run_dir).resolve()
    source = run_dir / 'logs' if (run_dir / 'logs').is_dir() else run_dir
    destination = Path(output_dir).resolve() if output_dir else run_dir / 'readable_logs'
    if destination == source:
        raise DataMateError('Chọn thư mục xuất riêng với thư mục JSON nguồn')
    records = []
    for path in sorted(source.glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(record, dict) and 'id' in record and isinstance(record.get('trace'), list):
            records.append((path, record))
    if not records:
        raise DataMateError('Không tìm thấy JSON log từng câu có id và trace')
    destination.mkdir(parents=True, exist_ok=True)
    index = ['# Log debug từng câu\n',
             'Mở một câu để xem nguyên văn LLM, SQL, tham số, kết quả/lỗi và đáp án cuối. '
             'SQL do LLM sinh và SQL do công cụ metric tạo được ghi riêng. '
             'Các file này được xuất từ log có sẵn, không chạy lại inference.\n',
             '| Câu | Type cuối | Số bước | Số bước có lỗi |', '|---|---|---:|---:|']
    for source_path, record in records:
        path = write_debug_log(destination / source_path.with_suffix('.md').name, record)
        errors = sum(bool(step.get('error') or step.get('observation', {}).get('error')) for step in record['trace'])
        label = str(record['id']).replace('|', '\\|').replace('\n', ' ')
        index.append(f"| [{label}]({quote(path.name)}) | {record.get('submission', {}).get('type', '?')} | {len(record['trace'])} | {errors} |")
    (destination / 'index.md').write_text('\n'.join(index) + '\n', encoding='utf-8')
    return destination, len(records)
