"""Organizer log fields, derived from the actual result and execution trace.

The detailed trace remains available. These fields do not infer hidden reasoning
from an action's `reason`, and never reconstruct model text that was not logged.
"""
from __future__ import annotations

from .domain import RunResult


def _reasoning(event):
    generation = event.get('generation') or {}
    explicit = generation.get('reasoning_content')
    if isinstance(explicit, str):
        return explicit
    raw = event.get('raw', '')
    if not isinstance(raw, str):
        return ''
    start = raw.find('<think>')
    if start >= 0:
        start += len('<think>')
        # Preserve multiple blocks and incomplete reasoning exactly as generated.
        end = raw.rfind('</think>')
        return raw[start:end if end >= start else len(raw)]
    if '</think>' in raw:
        # Some chat templates supply the opening tag in the prompt.
        return raw[:raw.index('</think>')]
    return ''


def competition_fields(run: RunResult):
    calls = [event for event in run.trace if 'raw' in event]
    calls.sort(key=lambda event:event.get('model_call_started',float('inf')))
    reasoning = [_reasoning(event) for event in calls]
    thinking = (reasoning if len(calls) > 1 else reasoning[0]) if any(reasoning) else None
    executed = []
    for event in run.trace:
        result = event.get('execution')
        if not isinstance(result, dict):
            observation = event.get('observation')
            if isinstance(observation, dict):
                result = observation if 'sql' in observation else observation.get('sample')
        if not isinstance(result, dict) or not isinstance(result.get('sql'), str):
            continue
        executed.append({'step': event.get('step'),
                         'source': 'src/datamate/data/sql.py::SQLExecutor.execute',
                         'sql': result['sql'], 'parameters': result.get('parameters', []),
                         'columns': result.get('columns'), 'rows': result.get('rows'),
                         'truncated': result.get('truncated', False)})
    fields = {'answer': run.answer.to_dict(),
            'elapsed_ms': max(0, round(run.elapsed_seconds * 1000)),
            'thinking': thinking,
            'code': '\n\n'.join(item['sql'] for item in executed) if executed else None,
            'inputs': [{'step': item['step'], 'source': item['source'],
                        'parameters': item['parameters']} for item in executed] or None,
            'output': [{'columns': item['columns'], 'rows': item['rows'],
                        'truncated': item['truncated']} for item in executed] or None}
    if run.executions is not None:
        ledger=run.executions
        fields.update(code='\n\n'.join(
            entry['inputs']['sql'] if entry['source'].endswith('SQLExecutor.execute') else entry['source']
            for entry in ledger) or None,
            inputs=[{'sequence':e['sequence'],'source':e['source'],'arguments':e['inputs']} for e in ledger] or None,
            output=[{'sequence':e['sequence'],**({'value':e['output']} if 'output' in e else {'error':e['error']})} for e in ledger] or None)
    return fields
