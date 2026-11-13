"""Measure actual first model requests through the unchanged agent, without weights/GPU."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys


class CapturedRequest(BaseException):
    """Exit the profiling call before generation; not an agent answer."""


def load_settings(settings_class, config, **overrides):
    options = {'use_env': False} if 'use_env' in inspect.signature(settings_class.load).parameters else {}
    return settings_class.load(config, **options, **overrides)


def profile(project, config, ids_path, output, *, context_limit=32768, history_reserve=8192):
    project = Path(project).resolve()
    sys.path.insert(0, str(project / 'src'))
    from datamate.config import Settings
    from datamate.agent.runner import Agent
    from datamate.data.split import DatasetSplit
    from transformers import AutoTokenizer
    import datamate
    if not Path(datamate.__file__).resolve().is_relative_to(project):
        raise ValueError('Wrong imported source tree; run this tool in a fresh process')
    settings = load_settings(Settings, config)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, local_files_only=True, trust_remote_code=False)
    ids = json.loads(Path(ids_path).read_text(encoding='utf-8-sig'))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Nonempty unique question IDs are required')
    selected = set(ids)
    split = DatasetSplit(settings.split_root)
    questions = [q for q in split.questions() if q.id in selected]
    if {q.id for q in questions} != selected:
        raise ValueError('IDs do not belong to this split')

    class CaptureBackend:
        counts_as_model = supports_token_budget = True
        last_generation = None
        record = None
        def generate(self, messages, remaining_seconds, *, max_new_tokens=None):
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_dict=True, enable_thinking=False)
            token_ids = encoded['input_ids']
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            budget = settings.max_new_tokens if max_new_tokens is None else max_new_tokens
            self.record = {'input_tokens': len(token_ids), 'output_reserved': budget,
                           'message_chars': sum(len(m['content']) for m in messages),
                           'messages_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            raise CapturedRequest()
        def close(self):
            pass

    backend = CaptureBackend()
    agent = Agent(settings, backend, split)
    rows = []
    for question in questions:
        backend.record = None
        try:
            result = agent.run(question)
        except CapturedRequest:
            rows.append({'id': question.id, 'business_id': question.business_id, **backend.record})
        else:
            rows.append({'id': question.id, 'business_id': question.business_id,
                         'error': 'Agent returned before reaching a model call', 'trace': result.trace})
    measured = [r for r in rows if 'input_tokens' in r]
    required = max((r['input_tokens'] + r['output_reserved'] for r in measured), default=0) + history_reserve
    suggested = max(16384, math.ceil(required / 4096) * 4096)
    ready = len(measured) == len(questions) and suggested <= context_limit
    report = {'source_root': str(project), 'settings': settings.to_dict(), 'ids': ids, 'questions': rows,
              'method': 'first real generate request from Agent.run; interrupted before any weights/inference',
              'history_reserve': history_reserve, 'suggested_context_tokens': suggested,
              'context_limit': context_limit, 'ready': ready,
              'warning': 'Later tool history/repair may be longer. No truncation or synthetic answers were performed.'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--ids', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context-limit', type=int, default=32768)
    parser.add_argument('--history-reserve', type=int, default=8192)
    args = parser.parse_args()
    if args.context_limit < 2048 or args.history_reserve < 0:
        parser.error('Invalid context/history budgets')
    report = profile(args.project_root, args.config, args.ids, args.output,
                     context_limit=args.context_limit, history_reserve=args.history_reserve)
    print(json.dumps({k: report[k] for k in ('ready','suggested_context_tokens','context_limit','history_reserve')}))
    if not report['ready']:
        raise SystemExit('Context preflight failed: inspect output before spending GPU time')


if __name__ == '__main__':
    main()
