"""Verify organizer input/output/log contracts after an actual Docker run.

This is a format/completeness check, not a score or proof of GPU compatibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from datamate.domain import Answer, DataMateError, Question


def _jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_result(data, result):
    data, result = Path(data), Path(result)
    questions = _jsonl(data / 'questions.jsonl')
    expected = {Question.from_dict(row).id for row in questions}
    if not expected or len(expected) != len(questions):
        raise ValueError('Input must contain distinct question IDs')
    answers = _jsonl(result / 'submission.jsonl')
    by_id = {}
    for row in answers:
        identifier = row.get('id')
        if identifier in by_id or identifier not in expected:
            raise ValueError('Submission has duplicate/unknown IDs')
        try:
            Answer.from_dict(row)
        except DataMateError as exc:
            raise ValueError('Invalid submission type/value: '+identifier) from exc
        by_id[identifier] = row
    if set(by_id) != expected:
        raise ValueError('Submission is missing questions')
    seen = set()
    for path in sorted((result / 'logs').glob('*.json')):
        raw = path.read_text(encoding='utf-8')
        if len(raw.splitlines()) != 1:
            raise ValueError('Each question log must occupy one line: '+path.name)
        log = json.loads(raw)
        identifier = log.get('id')
        if identifier not in expected or identifier in seen:
            raise ValueError('Log has duplicate/unknown/missing question ID')
        seen.add(identifier)
        if not {'id', 'answer', 'elapsed_ms', 'thinking'} <= log.keys():
            raise ValueError('Missing organizer-required log fields')
        if type(log['elapsed_ms']) is not int or log['elapsed_ms'] < 0:
            raise ValueError('Log elapsed_ms must be a nonnegative integer')
        thinking = log['thinking']
        if not (thinking is None or isinstance(thinking, str)
                or isinstance(thinking, list) and all(isinstance(s, str) for s in thinking)):
            raise ValueError('Log thinking must be raw string/list or null')
        if log['answer'] != {k:v for k,v in by_id[identifier].items() if k != 'id'}:
            raise ValueError('Log final answer differs from submission')
    if seen != expected:
        raise ValueError('Missing per-question logs')
    summary = json.loads((result / 'run_summary.json').read_text(encoding='utf-8'))
    if summary.get('error') or summary.get('time_budget_exhausted') or summary.get('written') != len(expected):
        raise ValueError('Run did not finish cleanly; preserve output and inspect logs')
    wall = summary.get('wall_time_seconds')
    if type(wall) not in (int, float) or not 0 <= wall <= 21600:
        raise ValueError('Invalid or over-six-hour run time')
    return {'questions':len(expected), 'answers':len(answers), 'logs':len(seen),
            'format_verified':True, 'quality_evaluated':False,
            'application_wall_seconds':wall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_result(args.data, args.result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
