from __future__ import annotations

import hashlib
import time
from collections import Counter
from copy import deepcopy

from ..config import Settings
from ..data.ingest import Ingestor, quote
from ..data.split import DatasetSplit
from ..domain import Answer, DataMateError, Question, RunResult, json_text
from ..models.backends import ChatBackend
from .prompts import initial_messages
from .protocol import answer_from_query, parse_action, verify_legacy_answer
from .tools import BusinessTools
from .verification import question_contract, question_periods, verify_metric_request


class Agent:
    pipeline_version = 2
    def __init__(self, settings: Settings, backend: ChatBackend, split: DatasetSplit | None = None):
        self.settings, self.backend = settings, backend
        self.split = split or DatasetSplit(settings.split_root)
        self.ingestor = Ingestor(self.split, settings.cache_dir)

    def run(self, question: Question, remaining_seconds: float | None = None) -> RunResult:
        started = time.monotonic()
        budget = min(self.settings.question_timeout_seconds, remaining_seconds) if remaining_seconds is not None else self.settings.question_timeout_seconds
        deadline = started + budget
        trace, calls, reviewed = [], 0, False
        results, evidence, repeats = {}, {}, Counter()
        last_id = None
        answer = Answer('abstain', reason='Hết ngân sách bước hoặc thời gian')
        try:
            catalog = self.ingestor.catalog(question.business_id)
            tools = BusinessTools(catalog, self.split.rulebook, question.as_of_date, self.settings.sql_timeout_seconds, self.settings.max_result_rows, deadline)
            contract = question_contract(question.question)
            expected_periods = question_periods(question.question, question.as_of_date, tools.policy)
            first_period = {'kind': 'range', **{k: expected_periods[0][k] for k in ('start', 'end')}} if expected_periods else None
            preflight = tools.check(question.question, first_period)
            if preflight['availability'] == 'missing':
                evidence['e0'] = preflight
                preflight = {**preflight, 'evidence_id': 'e0'}
            messages = initial_messages(question, self.split.rulebook, self.split.readme(question.business_id), catalog.summary(),
                                        {'contract': contract, 'periods': expected_periods, 'answerability': preflight})
            for step in range(self.settings.max_steps):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                raw = self.backend.generate(messages, remaining)
                calls += int(getattr(self.backend, 'counts_as_model', True))
                event = {'step': step + 1, 'raw': raw}
                if step == 0:
                    event['preflight'] = preflight
                generation = getattr(self.backend, 'last_generation', None)
                if generation:
                    event['generation'] = generation
                stage, action = 'action_parse', None
                try:
                    action = parse_action(raw)
                    event['llm_action'] = deepcopy(action)
                    event['action'] = action
                    kind = action['action']
                    messages.append({'role': 'assistant', 'content': json_text(action)})
                    if kind == 'final':
                        stage = 'answer_projection'
                        if action['type'] == 'abstain':
                            candidate = Answer.from_dict(action)
                        elif action['type'] == 'unanswerable':
                            proof = evidence.get(action.get('evidence_id', 'e0'))
                            if not proof or proof.get('availability') != 'missing' or not proof.get('missing'):
                                raise DataMateError('unanswerable cần evidence_id do check/metric xác nhận thiếu thông tin; query rỗng không đủ. Chưa kiểm chứng thì abstain')
                            candidate = Answer('unanswerable', reason=action.get('reason') or ', '.join(proof['missing']))
                            event['evidence'] = proof
                        else:
                            proof = evidence.get('e0')
                            if proof and proof.get('availability') == 'missing':
                                raise DataMateError('Câu hỏi đang thiếu thông tin đã được check xác nhận: ' + str(proof['missing']))
                            result_id = action.get('query_id', last_id)
                            result = results.get(result_id)
                            if result is None:
                                raise DataMateError('Không có query_id thành công; dùng metric/SQL trước')
                            if contract['metrics'] and tools.policy.sales_statuses and not result.get('verified'):
                                raise DataMateError('Query SQL thô chưa kiểm chứng đại lượng này. Dùng action metric ' + str(contract['metrics']) + '; không chốt từ sample/COUNT kiểm tra')
                            if not contract['metrics'] and tools.policy.sales_statuses and not result.get('verified'):
                                tools.verify_fact_sql(result,question.question)
                            answer_type = action['type']
                            expected = contract.get('type') or result.get('answer_type')
                            if expected and answer_type != expected:
                                if (answer_type, expected) in {('string', 'date'), ('string', 'boolean'), ('number', 'boolean')}:
                                    event['type_normalized'] = {'from': answer_type, 'to': expected}
                                    answer_type = expected
                                else:
                                    raise DataMateError('type không khớp câu hỏi/công cụ; cần ' + expected)
                            candidate = answer_from_query(answer_type, result, action.get('columns'),
                                                          percent_places=tools.policy.percent_places)
                            verify_legacy_answer(action, candidate)
                            event['query_id'] = result_id
                        if self.settings.mode == 'self_review' and not reviewed and candidate.type != 'abstain':
                            reviewed = True
                            event['review_requested'] = True
                            feedback = {'review': 'Kiểm tra query_id, đại lượng, kỳ và bằng chứng. Dùng metric/query độc lập nếu cần; sau đó final.', 'candidate': candidate.to_dict()}
                        else:
                            answer = candidate
                            event['accepted'] = True
                            trace.append(event)
                            break
                    elif kind == 'schema':
                        stage = 'schema'
                        raw_table = next((t for t in catalog.tables if t['name'] == action['table']), None)
                        fact = next((t for t in catalog.semantic['tables'] if t['name'] == action['table']), None)
                        binding = next((b for b in catalog.semantic['bindings'] if b.get('clean_table') == action['table']), None)
                        table = raw_table or fact
                        if table is None and binding:
                            table = {'name': binding['clean_table'], 'roles': binding['roles']}
                        if table is None:
                            raise DataMateError('Bảng không có trong catalog')
                        preview = tools.executor.execute(f"SELECT * FROM {quote(action['table'])} LIMIT 3", remaining_seconds=deadline - time.monotonic())
                        feedback = {'schema': table, 'sample': preview, 'full_column_profiles': tools.profile(action['table']), 'sample_is_not_distribution': True}
                    elif kind == 'profile':
                        stage = 'profile'
                        feedback = tools.profile(action['table'], action.get('column'), action.get('offset', 0), action.get('limit', 50))
                    elif kind == 'lookup':
                        stage = 'lookup'
                        feedback = tools.lookup(action.get('entity'), action.get('text'), action.get('limit', 10))
                    elif kind == 'period':
                        stage = 'period'
                        feedback = tools._period(action.get('period'))
                    elif kind == 'check':
                        stage = 'answerability'
                        period = first_period
                        if action.get('period'):
                            resolved = tools._period(action['period'])
                            if not expected_periods or any(resolved[k] != expected_periods[0][k] for k in ('start', 'end')):
                                raise DataMateError('check period phải khớp kỳ trong câu hỏi; không chọn một kỳ ngoài dữ liệu để kết luận thiếu')
                            period = action['period']
                        feedback = tools.check(question.question, period)
                        if feedback['availability'] == 'missing':
                            proof_id = 'e' + str(len(evidence) + 1)
                            evidence[proof_id] = feedback
                            feedback = {**feedback, 'evidence_id': proof_id}
                    else:
                        stage = 'metric' if kind == 'metric' else 'sql_execution'
                        if kind == 'metric':
                            verify_metric_request(action, question.question, question.as_of_date, tools.policy)
                            feedback = tools.metric(action)
                            if feedback.get('availability') == 'missing':
                                if not contract['metrics'] or action['name'] not in contract['metrics']:
                                    raise DataMateError('Bằng chứng thiếu dữ liệu không gắn với đại lượng của câu hỏi')
                                proof_id = 'e' + str(len(evidence) + 1)
                                evidence[proof_id] = feedback
                                feedback = {**feedback, 'evidence_id': proof_id}
                        else:
                            feedback = tools.executor.execute(action['sql'], action.get('parameters'), deadline - time.monotonic())
                            feedback['verified'] = False
                            if contract['metrics'] and tools.policy.sales_statuses:
                                feedback['next_action'] = {'action': 'metric', 'name': contract['metrics'][0]}
                                feedback['warning'] = 'SQL thô chỉ dùng kiểm tra; final của đại lượng này cần metric đã kiểm chứng. Trạng thái raw không đồng nghĩa nhãn canonical.'
                        if 'rows' in feedback:
                            last_id = 'r' + str(len(results) + 1)
                            feedback = {**feedback, 'query_id': last_id}
                            results[last_id] = feedback
                    feedback = {**feedback, 'remaining_steps': self.settings.max_steps - step - 1}
                    event['observation'] = feedback
                except (DataMateError, KeyError, TypeError, ValueError) as exc:
                    if not action:
                        messages.append({'role': 'assistant', 'content': '{"protocol_error":true}'})
                    if action and action['action'] in {'sql', 'metric'}:
                        last_id = None
                    feedback = {'error': str(exc), 'error_kind': stage, 'remaining_steps': self.settings.max_steps - step - 1,
                                'repair': 'Chỉ sửa lỗi được báo, dùng đúng một action. Không lặp query/final chưa thay đổi; không chắc thì abstain.'}
                    event['observation'] = feedback
                trace.append(event)
                repeated_action = {k:v for k,v in action.items() if k not in {'reason','from_query'}} if action else raw
                signature = hashlib.sha256(json_text({'action': repeated_action, 'error': feedback.get('error'),
                    'rows': feedback.get('rows'), 'availability': feedback.get('availability')}, sort_keys=True).encode()).hexdigest()
                repeats[signature] += 1
                if repeats[signature] >= self.settings.max_repeated_actions:
                    answer = Answer('abstain', reason='Lặp cùng action/kết quả không có tiến triển; dừng để tiết kiệm ngân sách')
                    event['loop_stopped'] = True
                    break
                messages.append({'role': 'user', 'content': json_text({'tool_result': feedback})})
        except Exception as exc:
            trace.append({'error': f'{type(exc).__name__}: {exc}', 'error_kind': 'runtime'})
            answer = Answer('abstain', reason=f'Lỗi xử lý: {str(exc)[:250]}')
        return RunResult(answer, time.monotonic() - started, trace, calls)
