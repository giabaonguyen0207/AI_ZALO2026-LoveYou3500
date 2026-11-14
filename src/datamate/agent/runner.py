from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from copy import deepcopy

from ..config import Settings
from ..data.ingest import Ingestor, quote
from ..data.split import DatasetSplit
from ..domain import Answer, DataMateError, Question, RunResult, json_text
from ..models.backends import ChatBackend
from ..models.contracts import BackendUnavailableError
from .prompts import initial_messages
from .protocol import answer_from_query, parse_action, verify_legacy_answer
from .tools import BusinessTools
from .verification import question_contract, question_periods, verify_metric_request
from .preparation import BusinessPreparer
from .planning import validate_plan, calculate, verify_sql_plan
from .schema_repair import SchemaRepair, repair_needs
from ..retrieval import NullRetriever
from ..retrieval.contracts import RetrievalFatalError
from ..retrieval.context import fit_context


class Agent:
    pipeline_version = 6
    def __init__(self, settings: Settings, backend: ChatBackend, split: DatasetSplit | None = None, retriever=None, *, ingestor=None):
        self.settings, self.backend = settings, backend
        self.split = split or DatasetSplit(settings.split_root)
        self.ingestor = ingestor if ingestor is not None else Ingestor(self.split, settings.cache_dir)
        self.preparer = BusinessPreparer(settings, self.split)
        self.schema_repair = SchemaRepair(settings, self.split)
        if retriever is None and settings.rag.mode != 'none':
            raise DataMateError('RAG bật cần retriever do application tạo')
        self.retriever = retriever if retriever is not None else NullRetriever()
        self._owns_resources, self._closed = True, False

    def close(self):
        if not self._owns_resources or self._closed:
            return
        self._closed = True
        try:
            close = getattr(self.backend, 'close', None)
            if close:
                close()  # Abort generation before waiting for an encoder batch.
        finally:
            self.retriever.close()

    def new_worker(self):
        """Isolate per-question/cache controller state; share engine and ingestion."""
        worker = Agent(self.settings, self.backend, self.split, self.retriever, ingestor=self.ingestor)
        worker._owns_resources = False
        return worker

    def run(self, question: Question, remaining_seconds: float | None = None) -> RunResult:
        started = time.monotonic()
        model_path = self.settings.backend != 'demo' and getattr(self.backend,'counts_as_model',True)
        adaptive = model_path and (self.settings.adaptive_preparation or self.settings.schema_mode == 'llm')
        plan_required = model_path and self.settings.require_plan
        auto_execute = model_path and self.settings.auto_execute_plan
        repair_attempted = False
        allowance = self.settings.question_timeout_seconds + (self.settings.preparation_timeout_seconds if adaptive else 0)
        budget = min(allowance, remaining_seconds) if remaining_seconds is not None else allowance
        deadline = started + budget
        trace, calls, reviewed = [], 0, False
        prompt_snapshot = []
        results, evidence, repeats = {}, {}, Counter()
        consecutive_parse_failures = 0
        last_id, plan = None, None
        answer = Answer('abstain', reason='Hết ngân sách bước hoặc thời gian')
        try:
            catalog = self.ingestor.catalog(question.business_id, normalize=not adaptive)
            if model_path and not adaptive and self.settings.schema_mode == 'code_first':
                catalog = self.schema_repair.cached_catalog(catalog,self.backend)
            if adaptive:
                try:
                    catalog, preparation_trace = self.preparer.prepare_for_business(catalog, question.business_id, self.backend,
                        min(deadline, started+self.settings.preparation_timeout_seconds))
                finally:
                    preparation_trace = getattr(self.preparer,'last_trace',[])
                    if preparation_trace:
                        preparation_trace[-1]['preparation_elapsed_seconds']=round(time.monotonic()-started,4)
                    trace.extend(preparation_trace)
                    calls += sum('raw' in event for event in preparation_trace)
                deadline = min(deadline,time.monotonic()+self.settings.question_timeout_seconds)
            tools = BusinessTools(catalog, self.split.rulebook, question.as_of_date, self.settings.sql_timeout_seconds, self.settings.max_result_rows, deadline)
            def repair_schema(proposal):
                nonlocal repair_attempted,catalog,tools,calls,results,evidence,last_id
                if not model_path or adaptive or self.settings.schema_mode!='code_first' or repair_attempted or not repair_needs(proposal,catalog.semantic):
                    return False
                repair_attempted=True
                before=catalog.fingerprint
                try:
                    catalog,repair_trace=self.schema_repair.repair(catalog,
                        self.ingestor.catalog(question.business_id,normalize=False),self.split.readme(question.business_id),
                        proposal,self.backend,deadline)
                except BackendUnavailableError:
                    raise
                except Exception as exc:
                    repair_trace=getattr(self.schema_repair,'last_trace',[]) or [{'phase':'preparation','preparation_stage':'targeted_schema_repair'}]
                    repair_trace[-1]['observation']={'error':str(exc),'baseline_preserved':True}
                trace.extend(repair_trace)
                calls+=sum(bool(item.get('model_call_attempted') or 'raw' in item) for item in repair_trace)
                changed=catalog.fingerprint!=before
                if changed:
                    tools=BusinessTools(catalog,self.split.rulebook,question.as_of_date,self.settings.sql_timeout_seconds,self.settings.max_result_rows,deadline)
                    results,evidence,last_id={},{},None
                return changed

            def execute_calculation():
                nonlocal plan,preflight
                try:
                    return calculate(plan,tools)
                except DataMateError:
                    # A global unknown label outside the requested date range
                    # must not trigger a model call. Try the guarded tool first.
                    if repair_schema(plan):
                        plan=validate_plan({k:v for k,v in plan.items() if k!='resolved_periods'},question,tools,self.split.readme(question.business_id))
                        event['catalog_changed']=True
                        periods=plan['resolved_periods']
                        period={'kind':'range',**{k:periods[0][k] for k in ('start','end')}} if periods else None
                        preflight=tools.check(question.question,period,plan=plan)
                        if preflight['availability']=='missing':
                            evidence['e0']=preflight
                        return calculate(plan,tools)
                    raise
            contract = question_contract(question.question)
            try:
                expected_periods = question_periods(question.question, question.as_of_date, tools.policy)
            except (DataMateError, ValueError):
                expected_periods = []  # Lexical hints cannot prevent planning.
            first_period = {'kind': 'range', **{k: expected_periods[0][k] for k in ('start', 'end')}} if expected_periods else None
            preflight = tools.check(question.question, first_period)
            if preflight['availability'] == 'missing' and not adaptive:
                evidence['e0'] = preflight
                preflight = {**preflight, 'evidence_id': 'e0'}
            request_contract = {'contract': contract, 'periods': expected_periods, 'answerability': preflight,
                                         'plan_required':plan_required, 'auto_execute_plan':auto_execute,
                                         'result_review_required':self.settings.mode=='self_review',
                                         'schema_mode':'llm' if adaptive else self.settings.schema_mode,
                                         'lexical_hints_are_advisory':True}
            prompt_args = (question, self.split.rulebook, self.split.readme(question.business_id), catalog.summary(), request_contract)
            prefix = self.settings.backend == 'vllm' and self.settings.vllm.cache_prompt_prefix
            messages = initial_messages(*prompt_args, cache_prefix=prefix)
            baseline_messages_sha256 = hashlib.sha256(json_text(messages,sort_keys=True).encode()).hexdigest()
            selected = []
            if self.settings.rag.mode != 'none':
                chunks, retrieval = self.retriever.retrieve(question.question, question.business_id,
                    self.ingestor.catalog(question.business_id, normalize=False),
                    min(deadline-time.monotonic(), self.settings.rag.timeout_seconds))
                messages, selected, packed = fit_context(chunks,
                    build_messages=lambda context: initial_messages(*prompt_args, cache_prefix=prefix, retrieved_context=context),
                    backend=self.backend, settings=self.settings)
                trace.append({'phase':'retrieval', **retrieval, **packed, 'selected': [c['source'] for c in selected],
                              'baseline_messages_sha256':baseline_messages_sha256})
            prompt_snapshot=deepcopy(messages)
            for step in range(self.settings.max_steps):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if selected:
                    history = messages[2:]
                    messages, selected, packed = fit_context(selected,
                        build_messages=lambda context: initial_messages(*prompt_args, cache_prefix=prefix, retrieved_context=context) + history,
                        backend=self.backend, settings=self.settings)
                    trace.append({'phase': 'retrieval_context', 'step': step + 1, **packed})
                calls += int(getattr(self.backend, 'counts_as_model', True))
                try:
                    raw = self.backend.generate(messages, remaining)
                except Exception as exc:
                    trace.append({'step':step+1,'model_call_attempted':True,'error':str(exc),'error_kind':'generation',
                                  'generation':deepcopy(getattr(self.backend,'last_generation',None))})
                    raise
                event = {'step': step + 1, 'raw': raw}
                if step == 0:
                    event['preflight'] = preflight
                generation = getattr(self.backend, 'last_generation', None)
                if generation:
                    event['generation'] = generation
                stage, action = 'action_parse', None
                try:
                    action = parse_action(raw)
                    consecutive_parse_failures = 0
                    event['llm_action'] = deepcopy(action)
                    event['action'] = action
                    kind = action['action']
                    messages.append({'role': 'assistant', 'content': json_text(action,separators=(',',':'))})
                    if kind == 'plan':
                        stage = 'question_plan'
                        try:
                            plan = validate_plan(action,question,tools,self.split.readme(question.business_id))
                        except DataMateError:
                            if not repair_schema(action):
                                raise
                            event['catalog_changed']=True
                            plan = validate_plan(action,question,tools,self.split.readme(question.business_id))
                        # A new interpretation invalidates previous calculations,
                        # projections and missing-data evidence.
                        results, evidence, last_id = {}, {}, None
                        reviewed=False
                        expected_periods = plan['resolved_periods']
                        first_period = {'kind':'range', **{k:expected_periods[0][k] for k in ('start','end')}} if expected_periods else None
                        preflight = tools.check(question.question,first_period,plan=plan)
                        if preflight['availability'] == 'missing':
                            evidence['e0'] = preflight
                            preflight = {**preflight,'evidence_id':'e0'}
                        feedback = {'plan':plan, 'answerability':preflight,
                                    'next_action':'calculate để code giữ đầy đủ kỳ/điều kiện; hoặc metric phù hợp/SQL có kiểm tra'}
                        if auto_execute:
                            if plan.get('query'):
                                stage='sql_execution'
                                query=plan['query']
                                feedback=tools.executor.execute(query['sql'],query.get('parameters'),deadline-time.monotonic())
                                feedback['verified']=False
                                feedback['verification_scope']='executed_only; pending_plan_validation'
                            elif plan['operation']!='custom':
                                stage='calculate'
                                feedback=execute_calculation()
                            if feedback.get('availability')=='missing':
                                evidence['e0']=feedback
                                feedback={**feedback,'evidence_id':'e0'}
                            event['auto_execute_plan']=True
                    elif kind == 'prepare':
                        raise DataMateError('prepare chỉ dùng trong bước chuẩn bị doanh nghiệp; câu hỏi hiện tại dùng plan')
                    elif kind == 'final':
                        stage = 'answer_projection'
                        if plan_required and plan is None and action['type'] not in {'abstain','unanswerable'}:
                            raise DataMateError('Cần action plan trước khi chốt để xác định đại lượng/kỳ/điều kiện/type; từ khóa chỉ là gợi ý')
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
                            sql_check = None
                            if not result.get('verified') and plan:
                                sql_check = verify_sql_plan(result,plan,tools)
                                event['sql_validation'] = {k:v for k,v in sql_check.items() if k!='expected_result'}
                            elif tools.policy.sales_statuses and not result.get('verified'):
                                tools.verify_fact_sql(result,question.question)
                            answer_type = action['type']
                            expected = plan['type'] if plan else result.get('answer_type')
                            if expected and answer_type != expected:
                                if (answer_type, expected) in {('string', 'date'), ('string', 'boolean'), ('number', 'boolean')}:
                                    event['type_normalized'] = {'from': answer_type, 'to': expected}
                                    answer_type = expected
                                else:
                                    raise DataMateError('type không khớp câu hỏi/công cụ; cần ' + expected)
                            candidate = answer_from_query(answer_type, result, action.get('columns'),
                                                          percent_places=tools.policy.percent_places)
                            if sql_check and 'expected_result' in sql_check:
                                compiled = answer_from_query(answer_type,sql_check['expected_result'],percent_places=tools.policy.percent_places)
                                try:
                                    verify_legacy_answer({'answer':candidate.answer},compiled)
                                except DataMateError as exc:
                                    raise DataMateError('SQL khác kết quả phép tính biên dịch từ plan. Kiểm tra kỳ, điều kiện, đơn vị hoặc dùng calculate') from exc
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
                        feedback = tools.check(question.question, period,plan=plan)
                        if feedback['availability'] == 'missing':
                            proof_id = 'e' + str(len(evidence) + 1)
                            evidence[proof_id] = feedback
                            feedback = {**feedback, 'evidence_id': proof_id}
                    else:
                        stage = kind if kind in {'metric','calculate'} else 'sql_execution'
                        if plan_required and plan is None and kind in {'metric','calculate'}:
                            raise DataMateError('Cần action plan trước khi tính; schema/profile/lookup/SQL khám phá vẫn dùng được')
                        if kind == 'calculate':
                            if plan is None:
                                raise DataMateError('calculate cần plan đã chấp nhận')
                            feedback = execute_calculation()
                        elif kind == 'metric':
                            verify_metric_request(action, question.question, question.as_of_date, tools.policy,plan=plan)
                            feedback = tools.metric(action)
                            if feedback.get('availability') == 'missing':
                                if not plan and not contract['metrics']:
                                    raise DataMateError('Bằng chứng thiếu dữ liệu không gắn với đại lượng của câu hỏi')
                                proof_id = 'e' + str(len(evidence) + 1)
                                evidence[proof_id] = feedback
                                feedback = {**feedback, 'evidence_id': proof_id}
                        else:
                            feedback = tools.executor.execute(action['sql'], action.get('parameters'), deadline - time.monotonic())
                            feedback['verified'] = False
                            feedback['verification_scope'] = 'executed_only; query_success_does_not_prove_question_meaning'
                            if plan:
                                feedback['next_action'] = 'final với query_id/columns nếu đúng plan; SQL sẽ được đối chiếu phép tính biên dịch khi hỗ trợ'
                        if kind == 'calculate' and feedback.get('availability') == 'missing':
                            proof_id = 'e' + str(len(evidence) + 1)
                            evidence[proof_id] = feedback
                            feedback = {**feedback,'evidence_id':proof_id}
                        if kind == 'metric' and plan and 'rows' in feedback:
                            if feedback.get('answer_type') != plan['type']:
                                raise DataMateError('Type của metric khác plan; chọn đúng phép tính, không đổi kiểu của số để chốt')
                            if plan['operation'] != 'custom':
                                expected_result = calculate(plan,tools)
                                if 'rows' not in expected_result:
                                    raise DataMateError('Chưa kiểm chứng được metric theo kế hoạch')
                                actual = answer_from_query(plan['type'],feedback,percent_places=tools.policy.percent_places)
                                expected = answer_from_query(plan['type'],expected_result,percent_places=tools.policy.percent_places)
                                verify_legacy_answer({'answer':actual.answer},expected)
                        if 'rows' in feedback:
                            last_id = 'r' + str(len(results) + 1)
                            feedback = {**feedback, 'query_id': last_id}
                            results[last_id] = feedback
                    if auto_execute and plan and kind in {'plan','metric','calculate'} and 'rows' in feedback:
                        # Use the same projection/SQL checks as explicit final.
                        # No model turn is needed to copy a query_id or number.
                        # Standalone SQL may be exploration; only an inline
                        # plan.query explicitly declares SQL to be the output.
                        if 'query_id' not in feedback:
                            last_id='r'+str(len(results)+1)
                            feedback={**feedback,'query_id':last_id}
                            results[last_id]=feedback
                        event['execution']=deepcopy(feedback)
                        if evidence.get('e0',{}).get('availability')=='missing':
                            raise DataMateError('Thiếu thông tin đã được check xác nhận: '+str(evidence['e0']['missing']))
                        stage='answer_projection'
                        columns=plan.get('query',{}).get('columns') if kind=='plan' else action.get('columns')
                        candidate=self._project_result(plan,tools,feedback,columns,event)
                        event['auto_final']={'type':candidate.type,'query_id':feedback['query_id'],'columns':columns,'answer':candidate.answer}
                        if self.settings.mode=='self_review' and not reviewed:
                            reviewed=True
                            event['review_requested']=True
                            feedback={**feedback,'candidate':candidate.to_dict(),
                                'review_checklist':['đúng đại lượng và đơn vị','đủ thực thể/nhóm/điều kiện trong câu hỏi',
                                    'đúng kỳ và mốc so','đúng cấp dòng/đơn/ngày','đúng type và số kết quả/thứ tự'],
                                'next_action':'Đối chiếu câu hỏi gốc với plan và kết quả; sai thì sửa plan có đủ điều kiện, đúng thì final query_id; không chỉ chép candidate'}
                        else:
                            answer=candidate
                            event.update(observation=feedback,accepted=True)
                            trace.append(event)
                            break
                    if event.get('catalog_changed'):
                        feedback['catalog_updates']=json.loads(initial_messages(question,'','',catalog.summary())[1]['content'])['catalog']
                    feedback = {**feedback, 'remaining_steps': self.settings.max_steps - step - 1}
                    event['observation'] = feedback
                except BackendUnavailableError:
                    raise
                except (DataMateError, KeyError, TypeError, ValueError) as exc:
                    if stage == 'action_parse':
                        consecutive_parse_failures += 1
                    if not action:
                        messages.append({'role': 'assistant', 'content': '{"protocol_error":true}'})
                    if action and action['action'] in {'sql', 'metric', 'calculate'}:
                        last_id = None
                    feedback = {'error': str(exc), 'error_kind': stage, 'remaining_steps': self.settings.max_steps - step - 1,
                                'repair': self._repair(stage,action,plan),
                                **({'expected_type':plan['type']} if plan and stage=='answer_projection' else {})}
                    event['observation'] = feedback
                trace.append(event)
                if consecutive_parse_failures:
                    event['consecutive_parse_failures'] = consecutive_parse_failures
                    if consecutive_parse_failures >= self.settings.max_repeated_actions:
                        answer = Answer('abstain', reason='Lỗi định dạng action liên tiếp dù đã yêu cầu sửa; dừng retry')
                        event.update(loop_stopped=True, loop_stop_kind='consecutive_action_parse')
                        break
                repeated_action = {k:v for k,v in action.items() if k not in {'reason','from_query'}} if action else raw
                signature = hashlib.sha256(json_text({'action': repeated_action, 'error': feedback.get('error'),
                    'rows': feedback.get('rows'), 'availability': feedback.get('availability')}, sort_keys=True).encode()).hexdigest()
                repeats[signature] += 1
                if repeats[signature] >= self.settings.max_repeated_actions:
                    answer = Answer('abstain', reason='Lặp cùng action/kết quả không có tiến triển; dừng để tiết kiệm ngân sách')
                    event['loop_stopped'] = True
                    break
                event['model_observation'] = self._feedback_for_model(feedback)
                messages.append({'role': 'user', 'content': json_text({'tool_result': event['model_observation']},separators=(',',':'))})
        except (BackendUnavailableError, RetrievalFatalError):
            raise
        except Exception as exc:
            trace.append({'error': f'{type(exc).__name__}: {exc}', 'error_kind': 'runtime'})
            answer = Answer('abstain', reason=f'Lỗi xử lý: {str(exc)[:250]}')
        return RunResult(answer, time.monotonic() - started, trace, calls, prompt_snapshot)

    @staticmethod
    def _project_result(plan,tools,result,columns,event):
        check=None
        if not result.get('verified'):
            check=verify_sql_plan(result,plan,tools)
            event['sql_validation']={k:v for k,v in check.items() if k!='expected_result'}
        if result.get('answer_type') and result['answer_type']!=plan['type']:
            raise DataMateError('Type công cụ không khớp plan')
        candidate=answer_from_query(plan['type'],result,columns,percent_places=tools.policy.percent_places)
        if check and 'expected_result' in check:
            compiled=answer_from_query(plan['type'],check['expected_result'],percent_places=tools.policy.percent_places)
            try:
                verify_legacy_answer({'answer':candidate.answer},compiled)
            except DataMateError as exc:
                raise DataMateError('SQL khác kết quả phép tính biên dịch từ plan; dùng calculate hoặc sửa SQL/kế hoạch') from exc
        return candidate

    @staticmethod
    def _feedback_for_model(feedback):
        # Full SQL provenance remains in logs. Repeating the SQLite authorizer's
        # column list and the complete plan costs context without new evidence.
        compact={k:v for k,v in feedback.items() if k not in {'references','functions','direct_tables','source_bindings','read_contexts','read_tables'}}
        if 'plan' in compact:
            compact['plan']={k:v for k,v in compact['plan'].items() if k in {'measure','operation','type','population','source','resolved_periods','date','filters','numerator_filters','series_filters','entities','unit','group_by','limit','threshold','flow'}}
        if 'answerability' in compact:
            compact['answerability']={k:v for k,v in compact['answerability'].items() if k!='source_bindings'}
        return compact

    @staticmethod
    def _repair(stage,action,plan):
        if stage == 'action_parse':
            return 'Trả đúng một JSON action ngắn, đóng đủ dấu ngoặc; không thêm reasoning hoặc gộp nhiều action.'
        if stage == 'question_plan':
            return ('Sửa đúng trường báo lỗi; mỗi enum chọn một giá trị. periods là list; rank dùng operation="rank", '
                    'group_by/display là tên cột chuỗi, limit là số nguyên, không lồng object rank. '
                    'Plan và từng filter cần evidence_id="Q" hoặc evidence trích nguyên văn. '
                    'Không bỏ điều kiện để qua kiểm tra; chưa đủ căn cứ thì abstain.')
        if stage == 'answer_projection':
            return 'Giữ query_id thành công. Dùng đúng type và columns để chiếu cột cần lấy; nhiều kết quả theo câu hỏi dùng list có thứ tự. SQL khác plan thì dùng calculate hoặc sửa kế hoạch có căn cứ.'
        if stage in {'metric','calculate'}:
            return 'Kiểm tra đại lượng/kỳ và tham số. Nếu metric không giữ đủ điều kiện thì dùng calculate theo plan; nghiệp vụ chưa biểu diễn được dùng SQL trên nguồn đã kiểm tra, chưa chắc thì abstain.'
        if stage == 'sql_execution':
            return 'Đọc lỗi SQL, xem schema/profile cột liên quan rồi sửa truy vấn; không lặp nguyên SQL lỗi. Không đổi sang đại lượng khác.'
        return 'Sửa đúng lỗi được báo bằng một action; nguồn chưa ánh xạ là chưa hỗ trợ, không tự kết luận bằng 0 hoặc unanswerable.'
