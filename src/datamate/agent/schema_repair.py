"""One small, on-demand schema patch; no per-business model survey by default."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import time

from ..data.interpretation import merge_binding, merge_policy
from ..data.normalization import prepare_catalog
from ..domain import DataMateError, json_text
from ..models.contracts import BackendUnavailableError
from .preparation import BusinessPreparer
from .protocol import parse_action

PATCH_PROMPT = '''Bổ sung CHỈ phần ánh xạ/quy tắc còn thiếu trong needs, không giải câu hỏi, không tính đáp án.
Một JSON ngắn: {"action":"prepare","policy":{"week_start":{"value":0,"evidence":"B3"}},"tables":{"T0":{"kind":"sales","roles":{"order_date":{"column":"T0C2","evidence":"R4"}},"status_map":{"Chốt":{"value":"completed","evidence":"R8"}}}}}.
Ví dụ minh họa, không phải giá trị mặc định. Bỏ trường không cần hoặc không có căn cứ. Không chép lại ánh xạ sẵn có.
evidence dùng đúng ID B = dòng rulebook, R = dòng README, TnCm = header. Cột dùng ID trong catalog.
Không dùng header để đoán nghĩa trạng thái mơ hồ như Hoàn. Nếu tài liệu không giải thích, để unknown.
kind: sales/orders/products/customers/payments/returns/inventory/conversion/other. Các roles dùng cùng tên như baseline.
status_map: completed/cancelled/returned/shipping/unknown; payment_map: debt/cod/cash/transfer/ewallet/marketplace/unknown.
policy chỉ dùng tên trong needs. Không SQL/Python, không tool phụ, không reasoning ngoài JSON.'''


def repair_needs(action, semantic):
    """Dependency hints from a proposed plan, never from question/business IDs."""
    measure, population = action.get('measure'), action.get('population')
    if measure not in {'revenue','quantity','orders','gross_profit','debt','inventory','price','custom'}:
        return {}
    bindings=semantic['bindings']
    relevant=[b for b in bindings if b['kind'] in {'sales','orders'}]
    roles=set().union(*(set(b['roles']) for b in relevant)) if relevant else set()
    required={'revenue':{'line_amount'},'quantity':{'quantity'},'orders':{'order_id'},
              'gross_profit':{'line_amount','quantity','cost'},'debt':{'order_id','order_total'},
              'inventory':{'stock'},'price':{'price'}}.get(measure,set())
    if measure in {'inventory','price'}:
        roles=set().union(*(set(b['roles']) for b in bindings)) if bindings else set()
    if measure=='quantity' and 'base_quantity' in roles:
        required.discard('quantity')
    if action.get('periods') and measure in {'revenue','quantity','orders','gross_profit'}:
        required |= {'order_date'}
    if population == 'sales':
        required |= {'status'}
    missing_roles=sorted(required-roles)
    policy=semantic['policy']; missing_policy=[]
    def need(name):
        if not policy.get(name):
            missing_policy.append(name)
    if population=='sales':
        need('sales_statuses')
    if measure in {'revenue','gross_profit'}:
        need('line_revenue')
    if measure=='gross_profit':
        need('gross_profit_line_cost')
    if measure=='debt':
        need('debt_supported')
    periods=action.get('periods',[])
    if isinstance(periods,list) and any(isinstance(p,dict) and p.get('kind') in {'this_week','last_week'} for p in periods):
        if policy.get('week_start') is None:
            missing_policy.append('week_start')
    filters=[*action.get('filters',[]),*action.get('numerator_filters',[])] if isinstance(action.get('filters',[]),list) and isinstance(action.get('numerator_filters',[]),list) else []
    if any(isinstance(f,dict) and f.get('op')=='shift' for f in filters):
        need('shifts')
        if 'timestamp' not in roles and 'hour' not in roles:
            missing_roles.append('timestamp')
    statuses=[name for name,profile in semantic.get('status_profiles',{}).items()
              if population in {'sales','all','debt'} and any(v.get('canonical')=='unknown' for v in profile.get('values',[]))]
    payments=[]
    if measure=='debt' or any(isinstance(f,dict) and str(f.get('column','')).endswith('payment_method') for f in filters):
        payments=[b['table'] for b in relevant if 'payment_method' in b['roles']]
    result={}
    if missing_roles: result['roles']=sorted(set(missing_roles))
    if missing_policy: result['policy']=sorted(set(missing_policy))
    if statuses: result['status_tables']=statuses
    # Inspect actual profiles before deciding to ask for payment labels.
    payments=[name for name in payments if any(v.get('canonical')=='unknown' for b in bindings if b['table']==name
              for v in semantic.get('profiles',{}).get(name,{}).get(b['roles']['payment_method'],{}).get('values',[]))]
    if payments: result['payment_tables']=payments
    return result


class SchemaRepair:
    def __init__(self,settings,split):
        self.settings,self.split=settings,split
        self.cache=BusinessPreparer(settings,split)
        self.cache.directory=Path(settings.cache_dir)/'schema-repairs'
        self.ready={}
        self.last_trace=[]

    def _identity(self,backend):
        return hashlib.sha256((self.cache._identity(backend)+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+
            json_text({'timeout':self.settings.schema_repair_timeout_seconds,'tokens':self.settings.schema_repair_max_new_tokens})).encode()).hexdigest()

    def cached_catalog(self,catalog,backend):
        return self.ready.get((catalog.fingerprint,self._identity(backend)),catalog)

    def repair(self,catalog,raw,readme,action,backend,deadline):
        self.last_trace=[]
        needs=repair_needs(action,catalog.semantic)
        if not needs:
            return catalog,[]
        started=time.monotonic()
        stop=min(deadline,started+self.settings.schema_repair_timeout_seconds)
        identity=self._identity(backend)
        bindings=catalog.semantic['bindings']
        priority=set(needs.get('status_tables',[])+needs.get('payment_tables',[]))
        missing=set(needs.get('roles',[]))
        candidates=[b for b in bindings if b['table'] in priority or missing and (b['kind'] in {'sales','orders','other'} or action.get('measure') in {'inventory','price','debt'})]
        candidates.sort(key=lambda b:(b['table'] not in priority, -len(missing & set(b.get('ambiguous',{}))),b['kind']=='other',-len(b['roles']),b['table']))
        candidates=candidates[:2]
        sources={**{'B'+str(i):line for i,line in enumerate(self.split.rulebook.splitlines()) if line.strip()},
                 **{'R'+str(i):line for i,line in enumerate(readme.splitlines()) if line.strip()}}
        columns={}; tables={}; context_tables={}
        for index,binding in enumerate(candidates):
            tid='T'+str(index)
            table=next(t for t in raw.tables if t['name']==binding['table'])
            tables[tid]=(binding,table)
            profiles=catalog.semantic['profiles'].get(table['name'],{})
            items={}
            for ci,column in enumerate(table['columns']):
                cid=tid+'C'+str(ci);columns[cid]=column;sources[cid]=column
                p=profiles.get(column,{})
                items[cid]={'name':column,**{k:p[k] for k in ('nonempty','nulls','distinct_count','min','max','parse_errors') if k in p},
                            'values':p.get('values',[])[:6], 'values_complete':bool(p.get('values_complete')) and len(p.get('values',[]))<=6}
            context_tables[tid]={'file':table['file'],'sheet':table.get('sheet'),'name':table['name'],
                                 'baseline':{'kind':binding['kind'],'roles':binding['roles'],'ambiguous':binding.get('ambiguous',{})},'columns':items}
        context={'needs':needs,'rulebook':{k:v for k,v in sources.items() if k.startswith('B')},
                 'readme':{k:v for k,v in sources.items() if k.startswith('R')},'tables':context_tables,
                 'policy':catalog.semantic['policy'],'scope':'all-row profiles; values may be incomplete'}
        # A policy-only failure is shared across businesses of the same split.
        source_key=raw.fingerprint if tables else self.split.rulebook_hash
        key=hashlib.sha256((identity+source_key+json_text(needs,sort_keys=True)+json_text(context_tables,sort_keys=True)).encode()).hexdigest()
        event={'phase':'preparation','preparation_stage':'targeted_schema_repair','needs':needs,'cache_key':key}
        self.last_trace=[event]
        cached=self.cache._load(key)
        if cached is None:
            messages=[{'role':'system','content':PATCH_PROMPT},{'role':'user','content':json_text(context)}]
            event['input_messages']=deepcopy(messages)
            try:
                if stop<=time.monotonic():
                    raise DataMateError('Hết thời gian sửa ánh xạ')
                options={'max_new_tokens':self.settings.schema_repair_max_new_tokens} if getattr(backend,'supports_token_budget',False) else {}
                event['model_call_attempted']=True
                response=backend.generate(messages,stop-time.monotonic(),**options)
                event['raw']=response
                if getattr(backend,'last_generation',None): event['generation']=deepcopy(backend.last_generation)
                proposal=parse_action(response)
                event['llm_action']=deepcopy(proposal)
                if proposal['action']!='prepare': raise DataMateError('Cần action prepare cho bản bổ sung')
                cached={'proposal':proposal}
            except BackendUnavailableError:
                raise
            except Exception as exc:
                cached={'error':str(exc)}
            self.cache._store(key,cached)
        else:
            event['cache_hit']=True
        rejected={};ready=catalog
        if 'proposal' in cached:
            proposal=cached['proposal']
            def cite(value):
                if value not in sources: raise DataMateError('ID căn cứ không có trong đầu vào: '+str(value))
                return sources[value]
            policy_patch={}
            for name,item in proposal.get('policy',{}).items() if isinstance(proposal.get('policy',{}),dict) else []:
                try:
                    if name not in needs.get('policy',[]): raise DataMateError('Chỉ bổ sung quy tắc đang thiếu')
                    if not str(item['evidence']).startswith('B'): raise DataMateError('Quy tắc phải dẫn ID rulebook B')
                    policy_patch[name]={'value':item['value'],'evidence':cite(item['evidence'])}
                except (DataMateError,KeyError,TypeError) as exc: rejected['policy.'+name]=str(exc)
            policy,errors=merge_policy(policy_patch,self.split.rulebook,catalog.semantic['policy'])
            rejected.update({'policy.'+k:v for k,v in errors.items()})
            patched=deepcopy(bindings)
            for tid,part in proposal.get('tables',{}).items() if isinstance(proposal.get('tables',{}),dict) else []:
                try:
                    if tid not in tables or not isinstance(part,dict): raise DataMateError('Bảng ngoài phạm vi sửa')
                    baseline,table=tables[tid];patch={'kind':part.get('kind',baseline['kind']),'roles':{},'evidence':{}}
                    for extra in set(part)-{'kind','roles','status_map','payment_map'}:
                        rejected[tid+'.'+extra]='Trường bổ sung chưa hỗ trợ; các trường hợp lệ khác vẫn được kiểm tra'
                    def entries(field):
                        value=part.get(field,{})
                        if not isinstance(value,dict):
                            rejected[tid+'.'+field]='Cần object'
                            return {}
                        return value
                    for role,item in entries('roles').items():
                        try:
                            cid=item['column']
                            if not cid.startswith(tid+'C') or cid not in columns: raise DataMateError('Cột không thuộc bảng')
                            patch['roles'][role]=columns[cid];patch['evidence'][role]=cite(item['evidence'])
                        except (DataMateError,KeyError,TypeError,AttributeError) as exc: rejected[tid+'.roles.'+role]=str(exc)
                    for field in ('status','payment'):
                        patch[field+'_map']={};patch[field+'_evidence']={}
                        for label,item in entries(field+'_map').items():
                            try:
                                patch[field+'_map'][label]=item['value'];patch[field+'_evidence'][label]=cite(item['evidence'])
                            except (DataMateError,KeyError,TypeError) as exc: rejected[tid+'.'+field+'.'+label]=str(exc)
                    checked,errors=merge_binding(patch,baseline,table,readme,raw,stop)
                    rejected.update({tid+'.'+k:v for k,v in errors.items()})
                    patched=[checked if b['table']==table['name'] else b for b in patched]
                except (DataMateError,KeyError,TypeError,AttributeError) as exc: rejected[tid]=str(exc)
            description={'mode':'code_first','policy':policy,'bindings':patched,'cache_key':key,
                         'validation_scope':'partial_additions_with_source_and_data_checks; semantics_not_proven'}
            if policy!=catalog.semantic['policy'] or patched!=bindings:
                ready=prepare_catalog(raw,self.split.rulebook,readme,self.settings.cache_dir,description)
                self.ready[(catalog.fingerprint,identity)]=ready
                self.ready[(ready.fingerprint,identity)]=ready
        else:
            rejected['proposal']=cached.get('error','Không có bản bổ sung')
        event['observation']={'cache_hit':bool(event.get('cache_hit')),'rejected_fields':rejected,
            'baseline_preserved':True,'changed':ready.fingerprint!=catalog.fingerprint,
            'remaining_needs':repair_needs(action,ready.semantic),'validation_scope':'provenance_and_data_consistency; semantics_not_proven'}
        event['preparation_elapsed_seconds']=round(time.monotonic()-started,4)
        return ready,[event]
