"""Per-question execution receipts, captured at real function/SQL boundaries.

ContextVar isolates concurrent Agent workers. Outside a question this is a no-op.
Receipts describe executed code and actual inputs/results, including nested
queries that a higher-level tool's public response does not expose.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import inspect
import math
from pathlib import Path
import time

_receipts = ContextVar('datamate_execution_receipts', default=None)


def _snapshot(value):
    if value is None or isinstance(value,(str,bool,int)):
        return value
    if isinstance(value,float):
        return value if math.isfinite(value) else {'python_type':'float','repr':repr(value)}
    if isinstance(value,Path):
        return str(value)
    if isinstance(value,dict):
        return {str(k):_snapshot(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_snapshot(v) for v in value]
    if isinstance(value,set):
        return [_snapshot(v) for v in sorted(value,key=repr)]
    executor = getattr(value,'executor',value)
    if hasattr(executor,'database'):
        context = {'python_type':type(value).__module__+'.'+type(value).__qualname__,
                   'database':str(executor.database)}
        for field in ('as_of_date','rulebook'):
            if hasattr(value,field):
                context[field]=_snapshot(getattr(value,field))
        catalog=getattr(value,'catalog',None)
        if catalog is not None:
            context['catalog_fingerprint']=catalog.fingerprint
        return context
    return {'python_type':type(value).__module__+'.'+type(value).__qualname__, 'repr':repr(value)}


def audit_execution(function):
    signature=inspect.signature(function)
    source='src/'+function.__module__.replace('.','/')+'.py::'+function.__qualname__
    @wraps(function)
    def measured(*args,**kwargs):
        sink=_receipts.get()
        if sink is None:
            return function(*args,**kwargs)
        bound=signature.bind(*args,**kwargs)
        bound.apply_defaults()
        receipt={'sequence':len(sink)+1,'source':source,
                 'inputs':_snapshot(dict(bound.arguments))}
        sink.append(receipt)  # Call-entry order, independent of nested completion.
        started=time.monotonic()
        try:
            result=function(*args,**kwargs)
            receipt['output']=_snapshot(result)
            return result
        except BaseException as exc:
            receipt['error']={'type':type(exc).__name__,'message':str(exc)}
            raise
        finally:
            receipt['elapsed_ms']=max(0,round((time.monotonic()-started)*1000))
    return measured


def audit_methods(cls):
    for name,member in list(vars(cls).items()):
        if inspect.isfunction(member) and not name.startswith('__'):
            setattr(cls,name,audit_execution(member))
    return cls


def record_executions(function):
    @wraps(function)
    def recorded(*args,**kwargs):
        sink=[]
        token=_receipts.set(sink)
        try:
            result=function(*args,**kwargs)
            result.executions=sink
            return result
        finally:
            _receipts.reset(token)
    return recorded
