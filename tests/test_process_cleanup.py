from types import SimpleNamespace


def test_kaggle_cleanup_stops_group_even_after_parent_exit(monkeypatch):
    from tools import kaggle_workflow as workflow
    monkeypatch.setattr(workflow,'signal',SimpleNamespace(SIGTERM=15,SIGKILL=9))
    sent = []
    alive = [True]
    def killpg(pid, sig):
        if not alive[0]: raise ProcessLookupError()
        sent.append(sig)
        if sig == workflow.signal.SIGKILL: alive[0] = False
    monkeypatch.setattr(workflow,'os',SimpleNamespace(name='posix',killpg=killpg))
    workflow.stop_process(SimpleNamespace(pid=123,poll=lambda:0,wait=lambda **kwargs:0),grace_seconds=0)
    assert workflow.signal.SIGTERM in sent and workflow.signal.SIGKILL in sent
