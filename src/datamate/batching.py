"""Bounded question concurrency and one caller-owned output writer."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading
import time


def run_questions(agent, questions, workers, deadline, state):
    if questions and time.monotonic() >= deadline:
        state['time_budget_exhausted'] = True
        return
    if workers == 1:
        for question in questions:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                state['time_budget_exhausted'] = True
                return
            yield question, agent.run(question, remaining)
        return

    local = threading.local()

    def run(question):
        if not hasattr(local, 'agent'):
            local.agent = agent.new_worker()
        try:
            return local.agent.run(question, max(0, deadline-time.monotonic()))
        except Exception as exc:
            exc.question_id = question.id
            raise

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='datamate-question')
    iterator, pending = iter(questions), {}

    def refill():
        while len(pending) < workers and time.monotonic() < deadline:
            question = next(iterator, None)
            if question is None:
                break
            pending[pool.submit(run, question)] = question

    try:
        refill()
        while pending:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                state['time_budget_exhausted'] = True
                break
            done, _ = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            if not done:
                state['time_budget_exhausted'] = True
                break
            errors = []
            # Persist every successful completion observed in this group before
            # propagating a fatal engine error from a different worker.
            for future in done:
                question = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    errors.append(exc)
                else:
                    if time.monotonic() <= deadline:
                        yield question, result
                    else:
                        state['time_budget_exhausted'] = True
            if errors:
                raise errors[0]
            refill()
    finally:
        for future in pending:
            future.cancel()
        try:
            # Abort outstanding generation BEFORE joining Agent threads.
            agent.close()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

