"""Bound conversation history without truncating the question or split rules."""
import json

from ..domain import json_text
from ..models.chat_format import context_budget


def fit_conversation(messages, backend, settings, *, plan, results, evidence, compact_result):
    counter = getattr(backend, 'count_tokens', None)
    if not callable(counter) or not getattr(backend, 'counts_as_model', True):
        return messages, None
    tokens = counter(messages)
    limit = settings.max_context_tokens - settings.max_new_tokens
    if tokens <= limit:
        return messages, None
    # The complete initial question/rulebook/README/catalog remains intact. Keep
    # executable state, so old query/evidence IDs still refer to the same values.
    if len(messages) > 4:
        state = {'history_compacted': True, 'current_plan': plan,
                 'query_results': {key: compact_result(value) for key, value in results.items()},
                 'missing_data_evidence': {key: compact_result(value) for key, value in evidence.items()},
                 'instruction': 'Giữ query_id/evidence_id và kế hoạch hiện tại. Cần chi tiết schema/profile cũ thì gọi tool lại; không suy ra từ phần lịch sử đã bỏ.'}
        for message in messages[2:]:
            if message.get('role') != 'user':
                continue
            try:
                payload = json.loads(message['content'])
            except (ValueError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            observation = payload.get('tool_result', payload)
            if isinstance(observation, dict) and 'catalog_updates' in observation:
                # A schema repair can change the catalog after the initial
                # prompt. Keep that update even when its exchange is omitted.
                state['catalog_updates'] = observation['catalog_updates']
        # Retain the most recent complete action/tool exchange. State is a user
        # message, paired with an assistant acknowledgement for chat templates.
        packed = messages[:2] + [
            {'role': 'assistant', 'content': '{"history_checkpoint":true}'},
            {'role': 'user', 'content': json_text(state, separators=(',', ':'))},
        ] + messages[-2:]
        packed_tokens = counter(packed)
        if packed_tokens < tokens:
            context_budget(packed_tokens, settings.max_new_tokens, settings.max_context_tokens)
            return packed, {'phase': 'context_compaction', 'input_tokens_before': tokens,
                            'input_tokens': packed_tokens, 'omitted_messages': len(messages)-4,
                            'preserved_query_ids': list(results), 'preserved_evidence_ids': list(evidence)}
    # Report an oversized baseline before a counted generation attempt. Never
    # silently cut input or shrink the reserved completion, which risks bad SQL.
    context_budget(tokens, settings.max_new_tokens, settings.max_context_tokens)
    return messages, None
