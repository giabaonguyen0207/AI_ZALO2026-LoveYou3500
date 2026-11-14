"""Fit optional evidence using the generator's complete chat tokenization."""


def fit_context(chunks, *, build_messages, backend, settings):
    baseline = build_messages([])
    if not chunks:
        return baseline, [], {'context_tokens': 0, 'selected_chunks': 0}
    baseline_tokens = backend.count_tokens(baseline)
    selected = list(chunks)
    while selected:
        messages = build_messages(selected)
        tokens = backend.count_tokens(messages)
        added = max(0, tokens - baseline_tokens)
        if (tokens + settings.max_new_tokens <= settings.max_context_tokens
                and added <= settings.rag.context_tokens):
            return messages, selected, {'context_tokens': added, 'input_tokens': tokens,
                                        'selected_chunks': len(selected)}
        selected.pop()  # Keep complete, highest-ranked chunks.
    return baseline, [], {'context_tokens': 0, 'input_tokens': baseline_tokens, 'selected_chunks': 0}
