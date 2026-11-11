"""One chat template and context check for generation and token counting."""
from ..domain import DataMateError

FORMAT_VERSION = 1


def encode_messages(tokenizer, messages, *, return_tensors=None):
    options = dict(tokenize=True, add_generation_prompt=True, return_dict=True, enable_thinking=False)
    if return_tensors is not None:
        options['return_tensors'] = return_tensors
    return tokenizer.apply_chat_template(messages, **options)


def input_ids(encoded):
    ids = encoded['input_ids']
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise DataMateError('Token counter chỉ nhận một conversation')
        ids = ids[0]
    return list(ids)


def context_budget(input_tokens, output_tokens, context_tokens):
    if any(type(x) is not int for x in (input_tokens, output_tokens, context_tokens)):
        raise DataMateError('Ngân sách context/token phải là số nguyên')
    if input_tokens < 0 or output_tokens < 1 or context_tokens < 1:
        raise DataMateError('Ngân sách context/token không hợp lệ')
    total = input_tokens + output_tokens
    if total > context_tokens:
        raise DataMateError(
            f'Context vượt giới hạn: input={input_tokens}, output_reserved={output_tokens}, '
            f'limit={context_tokens}, vượt={total-context_tokens}. Không tự cắt rulebook/câu hỏi.')
    return {'input_tokens': input_tokens, 'output_reserved': output_tokens,
            'context_tokens': context_tokens, 'remaining_tokens': context_tokens-total}


def generation_token_settings(tokenizer, generation_config, *, vocab_size=None):
    """Resolve stop/pad IDs from the loaded checkpoint without changing its files.

    Some chat checkpoints only configure end-of-document as generation EOS.
    Keep those IDs and add the tokenizer's EOS and atomic chat turn terminator.
    No numeric token ID or model key is assumed here.
    """
    size = min(len(tokenizer), vocab_size) if vocab_size is not None else len(tokenizer)

    def checked(value, label):
        if type(value) is not int or not 0 <= value < size:
            raise DataMateError(f'{label} không phải token ID hợp lệ trong vocabulary: {value!r}')
        return value

    configured = getattr(generation_config, 'eos_token_id', None)
    checkpoint_eos = list(configured) if isinstance(configured, (list, tuple)) else ([] if configured is None else [configured])
    eos = [checked(value, 'generation.eos_token_id') for value in checkpoint_eos]
    tokenizer_eos = getattr(tokenizer, 'eos_token_id', None)
    if tokenizer_eos is not None:
        eos.append(checked(tokenizer_eos, 'tokenizer.eos_token_id'))
    template = getattr(tokenizer, 'chat_template', None)
    if isinstance(template, dict):
        template = '\n'.join(template.values())
    turn_tokens = {}
    for marker in ('<|im_end|>', '<turn|>'):
        if not isinstance(template, str) or marker not in template:
            continue
        value = tokenizer.convert_tokens_to_ids(marker)
        if (value is None or value not in tokenizer.all_special_ids
                or tokenizer.convert_ids_to_tokens(value) != marker
                or tokenizer.encode(marker, add_special_tokens=False) != [value]):
            raise DataMateError(f'Chat template dùng {marker} nhưng tokenizer không có special token đơn tương ứng')
        turn_tokens[marker] = checked(value, 'chat turn EOS')
        eos.append(value)
    eos = list(dict.fromkeys(eos))
    if not eos:
        raise DataMateError('Checkpoint/tokenizer thiếu EOS; không chạy chat generation không có ranh giới dừng')
    pad = getattr(generation_config, 'pad_token_id', None)
    tokenizer_pad = getattr(tokenizer, 'pad_token_id', None)
    if pad is None:
        pad = tokenizer_pad if tokenizer_pad is not None else eos[0]
    pad = checked(pad, 'pad_token_id')
    return {'checkpoint_eos_token_ids': checkpoint_eos, 'tokenizer_eos_token_id': tokenizer_eos,
            'tokenizer_pad_token_id': tokenizer_pad, 'chat_turn_token_ids': turn_tokens,
            'effective_eos_token_ids': eos, 'effective_pad_token_id': pad}

