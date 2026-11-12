"""Training memory reductions that preserve complete prompts and loss targets."""
from ..domain import DataMateError


def restore_frozen_io_dtype(model, dtype):
    """Undo PEFT's FP32 upcast only for frozen input/output weights.

    Base weights were loaded in compute dtype before k-bit preparation. Keep
    norms and trainable adapter weights in FP32; never touch quantized weights.
    """
    import torch
    if dtype not in (torch.bfloat16, torch.float16):
        raise DataMateError('Frozen IO weights require a supported compute dtype')
    report, seen = [], set()
    for name in ('get_input_embeddings', 'get_output_embeddings'):
        module = getattr(model, name)()
        if module is None:
            continue
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.requires_grad or parameter.dtype != torch.float32:
                continue
            before = parameter.numel() * parameter.element_size()
            parameter.data = parameter.data.to(dtype=dtype)
            report.append({'module':name, 'parameters':parameter.numel(),
                           'dtype':str(dtype), 'bytes_saved':before-parameter.numel()*parameter.element_size()})
    return report


class AssistantCollator:
    def __init__(self, pad_token_id, *, supervised_suffix_logits=False):
        self.pad_token_id = pad_token_id
        self.supervised_suffix_logits = supervised_suffix_logits

    def __call__(self, features):
        import torch
        length = max(len(row['input_ids']) for row in features)
        batch = {}
        for key, padding in (('input_ids',self.pad_token_id), ('attention_mask',0), ('labels',-100)):
            batch[key] = torch.tensor([row[key]+[padding]*(length-len(row[key])) for row in features], dtype=torch.long)
        if self.supervised_suffix_logits:
            positions = (batch['labels'] != -100).nonzero()
            if not len(positions):
                raise DataMateError('Training batch has no supervised target')
            # Logit at t-1 predicts label at t: retain that predecessor too.
            # Input/attention tensors remain full length, including every rule.
            start = max(0, int(positions[:,1].min())-1)
            batch['logits_to_keep'] = length-start
            batch['labels'] = batch['labels'][:,start:].contiguous()
            batch['use_cache'] = False
        return batch
