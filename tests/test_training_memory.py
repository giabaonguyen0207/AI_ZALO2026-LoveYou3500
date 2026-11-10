import pytest

torch = pytest.importorskip('torch')


def test_only_frozen_io_weights_return_to_bfloat16():
    from datamate.training.memory import restore_frozen_io_dtype
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)
            self.head = torch.nn.Linear(8, 32, bias=False)
            self.norm = torch.nn.LayerNorm(8)
            self.requires_grad_(False)
            self.adapter = torch.nn.Parameter(torch.ones(8))
        def get_input_embeddings(self): return self.embedding
        def get_output_embeddings(self): return self.head
    model = Model()
    original = model.embedding.weight.detach().clone().to(torch.bfloat16)
    report = restore_frozen_io_dtype(model, torch.bfloat16)
    assert model.embedding.weight.dtype == model.head.weight.dtype == torch.bfloat16
    assert model.norm.weight.dtype == model.adapter.dtype == torch.float32
    assert torch.equal(model.embedding.weight, original)
    assert model.adapter.requires_grad and not model.embedding.weight.requires_grad
    assert sum(r['bytes_saved'] for r in report) == 1024


@pytest.mark.parametrize('features', [
    [{'input_ids':[1,2,3,4,5], 'attention_mask':[1]*5, 'labels':[-100,-100,-100,4,5]}],
    [{'input_ids':[1,2,3,4,5], 'attention_mask':[1]*5, 'labels':[-100,-100,-100,4,5]},
     {'input_ids':[2,3,4], 'attention_mask':[1]*3, 'labels':[-100,3,4]}],
    [{'input_ids':[1,2,3], 'attention_mask':[1]*3, 'labels':[1,2,3]}],
])
def test_suffix_logits_preserve_causal_loss_and_all_gradients(features):
    from datamate.training.memory import AssistantCollator
    from transformers.loss.loss_utils import ForCausalLMLoss
    full = AssistantCollator(0, supervised_suffix_logits=False)(features)
    small = AssistantCollator(0, supervised_suffix_logits=True)(features)
    torch.manual_seed(42)
    logits = torch.randn(len(features), full['input_ids'].shape[1], 11, requires_grad=True)
    baseline = ForCausalLMLoss(logits, full['labels'], vocab_size=11)
    baseline.backward()
    expected = logits.grad.clone()
    logits.grad = None
    loss = ForCausalLMLoss(logits[:, -small['logits_to_keep']:].contiguous(), small['labels'], vocab_size=11)
    loss.backward()
    torch.testing.assert_close(loss, baseline)
    torch.testing.assert_close(logits.grad, expected)
    assert torch.equal(full['input_ids'], small['input_ids'])
    assert torch.equal(full['attention_mask'], small['attention_mask'])
    assert (full['labels'] != -100).sum() == (small['labels'] != -100).sum()
    if len(features) == 1 and features[0]['labels'][0] == -100:
        assert small['logits_to_keep'] == 3


def test_suffix_loss_matches_on_tiny_actual_qwen35_model():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from datamate.training.memory import AssistantCollator
    config = Qwen3_5Config(text_config={
        'vocab_size':31, 'hidden_size':32, 'intermediate_size':64,
        'num_hidden_layers':1, 'layer_types':['full_attention'],
        'num_attention_heads':4, 'num_key_value_heads':2, 'head_dim':8,
        'rope_parameters':{'rope_type':'default','rope_theta':10000.0,
                           'partial_rotary_factor':0.5,'mrope_section':[1,1,0]}},
        vision_config={'depth':1, 'hidden_size':16, 'intermediate_size':32,
                       'num_heads':2,'out_hidden_size':32,'num_position_embeddings':16,
                       'patch_size':2,'spatial_merge_size':1,'deepstack_visual_indexes':[]})
    torch.manual_seed(7)
    model = Qwen3_5ForConditionalGeneration(config).float()
    rows = [{'input_ids':[1,2,3,4,5], 'attention_mask':[1]*5,'labels':[-100,-100,-100,4,5]}]
    full = AssistantCollator(0)(rows)
    small = AssistantCollator(0, supervised_suffix_logits=True)(rows)
    full_loss = model(**full).loss
    full_loss.backward()
    expected = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    output = model(**small)
    assert output.logits.shape[1] == 3
    torch.testing.assert_close(output.loss, full_loss)
    output.loss.backward()
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name])
