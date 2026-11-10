import math

import pytest

from datamate.domain import DataMateError


def test_dispatch_reports_selected_external_function_and_reference_fallback():
    pytest.importorskip('transformers')
    from transformers.integrations.hub_kernels import use_kernel_func_from_hub_with_fallback
    from datamate.training.kernels import describe_dispatch
    @use_kernel_func_from_hub_with_fallback('sin', 'math')
    def external(x): return x
    @use_kernel_func_from_hub_with_fallback('missing_function', 'nonexistent_datamate_kernel_fixture')
    def reference(x): return x
    assert external(0.5) == math.sin(0.5)
    fast = describe_dispatch(external)
    slow = describe_dispatch(reference)
    assert fast['external'] and fast['implementation'] == 'math.sin'
    assert not slow['external']


def test_kernel_requirement_rejects_silent_reference_fallback():
    pytest.importorskip('transformers')
    from datamate.training.kernels import require_optimized_dispatch
    from transformers.models.qwen3_5 import modeling_qwen3_5
    # CPU test environment intentionally has neither CUDA package installed.
    with pytest.raises(DataMateError, match='kernel'):
        require_optimized_dispatch(modeling_qwen3_5)


def test_fla_conv_adapter_preserves_layout_activation_and_gradients():
    torch = pytest.importorskip('torch')
    from datamate.training.kernels import make_fla_conv_adapter
    def backend(x, weight, bias=None, activation=None, backend=None, **kwargs):
        assert backend == 'triton'
        out = torch.nn.functional.conv1d(x.transpose(1,2),weight[:,None,:],bias,
                                        padding=weight.shape[-1]-1,groups=x.shape[-1])[:,:,:x.shape[1]]
        if activation == 'silu': out = torch.nn.functional.silu(out)
        return out.transpose(1,2), None
    torch.manual_seed(42)
    x = torch.randn(2,4,7,requires_grad=True)
    w = torch.randn(4,3,requires_grad=True)
    actual = make_fla_conv_adapter(backend)(x,w,activation='silu')
    expected = torch.nn.functional.silu(torch.nn.functional.conv1d(x,w[:,None,:],padding=2,groups=4)[:,:,:7])
    torch.testing.assert_close(actual,expected)
    actual.sum().backward()
    gx, gw = x.grad.clone(),w.grad.clone()
    x.grad=w.grad=None
    expected.sum().backward()
    torch.testing.assert_close(x.grad,gx)
    torch.testing.assert_close(w.grad,gw)


def test_kernel_gradient_check_rejects_zero_backward_even_for_tiny_values():
    torch=pytest.importorskip('torch')
    from datamate.training.kernels import gradient_relative_errors
    expected=[torch.tensor([1e-8,-2e-8])]
    with pytest.raises(DataMateError,match='gradient'):
        gradient_relative_errors([torch.zeros(2)],expected)
    errors=gradient_relative_errors([expected[0]*1.01],expected)
    assert errors[0]<0.02
