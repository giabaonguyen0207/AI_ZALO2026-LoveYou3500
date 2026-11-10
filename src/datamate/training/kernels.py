"""Fail-fast checks for Qwen's actual training-kernel dispatch; no model weights."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import time

from ..domain import DataMateError


def describe_dispatch(function):
    """Inspect HF's chosen implementation, rather than assuming import == active."""
    seen = set()
    while function is not None and id(function) not in seen:
        seen.add(id(function))
        if hasattr(function, '_datamate_kernel_dispatch'):
            return function._datamate_kernel_dispatch
        if inspect.isfunction(function):
            closure = inspect.getclosurevars(function).nonlocals
            if 'implementation' in closure and 'is_new_implementation' in closure:
                implementation = closure['implementation']
                return {'external':bool(closure['is_new_implementation']),
                        'implementation':implementation.__module__+'.'+implementation.__qualname__}
        function = getattr(function, '__wrapped__', None)
    raise DataMateError('Không xác minh được kernel dispatch của phiên bản Transformers này')


def make_fla_conv_adapter(implementation):
    """Map HF's [B,D,T] uncached training conv onto FLA's [B,T,D] Triton op."""
    def convolution(x, weight, bias=None, activation=None, **kwargs):
        if kwargs.get('seq_idx') is not None or kwargs.get('cu_seqlens') is not None:
            raise DataMateError('Speed profile chưa dùng packed conv sequences')
        out, _ = implementation(x=x.transpose(1,2), weight=weight.to(dtype=x.dtype),
                                bias=bias.to(dtype=x.dtype) if bias is not None else None,
                                activation=activation, output_final_state=False, backend='triton')
        return out.transpose(1,2)
    convolution._datamate_kernel_dispatch = {
        'external':True,'implementation':implementation.__module__+'.'+implementation.__qualname__,
        'backend':'triton','layout_adapter':'B,D,T <-> B,T,D'}
    return convolution


def configure_training_kernels():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    delta = describe_dispatch(module.torch_chunk_gated_delta_rule)
    if not delta['external']:
        raise DataMateError('Thiếu kernel FLA DeltaNet; cài flash-linear-attention[cuda]==0.6.0 trong runtime mới')
    if not describe_dispatch(module.causal_conv1d_fn)['external']:
        from fla.modules.causal_conv1d import causal_conv1d
        module.causal_conv1d_fn = make_fla_conv_adapter(causal_conv1d)
    return require_optimized_dispatch(module)


def require_optimized_dispatch(module):
    result = {'delta_rule':describe_dispatch(module.torch_chunk_gated_delta_rule),
              'causal_conv1d':describe_dispatch(module.causal_conv1d_fn)}
    if not all(value['external'] for value in result.values()):
        raise DataMateError('Thiếu kernel training tối ưu: '+json.dumps(result)+
                           '. Chạy ô cài FLA/causal-conv1d trong runtime mới; không train full bằng fallback.')
    return result


def gradient_relative_errors(actual, expected, tolerance=0.05):
    """A relative L2 bound cannot accept zero gradients just because they are tiny."""
    import torch
    if len(actual) != len(expected): raise DataMateError('Kernel gradient count mismatch')
    errors = []
    for index, (value, reference) in enumerate(zip(actual, expected)):
        if (value is None or reference is None or value.shape != reference.shape
                or not torch.isfinite(value).all() or not torch.isfinite(reference).all()):
            raise DataMateError(f'Kernel gradient {index} thiếu/sai shape hoặc NaN/Inf')
        norm = torch.linalg.vector_norm(reference.double())
        difference = torch.linalg.vector_norm(value.double()-reference.double())
        error = float(difference/norm) if norm > 0 else (0.0 if difference == 0 else float('inf'))
        if error > tolerance:
            raise DataMateError(f'Kernel gradient {index} relative L2 error {error:.6g} > {tolerance}')
        errors.append(error)
    return errors


def check_kernels():
    import importlib.metadata as metadata
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    if not torch.cuda.is_available(): raise DataMateError('Kernel probe cần CUDA GPU')
    # Keep native PyTorch reference functions before installing the explicit conv adapter.
    originals = {'delta':module.torch_chunk_gated_delta_rule,'conv':module.causal_conv1d_fn}
    dispatch = configure_training_kernels()
    if not torch.cuda.is_bf16_supported(): raise DataMateError('Speed profile cần BF16 GPU')
    torch.manual_seed(42)

    def reference(function):
        seen = set()
        while id(function) not in seen:
            seen.add(id(function))
            if inspect.isfunction(function):
                closure = inspect.getclosurevars(function).nonlocals
                if 'torch_function' in closure: return closure['torch_function']
            function = getattr(function, '__wrapped__', None)
            if function is None: break
        raise DataMateError('Không tìm thấy reference để kiểm kernel')

    def probe(fast, slow, inputs, kwargs, tuple_output=False):
        outputs, gradients = [], []
        times = []
        for function in (slow, fast):
            values = [value.detach().clone().requires_grad_(True) for value in inputs]
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = function(*values, **kwargs)
            if tuple_output: out = out[0]
            out.float().square().mean().backward()
            torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
            outputs.append(out.detach())
            gradients.append([value.grad.detach().clone() if value.grad is not None else None for value in values])
        torch.testing.assert_close(outputs[1],outputs[0],rtol=0.03,atol=0.003)
        errors = gradient_relative_errors(gradients[1],gradients[0])
        return {'forward_backward_verified':True,'rtol_output':0.03,'atol_output':0.003,
                'gradient_relative_l2_errors':errors,'gradient_relative_l2_tolerance':0.05,
                'max_abs_output_difference':float((outputs[1]-outputs[0]).abs().max()),
                'reference_seconds':times[0],'kernel_seconds_including_first_compile':times[1]}

    dtype = torch.bfloat16
    q = torch.randn(1,128,32,128,device='cuda',dtype=dtype)
    k, v = torch.randn_like(q), torch.randn_like(q)
    g = -torch.rand(1,128,32,device='cuda',dtype=torch.float32)
    beta = torch.rand(1,128,32,device='cuda',dtype=dtype)
    delta = probe(module.torch_chunk_gated_delta_rule,reference(originals['delta']),
                  [q,k,v,g,beta], {'output_final_state':False,'use_qk_l2norm_in_kernel':True}, True)
    x = torch.randn(1,256,128,device='cuda',dtype=dtype)
    w = torch.randn(256,4,device='cuda',dtype=dtype)*0.1
    conv = probe(module.causal_conv1d_fn,reference(originals['conv']),[x,w], {'activation':'silu'})
    return {'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
            'versions':{name:metadata.version(name) for name in
                        ('transformers','peft','accelerate','triton','flash-linear-attention','fla-core')},
            'dispatch':dispatch,'delta_rule':delta,'causal_conv1d':conv,
            'full_model_speed_measured':False}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = check_kernels()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
