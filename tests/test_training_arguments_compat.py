"""Exercise the trainer's constructor call against the installed HF API on CPU."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from datamate.training import finetune


def test_actual_training_arguments_call_accepts_api_and_preserves_three_percent_warmup(tmp_path):
    transformers = pytest.importorskip('transformers')
    # Isolate the actual constructor expression: avoid loading 9B weights/CUDA.
    tree = ast.parse(inspect.getsource(finetune.train))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'TrainingArguments']
    assert len(calls) == 1
    signature = inspect.signature(transformers.TrainingArguments)
    bound = eval(compile(ast.Expression(calls[0]), '<actual training arguments>', 'eval'), {
        'TrainingArguments': lambda **kwargs: signature.bind(**kwargs),
        'raw': {}, 'output': tmp_path, 'validation_rows': [], 'dtype': 'float32',
        'torch': SimpleNamespace(bfloat16='bf16', float16='fp16'),
    })
    # Verify scheduler semantics, not just that the new keyword is accepted.
    arguments = object.__new__(transformers.TrainingArguments)
    for name, value in bound.arguments.items():
        setattr(arguments, name, value)
    assert arguments.get_warmup_steps(1000) == 30
    assert arguments.get_warmup_steps(5) == 1
