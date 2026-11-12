"""Stable pre-load identity for caches; do not import GPU libraries here."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..domain import json_text
from .contracts import CONTRACT_VERSION


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_identity(settings):
    root = Path(__file__).parent
    modules = ('contracts.py', 'chat_format.py', 'runtime.py', 'backends.py', 'vllm_backend.py')
    return {
        'backend': settings.backend, 'contract_version': CONTRACT_VERSION,
        'generation': {k: settings.to_dict()[k] for k in (
            'model_key', 'assistant_key', 'mode', 'quantization', 'seed',
            'max_context_tokens', 'max_new_tokens', 'question_workers', 'vllm')},
        'packages': {name: package_version(name) for name in (
            'torch', 'transformers', 'vllm', 'tokenizers', 'bitsandbytes')},
        'sources': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in modules if (root / name).exists()},
    }


def template_hash(tokenizer):
    return hashlib.sha256(json_text(getattr(tokenizer, 'chat_template', None), sort_keys=True).encode()).hexdigest()
