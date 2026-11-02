from pathlib import Path

import pytest

from datamate.config import Settings
from datamate.domain import DataMateError
from datamate.models.backends import load_target
from datamate.models.registry import get_model, inventory


@pytest.mark.parametrize("message", [
    "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM.",
    "CUDA out of memory. Tried to allocate 128 MiB.",
])
def test_gpu_capacity_errors_suggest_small_local_preset(message):
    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError(message)
    with pytest.raises(DataMateError, match="Qwen 3.5 4B") as caught:
        load_target(Loader, "unused", device_map="auto")
    assert isinstance(caught.value.__cause__, ValueError)


def test_unknown_model_load_error_is_preserved():
    error = ValueError("config has an unsupported model_type")
    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise error
    with pytest.raises(ValueError) as caught:
        load_target(Loader, "unused")
    assert caught.value is error


def test_4b_local_preset_uses_pinned_full_checkpoint_inventory():
    root = Path(__file__).resolve().parents[1]
    settings = Settings.load(root / "configs/qwen35-4b-local-4bit.json")
    assert settings.quantization == "4bit"
    assert settings.max_context_tokens >= 11527 + settings.max_new_tokens
    spec = get_model(settings.model_key)
    assert spec.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert inventory(settings.model_key)["total_parameters"] == 4_659_865_088
