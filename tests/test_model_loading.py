import pytest

from datamate.domain import DataMateError
from datamate.models.backends import load_target
from datamate.models.registry import get_model


@pytest.mark.parametrize("message", [
    "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM.",
    "CUDA out of memory. Tried to allocate 128 MiB.",
])
def test_gpu_capacity_errors_suggest_remote_gpu(message):
    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError(message)
    with pytest.raises(DataMateError, match="Colab/Kaggle") as caught:
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


def test_retired_4b_is_not_available_for_new_runs():
    with pytest.raises(DataMateError, match="registry"):
        get_model("qwen35-4b")
