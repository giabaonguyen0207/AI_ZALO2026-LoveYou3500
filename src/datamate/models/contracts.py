"""Engine-independent generation contract; importing it never loads weights."""
from typing import Protocol

from ..domain import DataMateError

CONTRACT_VERSION = 1


class GenerationTimeoutError(DataMateError):
    """The current request exhausted its time budget."""


class BackendUnavailableError(DataMateError):
    """Fatal engine failure. Composition/agent must propagate this in integration."""


class ChatBackend(Protocol):
    counts_as_model: bool
    supports_token_budget: bool
    last_generation: dict | None

    def generate(self, messages: list[dict], remaining_seconds: float,
                 *, max_new_tokens: int | None = None) -> str: ...
    def count_tokens(self, messages: list[dict]) -> int: ...
    def runtime_info(self) -> dict: ...
    def close(self) -> None: ...
