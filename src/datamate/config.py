from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
import json

from .domain import DataMateError


@dataclass
class Settings:
    split_root: str = "train/train"
    cache_dir: str = ".cache/datamate"
    result_dir: str = "result"
    model_key: str = "qwen35-9b"
    model_dir: str = "models/qwen35-9b"
    backend: str = "transformers"
    mode: str = "single"
    assistant_key: str | None = None
    assistant_dir: str | None = None
    quantization: str = "none"
    max_steps: int = 10
    max_repeated_actions: int = 2
    max_new_tokens: int = 1024
    max_context_tokens: int = 16384
    question_timeout_seconds: float = 90
    sql_timeout_seconds: float = 12
    max_result_rows: int = 150
    wall_time_seconds: float = 21000  # 5h50, leaves a margin below six hours.
    # Full, model-written business descriptions are an opt-in research mode.
    # Planning is independent: turning off schema preparation must not bypass it.
    adaptive_preparation: bool = False  # Backward-compatible opt-in to full preparation.
    schema_mode: str = "code_first"  # code_first (bounded repair), code_only, llm
    require_plan: bool = True
    auto_execute_plan: bool = True
    schema_repair_timeout_seconds: float = 20
    schema_repair_max_new_tokens: int = 512
    preparation_timeout_seconds: float = 180
    preparation_max_attempts: int = 2
    preparation_max_new_tokens: int = 2048
    seed: int = 42

    def validate(self) -> Settings:
        if self.backend not in {"transformers", "demo"}:
            raise DataMateError("backend phải là transformers hoặc demo")
        if self.mode not in {"single", "self_review", "speculative"}:
            raise DataMateError("mode phải là single, self_review hoặc speculative")
        if self.quantization not in {"none", "4bit", "8bit"}:
            raise DataMateError("quantization phải là none, 4bit hoặc 8bit")
        if self.mode == "speculative" and (not self.assistant_key or not self.assistant_dir):
            raise DataMateError("Chế độ speculative cần assistant_key và assistant_dir")
        if self.mode != "speculative" and (self.assistant_key or self.assistant_dir):
            raise DataMateError("Chỉ khai báo assistant cho chế độ speculative")
        if self.max_steps < 1 or self.max_result_rows < 1 or self.max_new_tokens < 1:
            raise DataMateError("Ngân sách bước/token/dòng phải dương")
        if type(self.max_repeated_actions) is not int or self.max_repeated_actions < 2:
            raise DataMateError("max_repeated_actions tối thiểu 2")
        if self.max_context_tokens < 2048:
            raise DataMateError("max_context_tokens tối thiểu 2048")
        if self.question_timeout_seconds <= 0 or self.sql_timeout_seconds <= 0:
            raise DataMateError("Timeout phải dương")
        if type(self.adaptive_preparation) is not bool or self.preparation_timeout_seconds <= 0:
            raise DataMateError("adaptive_preparation phải là boolean; thời gian chuẩn bị phải dương")
        if self.schema_mode not in {'code_first', 'code_only', 'llm'}:
            raise DataMateError('schema_mode phải là code_first, code_only hoặc llm')
        if type(self.require_plan) is not bool or type(self.auto_execute_plan) is not bool:
            raise DataMateError('require_plan và auto_execute_plan phải là boolean')
        if self.adaptive_preparation and self.schema_mode == 'code_only':
            raise DataMateError('code_only không dùng adaptive_preparation')
        if self.schema_repair_timeout_seconds <= 0:
            raise DataMateError('schema_repair_timeout_seconds phải dương')
        if type(self.schema_repair_max_new_tokens) is not int or not 128 <= self.schema_repair_max_new_tokens < self.max_context_tokens:
            raise DataMateError('schema_repair_max_new_tokens trong [128,max_context_tokens)')
        if type(self.preparation_max_attempts) is not int or not 1 <= self.preparation_max_attempts <= 3:
            raise DataMateError("preparation_max_attempts trong 1..3")
        if type(self.preparation_max_new_tokens) is not int or not 256 <= self.preparation_max_new_tokens < self.max_context_tokens:
            raise DataMateError('preparation_max_new_tokens trong [256,max_context_tokens)')
        if not 0 < self.wall_time_seconds <= 21600:
            raise DataMateError("wall_time_seconds phải trong (0, 21600]")
        return self

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> Settings:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
        raw.update({key: value for key, value in overrides.items() if value is not None})
        try:
            return cls(**raw).validate()
        except TypeError as exc:
            raise DataMateError(f"Cấu hình không hợp lệ: {exc}") from exc

    def to_dict(self) -> dict:
        return asdict(self)


def configure_offline(cache_root: str | Path = ".cache") -> None:
    root = Path(cache_root).resolve()
    for name, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HOME": str(root / "huggingface"), "TORCH_HOME": str(root / "torch"),
        "XDG_CACHE_HOME": str(root), "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ[name] = value
