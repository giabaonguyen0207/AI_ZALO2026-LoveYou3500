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
    max_new_tokens: int = 1024
    max_context_tokens: int = 16384
    question_timeout_seconds: float = 90
    sql_timeout_seconds: float = 12
    max_result_rows: int = 150
    wall_time_seconds: float = 21000  # 5h50, leaves a margin below six hours.
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
        if self.max_context_tokens < 2048:
            raise DataMateError("max_context_tokens tối thiểu 2048")
        if self.question_timeout_seconds <= 0 or self.sql_timeout_seconds <= 0:
            raise DataMateError("Timeout phải dương")
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
