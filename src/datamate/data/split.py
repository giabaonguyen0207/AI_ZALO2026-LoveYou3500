from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..domain import DataMateError, Question


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("expected object")
                rows.append(value)
            except (ValueError, TypeError) as exc:
                raise DataMateError(f"JSONL lỗi ở {path.name}:{line_no}") from exc
    return rows


class DatasetSplit:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        for name in ("data", "rulebook.md", "calendar.csv", "questions.jsonl"):
            if not (self.root / name).exists():
                raise DataMateError(f"Split thiếu {name}: {self.root}")
        self.rulebook = (self.root / "rulebook.md").read_text(encoding="utf-8-sig")
        self.rulebook_hash = hashlib.sha256(self.rulebook.encode()).hexdigest()

    def business_path(self, business_id: str) -> Path:
        path = (self.root / "data" / business_id).resolve()
        if path.parent != (self.root / "data").resolve() or not path.is_dir():
            raise DataMateError("Doanh nghiệp không tồn tại trong split đã chọn")
        for file in path.rglob("*"):
            if not file.resolve().is_relative_to(path):
                raise DataMateError("Không đọc symbolic link trỏ ra ngoài thư mục doanh nghiệp")
        return path

    def businesses(self) -> list[str]:
        return sorted(path.name for path in (self.root / "data").iterdir() if path.is_dir())

    def questions(self) -> list[Question]:
        questions = [Question.from_dict(row) for row in read_jsonl(self.root / "questions.jsonl")]
        if len({q.id for q in questions}) != len(questions):
            raise DataMateError("Trùng question id trong split")
        return questions

    def readme(self, business_id: str) -> str:
        path = self.business_path(business_id) / "README.md"
        return path.read_text(encoding="utf-8-sig") if path.exists() else "Không có README."


def ensure_outside_dataset(path: str | Path, split_root: str | Path) -> Path:
    resolved, source = Path(path).resolve(), Path(split_root).resolve()
    if resolved == source or resolved.is_relative_to(source):
        raise DataMateError("Không được ghi cache/kết quả vào dataset")
    return resolved
