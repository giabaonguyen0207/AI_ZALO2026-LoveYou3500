import csv
import json
import sqlite3
from pathlib import Path

import pytest
from openpyxl import Workbook


@pytest.fixture
def split_root(tmp_path):
    root = tmp_path / "fixture_train"
    for business in ("shop_a", "shop_b", "shop_c"):
        folder = root / "data" / business
        folder.mkdir(parents=True)
        (folder / "README.md").write_text("Dữ liệu giả để kiểm thử. Tiền dùng dấu phẩy phân cách nghìn.", encoding="utf-8")
        with (folder / "sales.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Đơn", "Ngày", "Trạng thái", "Tiền", "Số lượng"])
            writer.writerows([["o1", "1 thg 6 2026", "Xong", "1,200", "2"],
                              ["o1", "1 thg 6 2026", "", "1,200", "2"],
                              ["o2", "02/06/2026", "Hủy", "2,500", "1"]])
    (root / "rulebook.md").write_text("Tuần bắt đầu thứ Hai. Không dùng ngày hệ thống.", encoding="utf-8")
    (root / "calendar.csv").write_text("date,lunar_day,lunar_month,lunar_year\n2026-06-01,16,4,2026\n", encoding="utf-8")
    questions = [{"id": f"question_{i}", "business_id": business, "question": "Tổng tiền là bao nhiêu?", "as_of_date": "2026-06-02"}
                 for i, business in enumerate(("shop_a", "shop_b", "shop_c"))]
    (root / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False)+"\n" for q in questions), encoding="utf-8")
    (root / "answers.jsonl").write_text("".join(json.dumps({"id": q["id"], "type": "number", "answer": 4900})+"\n" for q in questions), encoding="utf-8")
    return root


class ScriptedBackend:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def generate(self, messages, remaining_seconds):
        self.messages.append([dict(message) for message in messages])
        return json.dumps(next(self.actions), ensure_ascii=False)


@pytest.fixture
def scripted_backend():
    return ScriptedBackend
