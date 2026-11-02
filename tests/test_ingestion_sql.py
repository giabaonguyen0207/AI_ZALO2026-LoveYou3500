import hashlib
import sqlite3

import pytest
from openpyxl import Workbook

from datamate.data.ingest import Ingestor
from datamate.data.split import DatasetSplit
from datamate.data.sql import SQLExecutor
from datamate.domain import DataMateError


def source_hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def test_three_formats_header_detection_and_source_immutability(split_root, tmp_path):
    folder = split_root / "data/shop_a"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "T01_2026"
    sheet.append(["SỔ BÁN HÀNG"])
    sheet.append(["Kỳ báo cáo"])
    sheet.append([])
    sheet.append(["Mã", "Tiền", "Số"])
    sheet.append(["A", "1,500", 2])
    second = workbook.create_sheet("T02_2026")
    second.append(["Mã", "Tiền", "Số"])
    second.append(["B", "2,000", 1])
    workbook.save(folder / "months.xlsx")
    workbook.close()
    con = sqlite3.connect(folder / "pos.sqlite")
    con.execute('CREATE TABLE products ("Mã" TEXT, price REAL)')
    con.execute("INSERT INTO products VALUES ('A', 1.25)")
    con.commit()
    con.close()
    before = source_hashes(split_root)
    ingest = Ingestor(DatasetSplit(split_root), tmp_path / "cache")
    catalog = ingest.catalog("shop_a")
    tables = {t["name"]: t for t in catalog.tables}
    month = next(t for t in tables.values() if t.get("sheet") == "T01_2026")
    assert month["header_row"] == 4 and month["columns"] == ["Mã", "Tiền", "Số"]
    union = next(t for t in tables.values() if "view_of" in t)
    sql = SQLExecutor(catalog.database)
    assert sql.execute(f'SELECT SUM(num_en("Tiền")) FROM "{union["name"]}"')["rows"] == [[3500]]
    csv = next(t for t in tables.values() if t["file"] == "sales.csv")
    assert sql.execute(f'SELECT COUNT(*) FROM "{csv["name"]}"')["rows"] == [[3]]  # duplicate preserved
    assert source_hashes(split_root) == before
    assert Ingestor(DatasetSplit(split_root), tmp_path / "cache").catalog("shop_a").fingerprint == catalog.fingerprint


@pytest.mark.parametrize("statement", ["DELETE FROM calendar", "ATTACH DATABASE ':memory:' AS other", "PRAGMA query_only=OFF",
    "CREATE TEMP TABLE stolen(x)", "SELECT load_extension('x')", "SELECT 1; SELECT 2"])
def test_readonly_sql_rejects_mutation_and_file_access(statement, split_root, tmp_path):
    catalog = Ingestor(DatasetSplit(split_root), tmp_path / "cache").catalog("shop_a")
    with pytest.raises(DataMateError):
        SQLExecutor(catalog.database).execute(statement)


def test_bounded_rows_and_execution_time(split_root, tmp_path):
    catalog = Ingestor(DatasetSplit(split_root), tmp_path / "cache").catalog("shop_a")
    executor = SQLExecutor(catalog.database, timeout_seconds=0.01, max_rows=1)
    assert executor.execute("SELECT 1 UNION ALL SELECT 2")["truncated"] is True
    with pytest.raises(DataMateError, match="interrupted"):
        executor.execute("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) FROM n")


def test_never_writes_cache_inside_data(split_root):
    with pytest.raises(DataMateError):
        Ingestor(DatasetSplit(split_root), split_root / "cache")
