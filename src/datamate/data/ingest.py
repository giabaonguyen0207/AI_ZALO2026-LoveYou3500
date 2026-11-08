from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
import sqlite3
import threading
import unicodedata
import uuid
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from ..domain import DataMateError, json_text
from .split import DatasetSplit, ensure_outside_dataset

INGEST_VERSION = 1
_build_lock = threading.Lock()


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def slug(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.replace("đ", "d").replace("Đ", "D"))
    text = text.encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "sheet"


def cell(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if value is None or isinstance(value, (str, int, float)):
        return value
    return str(value)


def unique_headers(values) -> tuple[list[str], list[dict]]:
    headers, mapping, used = [], [], {"_dm_row", "_dm_file", "_dm_sheet"}
    for i, value in enumerate(values):
        original = str(value).strip() if value is not None else ""
        base = original or f"column_{i + 1}"
        name, suffix = base, 2
        while name.casefold() in used:
            name, suffix = f"{base}__{suffix}", suffix + 1
        used.add(name.casefold())
        headers.append(name)
        if name != original:
            mapping.append({"position": i + 1, "original": original, "sql_name": name})
    return headers, mapping


def header_index(rows: list[tuple]) -> int:
    # Excel reports often start with titles and blank lines. Prefer a dense,
    # textual header; earliest wins ties. This is generic, never keyed by ID.
    if not rows:
        return 0
    scores = []
    for i, row in enumerate(rows):
        present = [value for value in row if value is not None and str(value).strip()]
        if not present:
            scores.append((-1, -i))
            continue
        textual = sum(isinstance(value, str) and not re.fullmatch(r"[\d\s.,/%:+-]+", value.strip()) for value in present)
        scores.append((len(present) + textual / len(present), -i))
    return max(range(len(rows)), key=lambda i: scores[i])


def attach_excel_epochs(tables, files):
    epochs={}
    for path in files:
        if path.suffix.lower()!='.xlsx':
            continue
        with zipfile.ZipFile(path) as archive:
            root=ET.fromstring(archive.read('xl/workbook.xml'))
        props=root.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}workbookPr')
        epoch='1904-01-01T00:00:00' if props is not None and props.get('date1904','0') in {'1','true'} else '1899-12-30T00:00:00'
        epochs.setdefault(path.name,set()).add(epoch)
    for table in tables:
        if table['file'] in epochs and len(epochs[table['file']])==1:
            table['excel_epoch']=next(iter(epochs[table['file']]))


@dataclass
class Catalog:
    database: Path
    tables: list[dict]
    warnings: list[str]
    fingerprint: str
    semantic: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"tables": self.tables, "warnings": self.warnings, "semantic": self.semantic,
                "helpers": ["num(x)", "num_en(x)", "num_vi(x)", "iso_date(x)",
                            "iso_date_year(x, year)", "norm(x)", "round_half_up(x, places)",
                            "status_class(x)", "search_key(x)", "parse_number(x)"]}


class Ingestor:
    def __init__(self, split: DatasetSplit, cache_dir: str | Path):
        self.split = split
        self.cache_dir = ensure_outside_dataset(cache_dir, split.root)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._catalogs: dict[tuple[str,bool], Catalog] = {}

    def catalog(self, business_id: str, normalize: bool = True) -> Catalog:
        with _build_lock:
            cache_key = (business_id, normalize)
            if cache_key in self._catalogs:
                return self._catalogs[cache_key]
            directory = self.split.business_path(business_id)
            files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".sqlite", ".db"})
            if not files:
                raise DataMateError("Không tìm thấy CSV/XLSX/SQLite của doanh nghiệp")
            digest = hashlib.sha256(str(INGEST_VERSION).encode())
            inputs = files + [self.split.root / "calendar.csv", self.split.root / "rulebook.md"]
            if (directory / "README.md").exists():
                inputs.append(directory / "README.md")
            for path in inputs:
                digest.update(str(path.relative_to(self.split.root)).encode())
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            fingerprint = digest.hexdigest()
            target = self.cache_dir / fingerprint
            db_path, meta_path = target.with_suffix(".sqlite"), target.with_suffix(".json")
            if db_path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                result = Catalog(db_path, meta["tables"], meta["warnings"], fingerprint)
                attach_excel_epochs(result.tables,files)
                from .normalization import prepare_catalog
                if normalize:
                    result = prepare_catalog(result, self.split.rulebook, self.split.readme(business_id), self.cache_dir)
                self._catalogs[cache_key] = result
                return result
            tables, warnings = [], []
            staging = db_path.with_name(f"{fingerprint}.{uuid.uuid4().hex}.tmp")
            connection = sqlite3.connect(staging)
            try:
                for index, path in enumerate(files):
                    if path.suffix.lower() == ".csv":
                        self._csv(connection, path, f"raw_{index}_{slug(path.stem)}", tables, warnings)
                    elif path.suffix.lower() == ".xlsx":
                        self._xlsx(connection, path, f"raw_{index}_{slug(path.stem)}", tables, warnings)
                    else:
                        self._sqlite(connection, path, f"raw_{index}", tables, warnings)
                self._csv(connection, self.split.root / "calendar.csv", "calendar", tables, warnings)
                connection.commit()
            except Exception:
                connection.close()
                staging.unlink(missing_ok=True)
                raise
            finally:
                connection.close()
            staging.replace(db_path)
            temp_meta = meta_path.with_suffix(".json.tmp")
            temp_meta.write_text(json_text({"tables": tables, "warnings": warnings}, indent=2), encoding="utf-8")
            temp_meta.replace(meta_path)
            result = Catalog(db_path, tables, warnings, fingerprint)
            attach_excel_epochs(result.tables,files)
            from .normalization import prepare_catalog
            if normalize:
                result = prepare_catalog(result, self.split.rulebook, self.split.readme(business_id), self.cache_dir)
            self._catalogs[cache_key] = result
            return result

    def _table(self, connection, name, headers, rows, path, sheet, tables, warnings, header_row=1):
        headers, mapping = unique_headers(headers)
        if not headers:
            warnings.append(f"Bỏ sheet/file rỗng: {path.name}/{sheet}")
            return
        columns = ["_dm_row", "_dm_file", "_dm_sheet", *headers]
        # No affinity: numeric XLSX values stay numeric, text cells stay text.
        connection.execute(f"CREATE TABLE {quote(name)} ({','.join(quote(c) for c in columns)})")
        insert = f"INSERT INTO {quote(name)} VALUES ({','.join('?' for _ in columns)})"
        count, batch = 0, []
        for row_no, row in enumerate(rows, header_row + 1):
            if not any(value is not None and str(value) != "" for value in row):
                continue
            if len(row) > len(headers) and any(value is not None for value in row[len(headers):]):
                raise DataMateError(f"Dòng rộng hơn header: {path.name}/{sheet}:{row_no}")
            values = [cell(x) for x in row[:len(headers)]]
            values += [None] * (len(headers) - len(values))
            batch.append([row_no, path.name, sheet, *values])
            count += 1
            if len(batch) >= 2000:
                connection.executemany(insert, batch)
                batch.clear()
        if batch:
            connection.executemany(insert, batch)
        tables.append({"name": name, "file": path.name, "sheet": sheet, "columns": headers,
                       "rows": count, "header_row": header_row, "renamed_columns": mapping})
        if mapping:
            warnings.append(f"Header trống/trùng được đổi tên ở {name}; xem renamed_columns")
        if header_row > 1:
            warnings.append(f"{name}: phát hiện header ở dòng {header_row}; kiểm tra sample nếu bất thường")

    def _csv(self, connection, path, name, tables, warnings):
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                sample = handle.read(16384)
                handle.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.reader(handle, dialect)
                self._table(connection, name, next(reader, []), reader, path, "", tables, warnings)
        except UnicodeDecodeError as exc:
            raise DataMateError(f"CSV không phải UTF-8: {path.name}; thêm adapter encoding rõ ràng") from exc

    def _xlsx(self, connection, path, prefix, tables, warnings):
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        names = []
        try:
            for i, sheet in enumerate(workbook.worksheets):
                rows = iter(sheet.values)
                initial = list(itertools.islice(rows, 25))
                if not initial or not any(any(x is not None for x in row) for row in initial):
                    continue
                header = header_index(initial)
                values = list(initial[header])
                while values and values[-1] is None:
                    values.pop()
                name = f"{prefix}__s{i}_{slug(sheet.title)}"
                self._table(connection, name, values, itertools.chain(initial[header+1:], rows),
                            path, sheet.title, tables, warnings, header + 1)
                names.append((name, tables[-1]["columns"]))
            # Monthly sheets with identical columns can be queried as one view.
            if len(names) > 1 and all(columns == names[0][1] for _, columns in names):
                view = prefix + "__all"
                connection.execute(f"CREATE VIEW {quote(view)} AS " + " UNION ALL ".join(f"SELECT * FROM {quote(n)}" for n, _ in names))
                tables.append({"name": view, "file": path.name, "sheet": "ALL SHEETS",
                               "columns": names[0][1], "view_of": [name for name, _ in names]})
            warnings.append(f"{path.name}: đọc giá trị cache của công thức Excel (data_only); ô chưa được Excel tính có thể là NULL")
        finally:
            workbook.close()

    def _sqlite(self, connection, path, prefix, tables, warnings):
        source = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            names = source.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
            for i, (original,) in enumerate(names):
                cursor = source.execute(f"SELECT * FROM {quote(original)}")
                headers = [description[0] for description in cursor.description]
                self._table(connection, f"{prefix}_{i}_{slug(original)}", headers, cursor, path, original, tables, warnings, 0)
        finally:
            source.close()
