from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from ..domain import DataMateError, json_text
from ..execution_audit import audit_execution
from .transforms import register_functions


def authorize(action, arg1, arg2, database, trigger):
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
    if action in allowed:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        if (arg2 or "").lower() in {"load_extension", "writefile", "readfile", "fts3_tokenizer"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SQLExecutor:
    def __init__(self, database: Path, timeout_seconds: float = 12, max_rows: int = 150, deadline: float | None = None):
        self.database = database
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows
        self.deadline = deadline

    @audit_execution
    def execute(self, sql: str, parameters: list | None = None, remaining_seconds: float | None = None) -> dict:
        if not isinstance(sql, str) or not sql.strip() or len(sql) > 50000:
            raise DataMateError("SQL rỗng hoặc quá dài")
        started = time.monotonic()
        if self.deadline is not None:
            remaining_seconds = min(remaining_seconds, self.deadline-started) if remaining_seconds is not None else self.deadline-started
        budget = min(self.timeout_seconds, remaining_seconds) if remaining_seconds is not None else self.timeout_seconds
        if budget <= 0:
            raise DataMateError("Đã hết thời gian truy vấn")
        connection = sqlite3.connect(self.database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            if hasattr(connection, "setconfig") and hasattr(sqlite3, "SQLITE_DBCONFIG_DQS_DML"):
                connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 2_000_000)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 50000)
            register_functions(connection)
            views={row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='view'")}
            references, functions, direct_tables, read_contexts = set(), set(), set(), set()
            def observed_authorize(action, arg1, arg2, database, trigger):
                if action == sqlite3.SQLITE_READ:
                    references.add((arg1,arg2))
                    read_contexts.add((arg1,trigger))
                    if trigger is None:
                        direct_tables.add(arg1)
                elif action == sqlite3.SQLITE_FUNCTION:
                    functions.add(arg2)
                return authorize(action,arg1,arg2,database,trigger)
            connection.set_authorizer(observed_authorize)
            connection.set_progress_handler(lambda: int(time.monotonic() - started > budget), 1000)
            cursor = connection.execute(sql, parameters or [])
            if cursor.description is None:
                raise DataMateError("Chỉ chấp nhận truy vấn trả về bảng")
            columns = [description[0] for description in cursor.description]
            values = cursor.fetchmany(self.max_rows + 1)
            truncated = len(values) > self.max_rows
            rows = [list(row) for row in values[:self.max_rows]]
            result = {"columns": columns, "rows": rows, "truncated": truncated,
                      "sql": sql, "parameters": parameters or [],
                      "returned_rows": len(rows), "elapsed_seconds": round(time.monotonic() - started, 4),
                      "references": [{"table":t,"column":c} for t,c in sorted(references)], "functions":sorted(functions),
                      'direct_tables':sorted(direct_tables),
                      'read_tables':sorted({table for table,_ in references if table}),
                      'logical_tables':sorted({table for table,context in read_contexts if context not in views}
                                              | {context for _,context in read_contexts if context in views}),
                      'read_contexts':[{'table':table,'context':context} for table,context in sorted(read_contexts,key=lambda x:(x[0],str(x[1])))]}
            if len(json_text(result)) > 40000:
                raise DataMateError("Kết quả quá lớn; chọn ít cột hoặc tổng hợp trong SQL")
            return result
        except (sqlite3.Error, ValueError, TypeError) as exc:
            raise DataMateError(f"SQL thất bại: {exc}") from exc
        finally:
            connection.close()
