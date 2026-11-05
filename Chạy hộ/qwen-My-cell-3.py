# Chép toàn bộ nội dung file này thay cho cell thứ ba, rồi chạy cell đó và cell cuối.
# Không cần nạp lại model nếu phiên Colab hiện tại vẫn còn hoạt động.

import json
import sqlite3
import subprocess
import re
import pandas as pd

# 1. tự động trích xuất cấu trúc loại file (.csv, .xlsx, .sqlite)
def extract_business_schema(biz_dir: Path) -> str:
    schema_info = []
    for file_path in sorted(biz_dir.iterdir()):
        if file_path.name == "README.md" or file_path.name.startswith("."):
            continue
        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path, dtype=str, nrows=3, keep_default_na=False)
                schema_info.append({
                    "file_name": file_path.name,
                    "type": "csv",
                    "columns": list(df.columns),
                    "sample_rows": df.to_dict(orient="records")
                })
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                excel_file = pd.ExcelFile(file_path)
                sheets = {}
                for sheet in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet, dtype=str, nrows=3, keep_default_na=False)
                    sheets[sheet] = {
                        "columns": list(df.columns),
                        "sample_rows": df.to_dict(orient="records")
                    }
                schema_info.append({
                    "file_name": file_path.name,
                    "type": "excel",
                    "sheets": sheets
                })
            elif file_path.suffix.lower() in [".sqlite", ".db", ".sqlite3"]:
                conn = sqlite3.connect(file_path)
                cursor = conn.cursor()
                tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                db_tables = {}
                for (t_name,) in tables:
                    df = pd.read_sql_query(f"SELECT * FROM `{t_name}` LIMIT 3", conn)
                    db_tables[t_name] = {
                        "columns": list(df.columns),
                        "sample_rows": df.astype(str).to_dict(orient="records")
                    }
                conn.close()
                schema_info.append({
                    "file_name": file_path.name,
                    "type": "sqlite",
                    "tables": db_tables
                })
        except Exception as e:
            schema_info.append({"file_name": file_path.name, "error_reading_schema": str(e)})
            
    return json.dumps(schema_info, ensure_ascii=False, indent=2)

# 2. Bộ công cụ tiền xử lý vào mọi file code do AI viết ra
PRE_INJECTED_CODE = """import pandas as pd
import numpy as np
import sqlite3
import json
import re
from pathlib import Path

def parse_vn_date(series):
    '''Chuyển đổi mọi định dạng ngày tiếng Việt (vd: 12 thg 1 2026, 2026-03-04, 15/08/2026) sang datetime'''
    s = series.astype(str).str.strip()
    s = s.str.replace(r'\\s*thg\\s*', '-', regex=True)
    s = s.str.replace(r'\\s*tháng\\s*', '-', regex=True)
    s = s.str.replace(r'\\s*năm\\s*', '-', regex=True)
    return pd.to_datetime(s, dayfirst=True, format='mixed', errors='coerce')

def clean_num(series):
    '''Làm sạch cột tiền tệ/số lượng có dấu phẩy hoặc chữ cái và ép kiểu sang float'''
    s = series.astype(str).str.replace(',', '', regex=False).str.strip()
    s = s.str.extract(r'([-+]?\\d*\\.?\\d+)')[0]
    return pd.to_numeric(s, errors='coerce').fillna(0.0)
"""

def chat_token_ids(messages):
    # Transformers mới mặc định trả BatchEncoding (input_ids, attention_mask).
    # vLLM cần danh sách số nguyên, nên chỉ định rõ kiểu trả về.
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=False, return_tensors=None,
    )
    if not isinstance(ids, list) or not ids or any(type(item) is not int for item in ids):
        raise TypeError("Tokenizer phải trả danh sách số nguyên token; không gửi dict/chuỗi vào vLLM.")
    return ids


# 3. Hàm giải quyết từng câu hỏi với vòng lặp Tự sửa lỗi (tối đa 3 lần)
def solve_question(q_item: dict, data_dir: Path, max_retries: int = 3) -> dict:
    q_id = q_item["id"]
    biz_id = q_item["business_id"]
    biz_dir = data_dir / "data" / biz_id
    
    rulebook = (data_dir / "rulebook.md").read_text(encoding="utf-8") if (data_dir / "rulebook.md").exists() else ""
    calendar_path = data_dir / "calendar.csv"
    readme = (biz_dir / "README.md").read_text(encoding="utf-8") if (biz_dir / "README.md").exists() else "Không có README"
    schema_str = extract_business_schema(biz_dir)
    
    base_prompt = f"""Bạn là một AI Data Analyst xuất sắc. Nhiệm vụ của bạn là viết MỘT đoạn script Python hoàn chỉnh để truy vấn dữ liệu và trả lời câu hỏi.

THÔNG TIN CÂU HỎI:
{json.dumps(q_item, ensure_ascii=False)}

ĐƯỜNG DẪN THƯ MỤC DỮ LIỆU CỦA CỬA HÀNG (BIZ_DIR): {str(biz_dir)!r}
ĐƯỜNG DẪN FILE LỊCH ÂM-DƯƠNG (CALENDAR_CSV): {str(calendar_path)!r}

MÔ TẢ CỬA HÀNG (README.md):
{readme}

CẤU TRÚC CÁC FILE DỮ LIỆU (SCHEMA):
{schema_str}

QUY TẮC CHUNG (RULEBOOK):
{rulebook}

CÔNG CỤ CÓ SẴN TRONG HỆ THỐNG (Không cần định nghĩa lại, hãy gọi trực tiếp):
- `parse_vn_date(series)`: Chuyển cột ngày tháng lộn xộn sang chuẩn `datetime64` của Pandas.
- `clean_num(series)`: Xóa dấu phẩy và chuyển cột chuỗi sang số `float`.

YÊU CẦU BẮT BUỘC:
1. Đọc đúng file trong `{str(biz_dir)!r}` dựa vào SCHEMA. (Với CSV dùng `pd.read_csv(..., dtype=str)`, Excel dùng `pd.read_excel`, SQLite dùng `sqlite3.connect`).
2. Ngày `as_of_date` ({q_item['as_of_date']}) là mốc "hôm nay". Tuân thủ nghiêm ngặt quy tắc thời gian, doanh thu, đơn hủy/trả trong RULEBOOK.
3. Ở dòng cuối cùng của script, BẮT BUỘC dùng lệnh `print(json.dumps(result, ensure_ascii=False))` để in ra ĐÚNG 1 DÒNG JSON theo chuẩn sau:
   - Nếu trả lời được: {{"id": "{q_id}", "type": "<kiểu>", "answer": <giá_trị>}}
     (Trong đó <kiểu> chỉ được là: "string", "number", "percent", "date", "boolean", "list", "table").
     *Lưu ý kiểu "table"*: `answer` phải có dạng {{"columns": ["cot_1", "cot_2"], "rows": [["gt1", 100], ["gt2", 200]]}}
   - Nếu dữ liệu hoàn toàn không có thông tin để tính (ví dụ hỏi lợi nhuận nhưng không có giá vốn): {{"id": "{q_id}", "type": "unanswerable"}}
   - Nếu không chắc chắn: {{"id": "{q_id}", "type": "abstain"}}

Chỉ viết code nằm trong cặp thẻ ```python ... ```, không giải thích gì thêm."""

    messages = [{"role": "user", "content": base_prompt}]
    
    for attempt in range(max_retries):
        # Đếm đúng token sẽ gửi cho vLLM, dành đủ chỗ cho code đầu ra.
        prompt_ids = chat_token_ids(messages)
        if len(prompt_ids) + MAX_OUTPUT_TOKENS > MAX_MODEL_LEN and len(messages) > 3:
            # Khi lịch sử sửa lỗi quá dài, giữ câu hỏi gốc và lần sửa gần nhất.
            # Không cắt rulebook, README hoặc schema.
            messages = [messages[0], *messages[-2:]]
            prompt_ids = chat_token_ids(messages)
        if len(prompt_ids) + MAX_OUTPUT_TOKENS > MAX_MODEL_LEN:
            print(f"[{q_id}] Bỏ qua: {len(prompt_ids)} token đầu vào + "
                  f"{MAX_OUTPUT_TOKENS} token đầu ra vượt giới hạn {MAX_MODEL_LEN}.")
            return {"id": q_id, "type": "abstain"}
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_ids}], sampling_params, use_tqdm=False
        )
        response = outputs[0].outputs[0].text
        
        # Trích xuất khối code Python
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, flags=re.S)
        code = blocks[-1] if blocks else response.strip()
        full_script = PRE_INJECTED_CODE + "\n" + code
        
        script_file = Path("temp_runner.py")
        script_file.write_text(full_script, encoding="utf-8")
        
        try:
            # Chạy thử code với timeout 45 giây
            proc = subprocess.run(["python", str(script_file)], capture_output=True, text=True, timeout=45)
            
            if proc.returncode == 0 and proc.stdout.strip():
                last_line = proc.stdout.strip().splitlines()[-1]
                parsed_json = json.loads(last_line)
                # Kiểm tra tính hợp lệ của JSON đầu ra
                if "id" in parsed_json and "type" in parsed_json:
                    parsed_json["id"] = q_id # Đảm bảo đúng ID
                    return parsed_json
                else:
                    error_msg = f"Output JSON thiếu trường 'id' hoặc 'type': {last_line}"
            else:
                error_msg = f"Lỗi Runtime (Stderr):\n{proc.stderr[-1000:]}\nStdout:\n{proc.stdout[-500:]}"
        except subprocess.TimeoutExpired:
            error_msg = "Code chạy quá thời gian (Timeout > 45s). Hãy tối ưu hóa vòng lặp bằng vectorization của Pandas."
        except Exception as e:
            error_msg = f"Lỗi hệ thống hoặc JSON không hợp lệ: {str(e)}\nStdout: {proc.stdout[-500:] if 'proc' in locals() else ''}"
            
        # Nạp lỗi vào lịch sử chat để AI tự sửa ở vòng lặp tiếp theo
        messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
        messages.append({
            "role": "user", 
            "content": f"Code vừa chạy bị lỗi như sau:\n{error_msg}\n\nHãy kiểm tra lại tên cột, kiểu dữ liệu và viết lại toàn bộ đoạn code Python đã sửa lỗi trong thẻ ```python ... ```."
        })
        
    # Nếu sau 3 lần tự sửa vẫn lỗi, chọn "abstain" để KHÔNG BỊ TRỪ ĐIỂM trên bảng xếp hạng
    return {"id": q_id, "type": "abstain"}

print("Đã nạp xong hàm lõi của AI Agent!")