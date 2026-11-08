"""Participant-facing protocol: any agent, official-shaped JSONL, six metadata fields."""

SUBMISSION_EXAMPLES = [
    {"id": "q_example_number", "type": "number", "answer": 1234500},
    {"id": "q_example_percent", "type": "percent", "answer": 12.34},
    {"id": "q_example_string", "type": "string", "answer": "Sản phẩm minh họa"},
    {"id": "q_example_date", "type": "date", "answer": "2026-11-01"},
    {"id": "q_example_boolean", "type": "boolean", "answer": True},
    {"id": "q_example_list", "type": "list", "answer": ["Mặt hàng A", "Mặt hàng B"]},
    {"id": "q_example_table", "type": "table",
     "answer": {"columns": ["ten", "so_tien"], "rows": [["Khách minh họa", 125000]]}},
    {"id": "q_example_unanswerable", "type": "unanswerable", "reason": "Thiếu dữ liệu cần thiết"},
    {"id": "q_example_abstain", "type": "abstain"},
]


PARTICIPANT_README = """# Benchmark nhóm — {size} câu TRAIN

Benchmark **{benchmark_id}**, seed **{seed}**, chuẩn bị {date} theo giờ Việt Nam.
Mỗi người dùng hệ thống agent/model/framework/quantization của mình. Gói này
chỉ quy định cùng dữ liệu đầu vào và cùng định dạng kết quả, không yêu cầu
dùng một model hay source agent cụ thể.

## Chạy bộ câu chung

1. Mở thư mục `participant` sau khi giải nén. Có thể kiểm tra file bằng
   Python 3.11+: `python benchmark.py verify` (không cần pip hoặc GPU).
2. Cho hệ thống của bạn đọc {size} câu trong `dataset/questions.jsonl` theo
   thứ tự file. Không bốc lại mẫu. Mỗi câu có `id`, `business_id`, `question`,
   `as_of_date`; dữ liệu đầy đủ ở `dataset/data/<business_id>/`.
3. Dùng `dataset/rulebook.md`, `dataset/calendar.csv` và `as_of_date` tương
   ứng. Dataset chỉ đọc; output/cache ở thư mục riêng.
4. Khóa một phiên bản hệ thống, chạy tự động, ghi kết quả từng câu ngay khi
   xong; không dùng nhãn hay sửa đáp án tay. Nếu resume, giữ nguyên hệ thống
   và các đáp án đã ghi, cộng thời gian của mọi phiên.

## Chỉ gửi lại hai file

```text
ten-thanh-vien/
  submission.jsonl
  run_metadata.json
```

### submission.jsonl

Mỗi dòng là một JSON object, đúng định dạng submission cuộc thi đã lưu trong
ngữ cảnh của đội:

```jsonl
{{"id":"q_example_number","type":"number","answer":1234500}}
{{"id":"q_example_string","type":"string","answer":"Sản phẩm minh họa"}}
{{"id":"q_example_unanswerable","type":"unanswerable","reason":"Thiếu dữ liệu cần thiết"}}
{{"id":"q_example_abstain","type":"abstain"}}
```

`templates/submission.template.jsonl` có ví dụ cả chín type với ID và giá trị
MINH HỌA, không phải đáp án bộ câu. File nộp thật thay bằng {size} ID gốc từ
`question_ids.json` và đáp án do hệ thống tự sinh.

- `number`, `percent`, `string`, `date`, `boolean`, `list`, `table`: có
  `id`, `type`, `answer`; giữ đúng kiểu JSON của answer.
- `abstain`: `id`, `type`, không cần answer.
- `unanswerable`: `id`, `type`; reason tùy chọn, không cần answer.
- Không thêm thời gian, điểm, model hay reasoning vào submission. Không nộp
  alias; alias chỉ có trong đáp án chuẩn mà người điều phối giữ.
- List giữ thứ tự. Number/percent là số JSON, không phải chuỗi chứa đơn vị;
  percent xuất số cuối đã làm tròn hai chữ số.

### run_metadata.json — chỉ sáu trường

Sao chép `templates/run_metadata.template.json` và điền:

| Trường | Nội dung |
|---|---|
| participant | Tên thành viên/hệ thống |
| models | Danh sách tên model/checkpoint dùng khi inference; có thể ghi kèm revision. Hệ thống không dùng model để [] |
| gpu | GPU và số lượng; chạy CPU thì ghi CPU |
| load_seconds | Thời gian khởi tạo: nạp model và chuẩn bị chung trước batch |
| inference_seconds | Thời gian chạy batch {size} câu, gồm đọc dữ liệu/tool/retry và ghi kết quả |
| compute_units_used | Unit thực dùng nếu đo được; Kaggle/GPU thuê hoặc chưa đo thì null |

Hai khoảng thời gian không chồng nhau. Tổng thời gian = load_seconds +
inference_seconds. Không tính cài thư viện/tải model ban đầu; model và
dependency phải được chuẩn bị trước. Resume thì cộng cả các lần load lại.
Trường chưa biết giữ null, không dùng 0 thay cho chưa đo.

Không phải nộp question log, trace hay raw LLM output cho benchmark nội bộ.
Vẫn chấm PA và so từng đáp án được chỉ bằng submission. Khi nộp đủ {size}
câu, thời gian trung bình batch/câu = inference_seconds / {size}; đây là
trung bình từ tổng batch, không có p50/p95 hoặc thời gian riêng từng câu.
Thời gian giữa các GPU chỉ để tham khảo. Gói Docker thi chính thức vẫn phải
đáp ứng yêu cầu log riêng theo ngữ cảnh cuộc thi của đội.

## Điểm và giới hạn

Người điều phối giữ nhãn và chấm bằng cùng evaluator local: PA = (đúng − sai)
/ {size}. Đúng +1, sai type/giá trị −1, abstain/thiếu 0; unanswerable sai −1.
ID trùng/ngoài mẫu hoặc JSONL hỏng bị từ chối. Điểm tổng luôn có mẫu số {size}.
Evaluator local được khóa trong gói, không phải bộ chấm chính thức.

Mẫu ngẫu nhiên đều, không hoàn lại từ toàn bộ {population} câu TRAIN; không
cân đối type/doanh nghiệp, không lọc câu cũ/holdout. Đây là benchmark nội bộ
trên TRAIN có nhãn công khai, không phải holdout chưa quan sát. Không tune
thêm trên mẫu này trong lượt chọn. Các điều kiện eligibility/offline của
cuộc thi vẫn theo ngữ cảnh đội đã lưu; không có quyết định chọn model chung.
"""


COORDINATOR_README = """# Gói chấm riêng — {benchmark_id}, protocol v2

Người điều phối giữ gói này; gửi `participants.zip` cho cả nhóm. Đáp án gốc
của {size} câu ở `answers.jsonl`; không đưa cho agent khi inference.

Python 3.11+, không cần pip/GPU. Mở thư mục coordinator, kiểm tra bằng
`python benchmark.py verify`. Mỗi người nộp hai file:

```text
received/an/submission.jsonl
received/an/run_metadata.json
received/binh/submission.jsonl
received/binh/run_metadata.json
```

Chấm tập trung:

```bash
python benchmark.py compare --run an=received/an --run binh=received/binh --output comparison-01
```

Một người thì dùng một --run. Tên chỉ gồm chữ ASCII, số, _ hoặc -. Đường dẫn
có khoảng trắng thì đặt cả `an=...` trong dấu nháy. Output dùng thư mục mới.

Metadata v2 chỉ có participant, models, gpu, load_seconds, inference_seconds,
compute_units_used. Không yêu cầu question log hay một framework/model cụ thể.
Tổng thời gian = load_seconds + inference_seconds; nếu đủ {size} câu,
TB batch/câu = inference_seconds / {size}. Khi nộp thiếu không suy ra TB bằng
cách chia cho {size}. Thiếu timing giữ null, không biến thành 0.

Kết quả:

- REPORT.md, leaderboard.csv/json: PA, đúng/sai/abstain, coverage, precision,
  số thiếu/invalid, model, GPU, thời gian tổng/TB và unit khai báo.
- comparison.csv: một dòng/câu, có question/as_of_date/doanh nghiệp, nhãn/alias
  chuẩn và type/giá trị/đúng-sai/điểm của từng hệ thống. Không cần log để so
  đáp án. Nếu người gửi tự nguyện kèm log cũ, đọc thêm timing/trace hiện có;
  thiếu log không bị coi là lỗi, không có p50/p95/max khi không đo từng câu.
- by_business.csv: điểm theo doanh nghiệp; tổng vẫn micro PA trên {size} câu.
- pairwise.csv (từ hai người): câu cùng trả lời, đáp án tương đương, câu chỉ
  mỗi người trả lời và số câu có điểm cao hơn đối phương.

Xếp theo PA giảm dần, đồng điểm giữ đồng hạng. Khác GPU thì tốc độ tham khảo.
ID trùng/ngoài mẫu hoặc JSONL hỏng bị từ chối; sai schema −1, thiếu/abstain 0.
Không tải model, không chạy inference, không sửa submission.

Snapshot evaluator local ở _vendor/, hash trong manifest. Hành vi chấm giữ
nguyên: type chặt, alias/Unicode, list theo vị trí, table theo tập hàng,
scalar số sai số tối đa 1e-8; không tự làm tròn percent trong lúc chấm.
Thành viên xuất percent đã làm tròn hai chữ số. Đây không phải evaluator
chính thức. Protocol v2 chỉ đơn giản hóa hồ sơ nộp, không đổi điểm hay mẫu.

selection_audit.json thống kê sau khi khóa mẫu. Đây là TRAIN, có thể trùng
fine-tune/debug; không gọi PA này là hiệu năng private hoặc holdout độc lập.
Không có inference thật nào được chạy chỉ bằng việc tạo gói benchmark.
"""
