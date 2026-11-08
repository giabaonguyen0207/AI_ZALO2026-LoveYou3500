# Gói benchmark nhóm — 200 câu TRAIN, protocol v2

Để chủ dự án chạy agent hiện có trên Colab, dùng [notebook 200 câu riêng](team200-colab.md).
Notebook xuất đúng hai file nộp nhóm, đồng thời lưu review/log riêng và tự
ngắt runtime. Tiện ích này không yêu cầu đồng đội dùng cùng model/framework.

Chuẩn bị ngày **07/11/2026**, theo yêu cầu chọn ngẫu nhiên 200 câu từ train
để mọi thành viên chạy agent riêng và chấm bằng đáp án có sẵn. Đây là dữ liệu
đã có trong workspace. Bản v2 theo yêu cầu rút gọn: mọi hệ thống agent/model/
framework dùng được; chỉ nộp submission và sáu trường metadata, không cần
question log. **Giữ nguyên 200 câu, thứ tự, dữ liệu và đáp án của bản đầu.**

## File để gửi

Trong `outputs/team-benchmark-train200-2026-11-07-v2/` (bản hiện dùng):

| File | Mục đích |
|---|---|
| `participants.zip` | **Gửi cho cả nhóm**, khoảng 125,50 MB / 119,69 MiB. Có dữ liệu đầy đủ của các doanh nghiệp trong mẫu, rulebook, calendar, 200 câu, ID, manifest, hướng dẫn và mẫu kết quả; không có nhãn. |
| `coordinator-private.zip` | Người điều phối giữ. Có 200 nhãn/alias gốc và evaluator local đã khóa; Python 3.11+, không cần cài package hoặc GPU. |
| `TIN_NHAN_GUI_NHOM.txt` | Nội dung tin nhắn có thể sao chép gửi kèm gói participant. |
| `SHA256SUMS.txt` | SHA256 của hai ZIP để kiểm tra truyền file. |
| `release_verification.json` | Nguồn, seed, hash, số câu, kiểm tra ZIP và trạng thái chưa inference. |

## Danh sách đã khóa

- Benchmark ID: `datamate-train200-seed4000930841`.
- Seed sinh ngẫu nhiên một lần: **4000930841**.
- Ngẫu nhiên đều, không hoàn lại: `random.Random(seed).sample(questions sorted by id, 200)`.
- Phạm vi: toàn bộ **3.000 câu TRAIN**, nguồn đã khớp allowlist
  `configs/training_sources.json`, fingerprint
  `5d598c5d603020c8077df7802cc6fb06b364af9914eb88425bd7664c639978ea`.
- Không cân đối type/doanh nghiệp, không ép câu lỗi, không loại câu cũ hoặc
  holdout. Nhãn chỉ được ghép vào sau khi danh sách đã cố định.
- Mẫu có **200 ID duy nhất, 28 doanh nghiệp**. Giữ nguyên dữ liệu gốc của 28
  doanh nghiệp; chỉ giảm tập câu hỏi. Chỉ dùng rulebook/calendar TRAIN.
- SHA256 của file 200 câu:
  `7ae2079219d0a79ca1dd66952610ceb091e36c0ecf65359b11dac8263c67b782`.
- Gói participant SHA256:
  `f24e938bd60764034c9272881aa760b53ebc0116a8975e3449678d25a7151652`.
- Gói coordinator SHA256:
  `0bf2b5c1f44ea56f63ea77d61cb64aa0d53cb7006761d524d0bdadb9e092a3a3`.

Thống kê **sau khi chọn**, không dùng để chọn lại: 136 number, 35 percent,
15 list, 5 date, 4 string, 4 unanswerable, 1 boolean; trùng 2 câu với mẫu
random100 trước đây. Audit chi tiết nằm trong gói coordinator. Không suy ra
đây là tập chưa dùng để fine-tune/debug; các hệ thống khác cần khai báo phần
train đã dùng. Phân chia training/validation và source inference hiện tại
không bị thay đổi bởi yêu cầu chuẩn bị này.

## Người nhận chạy thế nào

Giải nén `participants.zip`, đọc `participant/README.md`, rồi chạy:

```bash
python benchmark.py verify
```

Agent riêng đọc tuần tự `participant/dataset/questions.jsonl` và tra dữ liệu
ở `participant/dataset/data/<business_id>/`. Dataset chỉ đọc, output/cache ở
ngoài. Bộ câu đã khóa sẵn; không bốc lại chỉ bằng cách nhập seed.

Mỗi người tích hợp đầu vào với hệ thống đang có của mình. Hướng dẫn chung
không gắn với một model hoặc yêu cầu source agent DataMate. Chưa chạy model
thật trong phiên chuẩn bị này.

## Những gì mọi người nộp lại

Mỗi người gửi một thư mục/ZIP riêng, tối thiểu:

```text
ten/
  submission.jsonl
  run_metadata.json
```

`submission.jsonl` giữ schema cuộc thi: mỗi dòng có ID gốc, type và answer
đúng kiểu JSON đối với bảy loại giá trị. Ví dụ (ID/giá trị chỉ minh họa):

```jsonl
{"id":"q_example_number","type":"number","answer":1234500}
{"id":"q_example_string","type":"string","answer":"Sản phẩm minh họa"}
{"id":"q_example_unanswerable","type":"unanswerable","reason":"Thiếu dữ liệu cần thiết"}
{"id":"q_example_abstain","type":"abstain"}
```

`abstain` và `unanswerable` không cần answer; reason là tùy chọn. Không trộn
timing/model/log vào submission. Template mới có ví dụ đủ chín type, không
phải 200 dòng abstain; thay ID minh họa bằng đúng 200 ID gốc và đáp án tự sinh
khi nộp. Không phát đáp án chuẩn trong gói participant.

`run_metadata.json` chỉ có sáu trường:

```json
{
  "participant": null,
  "models": [],
  "gpu": null,
  "load_seconds": null,
  "inference_seconds": null,
  "compute_units_used": null
}
```

Models liệt kê model inference của hệ thống; hệ thống không dùng model để
[]. GPU ghi tên/số lượng hoặc CPU. Load là khởi tạo/nạp model/chuẩn bị chung
trước batch; inference là wall time batch 200 câu, bao gồm tool/retry/ghi
kết quả. Hai khoảng không chồng nhau. Tổng = load + inference; TB batch/câu
= inference / 200 khi nộp đủ. Không tính cài/tải tài nguyên ban đầu. Nếu
resume cộng mọi phiên, kể cả load lại; chưa biết giữ null. Unit tùy chọn.

Không yêu cầu question log để chấm hay so đáp án. Không có log thì không
có p50/p95/max hoặc thời gian từng câu; không tự suy ra chúng từ thời gian
batch. Nếu ai tự nguyện gửi log cũ, công cụ vẫn đọc được. Yêu cầu log của
Docker thi chính thức trong ngữ cảnh cuộc thi là yêu cầu riêng.

## Chấm và so sánh

Trong thư mục coordinator sau khi giải nén, nhận các thư mục kết quả rồi:

```bash
python benchmark.py compare --run an=received/an --run binh=received/binh --output comparison-01
```

Xuất `REPORT.md`, `leaderboard.csv/json`, `comparison.csv`, `by_business.csv`
và `pairwise.csv` khi có ít nhất hai người. Bảng từng câu có nhãn/alias chuẩn
và dự đoán/điểm của từng hệ thống, không phụ thuộc question log. Thời gian
từng câu/lỗi chỉ có nếu log tự nguyện đã ghi. CSV có UTF-8 BOM.

Điểm chính là PA = (đúng − sai) / **200**. Thiếu/abstain 0, sai type/giá trị
hoặc payload sai schema −1; `unanswerable` sai vẫn −1. ID trùng/ngoài mẫu hoặc
JSONL hỏng bị từ chối. Đồng điểm là đồng hạng; thời gian khác GPU chỉ dùng
tham khảo. Cùng dùng một evaluator được đóng băng, không dùng evaluator mỗi
người tự viết. Gói chấm mô tả chính xác hành vi local (gồm sai số scalar 1e-8
và không tự round percent); chưa thay thế evaluator chính thức.

## Kiểm chứng đã thực hiện

- **18 kiểm thử qua**: bốn kiểm thử release/đối chiếu và các kiểm thử hợp đồng
  type/matching hiện có. Kiểm tra giữ nhãn riêng, đầy đủ dữ liệu, ID/metadata
  sai, mẫu số của lượt nộp thiếu, ví dụ JSONL hợp lệ cho cả chín type, metadata
  sáu trường không cần log, và chạy scorer độc lập ngoài source dự án.
- Đã xác minh nguồn TRAIN trước khi sao chép; hai ZIP vượt CRC và đối chiếu
  SHA256 từng file. Gói participant không chứa `answers.jsonl`.
- Đủ 200 nhãn khớp nguyên bản train, ghép đúng ID/thứ tự. Đã tái tạo đúng mẫu
  từ seed, không bốc lại vì thống kê sau chọn.
- Đối chiếu byte của 200 câu và nhãn v2 với v1: giống hoàn toàn. Kiểm tra
  trên release v2 thật bằng submission/metadata sáu trường **giả lập**, không
  log: chấm độc lập thành công, tính mean từ batch và giữ p95 null. Đây chỉ
  là kiểm tra công cụ, **không phải kết quả agent**. Bằng chứng nằm ở
  `outputs/verification-team-benchmark-v2-2026-11-07/verification.json`.
- **Chưa inference 200 câu bằng model**, chưa có PA/tốc độ/unit thực tế cho
  bộ benchmark này. Không có public/private test trong hai gói.

Mã tạo gói: `tools/prepare_team_benchmark.py`; nội dung hướng dẫn/ví dụ:
`tools/team_benchmark_instructions.py`; mã kiểm tra/chấm: `tools/team_benchmark.py`.
Muốn tái tạo danh sách, dùng `--seed 4000930841`
và một output-dir mới; không ghi đè bản đã phát. Danh sách câu tái tạo được,
byte ZIP có thể khác theo timestamp hoặc phiên bản mã; bản phát căn cứ hash
đã lưu ở trên. Không tạo mẫu mới khi chỉ muốn chạy lại agent.

Đối chiếu nguồn ngày 07/11/2026: kết quả lập chỉ mục phần Description của
[portal DataMate chính thức](https://challenge.zalo.ai/portal/datamate/)
liệt kê chín type và ví dụ output có id/type/answer, phù hợp ngữ cảnh đã lưu.
Mở trang trực tiếp vẫn không trả nội dung đọc được; chưa xác minh độc lập
toàn bộ Evaluation/Rule/Submission Guidance. Các chi tiết còn lại vẫn theo
bản tổng hợp của đội, không coi việc kiểm tra Description là xác nhận mọi luật.

Bản đầu ở `outputs/team-benchmark-train200-2026-11-07/` được giữ để đối chiếu;
gửi gói **v2** cho nhóm để dùng hồ sơ nộp đã rút gọn.
