# Phân tích lượt inference 72 câu · 03/11/2026

**Cập nhật sau báo cáo:** người dùng đã duyệt triển khai các nhóm sửa 1–3. Xem [pipeline v2 và trạng thái kiểm chứng](pipeline-v2-2026-11-03.md). Nội dung bên dưới ghi nhận thời điểm phân tích baseline trước khi sửa; các hash so sánh source là của thời điểm đó.

**Kết luận:** baseline hiện đúng 9/72 câu, sai 42, bỏ qua 21; PA = −0,45833. Có lỗi xác định được ở phần chốt đáp án, giao thức model–công cụ, đọc ngày/số và áp dụng nghiệp vụ. Đề xuất sửa các phần này và kiểm chứng bằng CPU trước; sau đó chạy lại cùng 72 câu trong một phiên GPU. Chưa có cơ sở dùng lượt này để kết luận Gemma thua Qwen hoặc fine-tuning sẽ cải thiện bao nhiêu.

Trạng thái công việc: **đã phân tích, chưa sửa agent, chưa inference LLM thêm, chưa fine-tune**. Các phương án ở cuối tài liệu là đề xuất để người dùng duyệt. Báo cáo này nằm trong `docs/`; tệp kiểm chứng bổ sung nằm trong `outputs/diagnosis-2026-11-03/`.

## 1. Nguồn và độ tin cậy của kết quả

- Nguồn mới: thư mục `outputs/diagnosis-2026-11-02/diagnostic_bundle/` do người dùng cung cấp để phân tích ngày **03/11/2026**; gồm manifest, runtime, 72 log, predictions, references và trace. Tên thư mục không đủ để khẳng định ngày thực hiện inference.
- Đã kiểm tra: đủ 72 ID duy nhất, đúng 3 câu của mỗi doanh nghiệp development; `submission.jsonl`, `review.jsonl` và từng log khớp nhau; câu hỏi, `as_of_date`, đáp án và alias khớp train local nguyên bản.
- Fingerprint toàn bộ train khớp manifest: `5d598c5d603020c8077df7802cc6fb06b364af9914eb88425bd7664c639978ea`. Hash của **8 môđun source được ghi trong bundle** đều khớp code local hiện tại. Hash identity của manifest cũng hợp lệ.
- Chấm lại trực tiếp `submission.jsonl` với nhãn train bằng local evaluator cho đúng các số trong `summary.json`. Chưa có bộ chấm chính thức để đối chiếu độc lập.
- 6 doanh nghiệp holdout không nằm trong lượt này: `biz_107`, `biz_111`, `biz_115`, `biz_120`, `biz_123`, `biz_127`. Không dùng câu hỏi/đáp án của chúng cho phân tích lỗi hoặc lựa chọn phương án trong báo cáo.
- Đã phát lại **226 action SQL** trên CPU với dữ liệu train chỉ đọc và chế độ kiểm tra tên cột chặt hơn: 208 query cho cùng các dòng kết quả đã lưu; 11 query gặp lại lỗi SQL; 7 query bị phát hiện thêm lỗi tên cột. Không gọi LLM hoặc mở phiên Colab.

Đây là mẫu chủ động chọn câu không dấu, viết tắt và có tín hiệu khó, cân đối loại đáp án. Điểm trên mẫu này dùng chẩn đoán và so sánh cùng mẫu; **không phải ước lượng điểm trên toàn train hoặc private test**. Thông tin cuộc thi tổng quát vẫn có nguồn là bản tổng hợp của người dùng; các phát hiện nghiệp vụ dưới đây đối chiếu với **rulebook train thực tế**, không thiết lập luật cho split khác.

## 2. Điểm số và chi phí quan sát được

| Chỉ số | Kết quả |
|---|---:|
| Hoàn tất | 72/72 câu, 24 doanh nghiệp |
| Đúng | 9 — 12,50% toàn mẫu |
| Sai | 42 — 58,33% toàn mẫu |
| Bỏ qua | 21 — 29,17% toàn mẫu |
| PA = (đúng − sai)/72 | **−45,83%** |
| Đúng trong số câu đã trả lời | 9/51 — 17,65% |
| Lượt gọi model | 500 — trung bình 6,94/câu |
| Thời gian riêng batch | 3.964,18 giây — khoảng 66 phút 4 giây |
| Trung bình / trung vị / p95 | 55,05 / 46,70 / 139,01 giây |

Theo mức **10 Unit/giờ** người dùng đã cung cấp, 66 phút tương đương **khoảng 11,01 Unit**, chưa gồm load/chuẩn bị model và thời gian giữ GPU chờ. Đây là phép tính từ thời gian log, không phải hóa đơn Colab.

Runtime thực tế: `gemma4-12b`, chế độ `single`, không quantization, BF16, toàn bộ tham số ở `cuda:0`; GPU **NVIDIA RTX PRO 6000 Blackwell Server Edition**, VRAM báo khoảng **94,97 GiB**. PyTorch `2.11.0+cu130`, Transformers `5.21.0`. Không thấy lỗi hết VRAM hay vượt context trong 72 log. Điểm kém trong lượt này không có bằng chứng là do model bị đẩy sang CPU vì thiếu VRAM.

`runtime.json` ghi `generation_config.do_sample=true` của checkpoint, nhưng source backend khớp hash và khi gọi `generate()` đã truyền **`do_sample=False`, `num_beams=1`**. Vì vậy không thể lấy giá trị mặc định trong runtime để kết luận lượt này đã bật sampling.

| Loại đáp án chuẩn | Số câu | Đúng | Sai | Bỏ qua |
|---|---:|---:|---:|---:|
| number | 10 | 1 | 9 | 0 |
| percent | 10 | 0 | 9 | 1 |
| date | 10 | 1 | 6 | 3 |
| boolean | 8 | 2 | 1 | 5 |
| string | 10 | 5 | 4 | 1 |
| list | 9 | 0 | 6 | 3 |
| table | 7 | 0 | 3 | 4 |
| unanswerable | 8 | 0 | 4 | 4 |

Nhóm không dấu: **2/33 đúng**; viết tắt: **2/33 đúng**; nhóm có tín hiệu khó: **2/34 đúng**. Các nhóm chồng lấp, không cộng để ra tổng. Không có cặp cùng câu viết chuẩn và viết tắt để đối chiếu, nên chưa thể quy chênh lệch điểm riêng cho cách viết.

## 3. Lỗi chốt đáp án đang làm mất kết quả đã tính đúng

**Hai câu công nợ `q_8659862171ab`, `q_b97a6a037819` là bằng chứng rõ nhất.** SQL đã trả đúng khách và số tiền. Chỉ tên cột khác nhau:

- Query: `KH`, `total_debt`; về sau alias thành `khach_hang`, `so_no` hoặc `khách_hàng`, `số_nợ`.
- Model final: `khách hàng`, `số nợ`.
- Runner so sánh nguyên object `action['answer'] != candidate.answer`, gồm cả tên cột. Nó báo “answer khác query cuối”, dù giá trị các dòng không đổi.
- Model lặp final tới hết 10 bước; kết quả nộp là `abstain`.

Đã lấy trực tiếp kết quả query lưu trong log qua chính `answer_from_query('table', ...)`, rồi chấm: **cả hai khớp nhãn train**. Quy tắc chấm bảng cho phép so theo vị trí khi tên cột không khớp. Đây là vấn đề của hợp đồng giữa model và chương trình, có thể kiểm chứng bằng CPU.

Ngoài ra có **5 câu sai mà giá trị có thể khớp nhãn chỉ bằng sửa type/format**: 4 ngày được nộp dưới dạng `string` như `03/01/2026`, và 1 boolean được nộp `string: "Không"`. Đây là đối chiếu sau chạy; **không cộng chúng vào kết quả baseline, không coi là mức tăng của một agent đã sửa**.

Toàn batch có **69 lần final bị từ chối trên 40 câu**:

| Lý do | Số lần |
|---|---:|
| Đáp án đơn nhưng query không phải 1 dòng × 1 cột | 31 |
| Model answer khác object lấy từ query | 25 |
| Query còn ngày raw, không đúng format date | 9 |
| Query trả chuỗi thay vì boolean | 2 |
| Hình dạng query không phù hợp list | 2 |

Việc bắt đáp án xuất phát từ query là đúng hướng chống bịa số. Cần sửa hợp đồng, **không bỏ toàn bộ kiểm tra rồi tin số model viết ra**. Đề xuất final chỉ định `type`, kết quả query nào và cột cần lấy; chương trình xuất giá trị từ kết quả đó. Đổi tên cột và chuyển format chỉ được thực hiện bằng phép biến đổi rõ ràng, có kiểm tra kiểu và giữ nguồn gốc. Boolean tính bằng SQL 0/1; ngày chuẩn hóa theo kiểu cột, không đoán ngày/tháng mơ hồ; phép so sánh/tính phần trăm vẫn do SQL hoặc công cụ xác định thực hiện.

## 4. Giao thức JSON chưa phù hợp với hành vi thực tế của model

**82/500 lượt sinh không đọc được thành action**, xuất hiện trên **30/72 câu**. Trong đó:

- 54 output bắt đầu bằng `thought`.
- 14 output bắt đầu bằng `call:...` thay vì JSON action của DataMate.
- 14 output còn lại có văn xuôi, JSON/markdown kèm nội dung thừa hoặc định dạng hỏng.

`parse_action()` hiện chỉ bỏ `<think>...</think>` và một lớp code fence; không hiểu các dạng `thought`/native call trên. Chẳng hạn câu công nợ `q_9be15002fb02` dành 9 bước cho các output không đọc được; chỉ có một SQL thực thi trước khi hết bước.

Source backend truyền `enable_thinking=False`; runtime cũng lưu template Gemma có xử lý channel/thought và tool call. Tuy nhiên decoder đang dùng `skip_special_tokens=True`, log không giữ token ID hoặc token đặc biệt nguyên bản. **Chưa đủ bằng chứng để kết luận template/checkpoint bị lỗi**; đã đủ bằng chứng về sự lệch giữa giao thức DataMate yêu cầu và output nhận được.

Đề xuất:

1. Kiểm tra đúng prompt sau khi render và output còn token đặc biệt trên một số ca; ghi số token vào/ra và lý do dừng để biết có chạm 768 token hay không.
2. Chọn một giao thức nhất quán: action JSON có schema ràng buộc khi sinh, hoặc adapter tool-call đúng chuẩn checkpoint. Kiểm tra khả năng hỗ trợ của backend trước khi chọn thư viện.
3. Parser chỉ xử lý envelope đã xác định chắc chắn; không chọn tùy ý một JSON trong đoạn có nhiều action hoặc tự đoán phần bị cắt.
4. Phản hồi lỗi nêu đúng trường cần sửa và ví dụ ngắn. Không đưa toàn bộ output hỏng lặp lại vào lịch sử vô hạn.

Chỉ tăng `max_new_tokens` chưa có cơ sở: nhiều lỗi là prefix/giao thức hoặc lặp, tăng trần có thể tốn thêm Unit.

## 5. Dữ liệu đang được đưa cho model quá gần dạng xuất thô

### Ngày đúng cú pháp SQL nhưng sai nghĩa

`q_1cc70c12aa27`: model so cột raw `"Ngày tháng"` với ISO `2026-02-22`, trong khi file dùng `22 thg 2 2026`. Query trả COUNT = 0 rồi final `unanswerable`.

Kiểm chứng CPU trên cùng dữ liệu:

- So raw với ISO: **0 dòng**.
- `iso_date("Ngày tháng")='2026-02-22'`: **243 dòng**.
- Trong đó 240 dòng `Xong`, tổng tiền **13.479.900**, khớp nhãn; 3 dòng hủy, tổng **56.000**.

Các log khác dùng `BETWEEN` trực tiếp trên `dd/mm/yyyy`, `strftime` trên ngày chưa parse hoặc `LIKE` đảo ngày và tháng. Câu so ca sáng/chiều tuần trước `q_9fd096dfdb84` còn lấy 7 ngày lùi từ `as_of_date` thay cho tuần Thứ Hai–Chủ nhật đã quy định.

### Số tiền có nhiều cách viết, kể cả trong cùng cột

SQLite có thể cộng chuỗi bằng cách ép số một phần mà không báo lỗi. Ví dụ `SUM()` trực tiếp trên chuỗi có dấu phân cách tiền không chứng minh đã cộng đúng VND. Các helper hiện có cũng không xử lý mọi kiểu dữ liệu quan sát được.

Kiểm tra `biz_129/chi_tiet_don.csv`, cột `Tiền bán`:

- **144.152 dòng** có hậu tố `đ`, như `50.000đ`.
- **1.327 dòng** có hậu tố `tr`, như `1,45tr`, `1,156tr`.
- Mẫu `1,45tr` có số lượng 10, giá 145.000, giảm giá 0: phù hợp giá trị **1.450.000 VND**.
- `num('50.000đ')` và các helper hiện tại với suffix chưa làm sạch không đọc được; chỉ chọn `num_vi` hay `num_en` cho cả cột vẫn chưa giải quyết hậu tố/quy mô đơn vị.

Cần profile cách viết và chuẩn hóa tiền/ngày một lần trước khi tính; giữ raw, giá trị chuẩn hóa, tình trạng parse và nguồn dòng. Kiểu mơ hồ hoặc không parse được phải hiện thành cảnh báo có thể xử lý, không lặng lẽ đổi thành 0. Dùng số thập phân chính xác ở bước parse tiền; chỉ làm tròn ở bước rulebook yêu cầu.

### Tên không dấu chưa được liên kết với tên thật trong dữ liệu

`q_3aae102f4eb2`: tìm chính xác `salonpas cao dan hop 24 goi x 10 mieng` không có kết quả; sau đó tìm `%salonpas%` mới thấy tên có dấu và mã `SP0009`, nhưng gần hết 10 bước. Tương tự, câu công nợ dùng `To Dung Thao` làm điều kiện tên chính xác.

Đề xuất công cụ tìm đối tượng theo tên không dấu/viết tắt và trả về ứng viên cùng mã, tên gốc, nguồn. Khi tính thì dùng mã hoặc tên gốc đã xác nhận. Giữ số, size và thông tin quy cách để tránh nhập nhằng; không xóa dấu trong đáp án xuất ra. Hướng này có thể bắt đầu bằng tìm kiếm chuỗi, chưa cần thêm embedding model.

## 6. Luật nghiệp vụ chưa được bảo đảm bằng công cụ

### Lọc hủy/trả không theo nhãn thực tế

Model lặp điều kiện `NOT IN ('huỷ', 'trả hàng')`, nhưng dữ liệu có nhiều cách ghi khác.

| Câu | Dự đoán | Chuẩn | Kiểm chứng nguyên nhân |
|---|---:|---:|---|
| `q_489bb0e084b6`, số đơn Q3/2025 | 2.132 | 2.071 | 2.071 đơn `đã lấy` + 41 `Đã hủy đơn` + 20 `Trả hàng` |
| `q_93e78b3a3a98`, số đơn Q4/2025 | 2.160 | 2.101 | 2.101 đơn hợp lệ + 40 hủy + 19 trả |
| `q_b3b45f9adceb`, doanh thu Q3/2025 | 467.188.500 | 456.544.000 | Chênh 10.644.500 = 8.428.500 tiền `khách hủy` + 2.216.000 tiền `Hoàn trả` |
| `q_b0728619f5b4`, tác động tăng giá | 19,57% | 22,73% | SQL còn tính trạng thái `cancel`; số hợp lệ trước/sau là 44 và 54, nên tăng 22,73% |

Lỗi khác trong SQL quý là nối `OR` và `AND` không có ngoặc, khiến điều kiện trạng thái chỉ áp cho nhánh cuối. Cần chuẩn hóa trạng thái theo giá trị thực tế và rulebook, xử lý ở **cấp đơn**. Không chỉ lọc từng dòng khi luật yêu cầu loại cả đơn có trả một phần.

### Nhầm tổng phải trả với doanh thu dòng hàng

`q_a536e1622568` lấy `Phải thu` ở bảng đơn để tính tăng trưởng, ra **6,04%**. README xác định đây là tổng tiền đơn; rulebook train yêu cầu doanh thu là thành tiền dòng, không dùng các điều chỉnh ở cấp tổng đơn.

Kiểm chứng bằng join khóa đơn, parse đúng `Tiền bán` và chỉ lấy trạng thái hợp lệ:

- Tháng 4/2025: doanh thu dòng **1.123.026.000**.
- Tháng 5/2025: doanh thu dòng **1.177.286.000**.
- Tăng trưởng làm tròn: **4,83%**, khớp nhãn.

Đây là sửa đồng thời lựa chọn đại lượng, cách đọc tiền và trạng thái; không phải bằng chứng rằng chỉ thay tên cột sẽ sửa được toàn bộ câu doanh thu.

### Quy đổi sai cả dòng đã ở đơn vị cơ bản

`q_db64c1b7436d`: query join bảng quy đổi chỉ theo tên sản phẩm, rồi nhân mọi dòng Bánh croissant với 6. Trong kỳ có **87 cái + 1 hộp**, nghĩa là 93 cái; query cũ biến tổng lượng raw 88 thành **528** và xếp croissant đứng đầu.

Kiểm chứng giữ nguyên điều kiện kỳ và trạng thái trong query cũ, chỉ sửa join quy đổi thêm **đơn vị bán = đơn vị quy đổi**: kết quả đứng đầu là **Cà phê sữa (M)**, khớp nhãn. Các vấn đề trạng thái/dedup khác vẫn cần kiểm riêng, không suy ra toàn query đã hoàn toàn đúng chỉ từ việc khớp một đáp án.

### Công nợ và tồn kho cần nguồn đúng loại

Các câu công nợ có query bỏ ngày chốt, cộng chuỗi tiền, lấy dư nợ hiện tại cho một ngày quá khứ, hoặc không đối chiếu lịch sử trả tiền. Các câu tồn có trường hợp dùng tổng lượng bán làm lượng còn tồn. Cần công cụ có kiểm tra nguồn đầu kỳ, nhập/xuất/trả và ngày chốt; công nợ tính theo thanh toán đến ngày hỏi, quy tắc COD và trạng thái của split. Không suy tồn kho chỉ từ lịch sử bán.

## 7. Phân biệt thiếu dữ liệu với không có phát sinh

- Model dự đoán `unanswerable` **5 lần, cả 5 đều sai**.
- Trong **8 câu thực sự unanswerable**, model đúng 0: 4 câu trả số/list/percent sai, 4 câu bỏ qua sau hết bước.
- `q_ec5c0d7eed96`: hỏi số đơn hủy **vì giao trễ**, nhưng query chỉ kiểm tra chuỗi “giao trễ” trong trạng thái rồi trả 0. Các bảng của doanh nghiệp không có nguyên nhân hủy để chứng minh điều này.
- `q_2306395cb34a`: hỏi số **khách** có ít nhất 14 đơn, nhưng nguồn không có trường nhận diện khách; query lại đếm số dòng theo mã **đơn**.
- `q_151d139d77ca`, `q_8a8fd459a0a3`: model thấy top kết quả là các cách ghi khách lẻ rồi kết luận không có khách hợp lệ. Một query `LIMIT 1/3` với bộ lọc khách sai không chứng minh mọi dữ liệu đều không có khách.
- `q_c42078e25d28`: model tự thừa nhận không có tồn thực tế, nhưng vẫn dùng lượng bán và nộp `[1313,1313]` thay cho kết luận thiếu dữ liệu.

Runner hiện chỉ đòi có ít nhất một SQL thành công trước final `unanswerable`. Điều này không kiểm tra được SQL có hỏi đúng dữ liệu hay không.

Đề xuất bước kiểm tra khả năng trả lời trước tính toán: đại lượng cần cột nào, quan hệ nào, khoảng dữ liệu nào; có đủ hay thiếu thật? Một kết quả rỗng cần được đối chiếu với tình trạng parse, phạm vi ngày và tìm đối tượng. Chỉ xuất 0 khi đã xác nhận nguồn/kỳ phù hợp mà không có phát sinh. Chỉ xuất `unanswerable` khi có bằng chứng thiếu thông tin cần thiết; lỗi công cụ hoặc chưa xác minh thì `abstain`.

## 8. Tốc độ: đang trả Unit cho các vòng sửa không tiến triển

Phân loại theo lượt model, không phải các nhóm câu tách biệt:

| Lỗi sau khi sinh | Số lượt | Số câu gặp |
|---|---:|---:|
| Không đọc được action | 82 | 30 |
| Final bị từ chối | 69 | 40 |
| SQL báo lỗi | 11 | 5 |
| Tổng lượt bị từ chối | **162/500 — 32,4%** | Các nhóm câu chồng lấp |

Thời gian sinh của 162 lượt trên là **1.470,23 giây**, khoảng **24,5 phút**, tương đương **37,1% thời gian batch**. Đây là thời gian gắn với các lượt lỗi quan sát được, **không phải cam kết sẽ tiết kiệm toàn bộ sau khi sửa**: lượt mới vẫn có thể cần suy luận hoặc truy vấn bổ sung.

Tổng thời gian generate đo được khoảng 3.753,76 giây. Thời gian SQL thành công và schema query được ghi trong log cộng lại chỉ khoảng 8,42 giây, chưa gồm ingest và các SQL lỗi. Ưu tiên giảm lượt model lặp vô ích trước khi tối ưu tốc độ SQLite.

**21/21 câu abstain đều dùng hết 10 lượt model**, thời gian từng câu dưới trần 240 giây. Có 23 câu chạm 10 lượt, trong đó 2 vẫn trả đúng. Vì vậy tăng timeout đơn thuần không gỡ được trần bước; tăng max_steps mà không sửa phản hồi sẽ kéo dài những vòng lặp đã thấy. Ví dụ một câu tồn lặp **7 lần cùng SQL sai tên bảng**.

Phát lại SQL với chế độ không cho xem tên cột trong dấu nháy kép là chuỗi đã phát hiện **7 lượt trên 2 câu** dùng `Trạng thái` ở scope không còn cột này, nhưng lượt cũ không báo lỗi. Đề xuất tắt fallback đó nếu phiên bản runtime hỗ trợ, hoặc kiểm tra tên cột/scope trước khi chạy. Đây là phát hiện local, chưa thay cấu hình SQLite của agent.

Cờ `runtime_or_tool_error` trong summary khá rộng: 40 câu mang cờ này ở lượt này là các lỗi final/chiếu đáp án, không phải 40 lần GPU/runtime bị hỏng. Cần tách log `action_parse`, `sql_execution`, `answer_projection`, `answer_validation`, `context`, `time_budget`, `step_budget` để lần sau không chẩn đoán nhầm.

## 9. Phương án triển khai đề xuất để duyệt

Giữ kiến trúc **modular monolith**, dùng chung lõi cho CLI, web, Docker và notebook. Không cần tách microservice hoặc thêm model phụ ở bước này.

| Ưu tiên | Công việc cụ thể | Bằng chứng cần đạt trước khi tốn GPU |
|---|---|---|
| **1. Giao thức và chốt đáp án** | Hợp đồng action nhất quán; final tham chiếu query/cột; xử lý alias/kiểu có kiểm soát; ghi token và lý do dừng; phản hồi lỗi chính xác; chặn lặp cùng lỗi không thay đổi | Replay hai bảng công nợ lấy đúng từ SQL; test ngày/boolean/list/table; dữ liệu model tự bịa khác query vẫn bị chặn; output nhiều action không được chọn tùy ý |
| **2. Chuẩn hóa và liên kết dữ liệu** | Profile cột, chuẩn hóa ngày/tiền có suffix, ánh xạ nhãn trạng thái, khóa đơn/khách/sản phẩm, tìm tên không dấu; view có kiểu và nguồn gốc; xử lý ô trống/dedup theo rulebook | Kiểm tra các phép parse thực tế; đối chiếu raw với view, kiểm tra khóa và số dòng; các probe ngày, đơn hợp lệ, doanh thu dòng, quy đổi đơn vị khớp tính độc lập |
| **3. Kế hoạch tính và bằng chứng** | Công cụ cho doanh thu/số đơn/ranking/tăng trưởng/công nợ/tồn, hoặc sinh SQL từ kế hoạch có trường rõ; giải thời gian trước SQL; kiểm tra khả năng trả lời; dừng khi không có tiến triển | Phân biệt thiếu cột, ngoài phạm vi, không phát sinh; không tính lượng bán thành tồn; không lấy công nợ tương lai cho ngày quá khứ; SQL dùng cột/khóa không tồn tại phải lỗi rõ |
| **4. Đo lại rồi cân nhắc fine-tune** | Sau khi CPU qua, chạy cùng 72 câu một lần trên GPU, model/revision/quantization giữ như baseline; lưu nguồn mới và output riêng | Báo cáo chênh lệch từng câu, PA, lỗi action/final, false-unanswerable, số calls, latency và Unit ước tính; nêu cả câu bị giảm kết quả |

Các môđun có thể bổ sung bên trong package: `data/profiling`, `data/normalization`, `semantics/schema_binding`, `tools/metrics`, `agent/protocol`, `agent/verification`. Đây là hướng tổ chức trách nhiệm, không phải yêu cầu viết lại toàn bộ ứng dụng. Các sửa nhỏ ở hợp đồng final/SQL có thể làm trước và kiểm chứng bằng CPU.

Ánh xạ schema và luật phải dựa trên README, rulebook **của split đang chạy**, metadata và giá trị quan sát; giữ nguồn của quyết định và phát hiện trường mơ hồ. Không tạo bộ query/đáp án riêng theo question ID hoặc business ID. Các SQL gắn doanh nghiệp trong tệp audit chỉ là kiểm chứng lỗi, không được chép thành logic inference định tuyến theo ID.

Không thể cam kết mức accuracy sau sửa khi chưa chạy model lại. Các mốc kỹ thuật đề xuất cho lượt đo kế tiếp: giảm tỉ lệ action không đọc được từ 16,4% xuống dưới 1%; giảm final bị từ chối; giải quyết các lỗi đã xác minh mà không nới lỏng kiểm tra nguồn đáp án. **PA > 0 trên cùng 72 câu** có thể dùng làm mốc ban đầu, không phải tiêu chí đủ để nộp thi. Mỗi chỉ số và phạm vi thay đổi cần ghi cùng kết quả thực tế.

## 10. Dùng Unit cho fine-tuning khi nào?

**Khuyến nghị hiện tại: dành lượt tiếp theo để kiểm chứng bản sửa pipeline, chưa bắt đầu fine-tuning.** Các lỗi alias, kiểu SQL, ngày raw, locale tiền và trạng thái là vấn đề có thể sửa/kiểm bằng CPU. Train trên pipeline hiện tại có nguy cơ học cách bù lỗi thay vì giải đúng bài toán.

Không lấy máy móc 9 trace có đáp án đúng làm dữ liệu học. Có case `q_3919f65b83aa` lọc `LIKE '%/2026%' AND LIKE '%01%'`: điều kiện có thể nhận ngày 01 của các tháng khác, dù đáp án cuối trùng nhãn. Cần kiểm tra nghĩa SQL, không chỉ đáp án cuối.

Sau khi pipeline ổn, nếu còn nhiều lỗi lựa chọn đại lượng, hiểu viết tắt, chọn công cụ hoặc SQL thì mới cân nhắc LoRA/QLoRA trên trajectory development đã kiểm chứng; bổ sung ví dụ đối chiếu “thiếu dữ liệu / có dữ liệu nhưng bằng 0” và các kiểu câu không dấu. Huấn luyện chỉ định dạng đầu ra không tự dạy được phép join hoặc luật doanh thu. Giữ 6 doanh nghiệp holdout để đánh giá phương án đã chọn; không dùng chúng để tạo prompt, sửa quy tắc hay huấn luyện sau khi đã xem lỗi.

Lượt hiện tại chỉ có **Gemma single**. Chưa có so sánh Qwen, Gemma fine-tuned, assistant hay self-review. Assistant chính thức nhằm hỗ trợ sinh token nhanh hơn, không phải bằng chứng kiểm tra nghiệp vụ; bật assistant hoặc thêm lượt tự phản biện chưa giải quyết các lỗi dữ liệu/hợp đồng đã thấy. Không có lý do từ log này để kết luận cần GPU lớn hơn.

Để tiết kiệm Unit, có thể gộp các gói 1–3 đã qua kiểm tra CPU vào **một lượt chạy lại 72 câu**, thay vì bật GPU cho từng sửa nhỏ. Đổi tên lượt chạy và đóng gói source mới; giữ bundle baseline nguyên vẹn. Đo tốc độ lại trước khi dự toán chi phí fine-tune hoặc toàn private test, vì mẫu khó này không đại diện tải inference chính thức.

## Tệp kiểm chứng kèm theo

- `outputs/diagnosis-2026-11-03/audit_metrics.json`: tính lại điểm, kiểm tra fingerprint/hash, thống kê lỗi/thời gian.
- `outputs/diagnosis-2026-11-03/strict_sql_replay.json`: 226 SQL được kiểm chứng trên CPU.
- `outputs/diagnosis-2026-11-03/sql_probes.json`: truy vấn và kết quả kiểm chứng ngày, trạng thái, số tiền, doanh thu, đơn vị.
- `outputs/diagnosis-2026-11-03/projection_audit.json`: hai bảng công nợ khớp nhãn khi chiếu đúng kết quả SQL.
- `outputs/diagnosis-2026-11-03/case_events.json` và `trace-digest.txt`: sự kiện của toàn bộ 72 câu để tra cứu chi tiết.

Hai script audit trong thư mục output chỉ phục vụ tái lập phân tích trên CPU; không được import vào agent production. Các tệp prediction/log gốc không bị chỉnh sửa. Không có benchmark sau sửa trong báo cáo này.
