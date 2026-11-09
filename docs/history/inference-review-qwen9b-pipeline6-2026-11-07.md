# Phân tích Qwen 3.5 9B — shared-quality/pipeline 6, nhận ngày 07/11/2026

Nguồn người dùng cung cấp: `outputs/diagnosis-2026-11-07/qwen35-9b-random100-shared-quality-Nov7th-01/`. Đợt này chỉ đọc log, chấm lại, phân loại lỗi và probe công cụ trên CPU. Không sửa runtime/prompt/config/notebook/source ZIP hoặc gói SFT, không gọi model mới. Đây vẫn là cùng mẫu TRAIN đã được xem nhiều lần, không phải holdout độc lập hoặc điểm private test.

## Kết quả và mức độ thay đổi

| Chỉ số | Qwen pipeline 5, 06/11 | Qwen pipeline 6, lượt này |
|---|---:|---:|
| Đúng | 49 | 43 |
| Sai | 32 | 20 |
| Abstain | 19 | 37 |
| PA = (đúng − sai)/100 | 17% | 23% |
| Độ đúng trong các câu đã trả lời | 60,49% | 68,25% |
| Trung bình giây/câu | 9,64 | 10,68 |
| Trung vị giây/câu | 2,96 | 6,11 |
| P95 giây/câu | 30,85 | 30,01 |
| Tổng riêng 100 câu | 964,48 giây | 1.067,73 giây |
| Model calls | 194 | 311 |
| Tổng input token qua các calls | 2.057.036 | 3.287.033 |
| Tổng output token | 19.933 | 18.962 |

PA tăng 6 điểm phần trăm, nhưng số đúng giảm sáu. Không thể kết luận hệ thống hiểu câu hỏi tốt hơn tương ứng với mức tăng PA. Trong 49 câu từng đúng, chỉ 31 vẫn đúng: 13 chuyển thành abstain, năm chuyển thành sai. Có 12 câu mới đúng: mười từ nhóm sai và hai từ nhóm abstain. Trong 20 câu sai hiện tại có mười câu sai mới.

| Kết quả cũ → mới | Đúng | Sai | Abstain |
|---|---:|---:|---:|
| Đúng cũ | 31 | 5 | 13 |
| Sai cũ | 10 | 10 | 12 |
| Abstain cũ | 2 | 5 | 12 |

Thời gian trung bình tăng 10,7%, calls tăng 60,3%, input token tăng 59,8%; output token lại giảm 4,9%. Lượt kiểm tra/final và retry ngắn tăng chi phí đọc lại context, không phải do model sinh nhiều output hơn. Tổng thời gian các model calls tăng từ 588,66 lên 685,13 giây; phần thời gian còn lại tăng từ 375,82 lên 382,60 giây. Đây là phân rã theo timer, không phải phép đo riêng tốc độ giải mã GPU.

## Đã xác minh đây là lượt chạy đúng phiên bản

- Đủ 100 ID và đúng thứ tự mẫu cũ; câu hỏi, business và as_of_date khớp nguồn TRAIN đã duyệt. Chấm lại bằng evaluator local khớp summary: 43/20/37.
- Manifest đúng pipeline **6**, diagnostic **8**, backend protocol **3**, `mode=self_review`, `code_first`, 6 bước, 768 output token, context 16K, timeout 90 giây. Toàn bộ hash runtime trong manifest khớp source hiện tại; không có dấu hiệu dùng lại pipeline 4/5 trong RAM.
- Kiểm tra được cả **311 hash raw output** và **207 thành viên ZIP** khớp file; CRC hợp lệ. Checkpoint manifest SHA giống chính lượt Qwen trước.
- Cùng GPU RTX PRO 6000 Blackwell khoảng 95 GiB, BF16, trọng số trên cuda:0, greedy và thinking tắt. Torch đổi **2.11.0 → 2.14.1**, Transformers **5.22.0 → 5.22.1**. Vì prompt/tools/review và thư viện cùng thay đổi, đây không phải ablation chỉ đổi review.
- Cả 311 lần generation đều dừng EOS, token cuối **248046**; output dài nhất 508 token, input dài nhất 15.556 token. Không có context overflow hoặc câu chạm 90 giây; câu lâu nhất 51,36 giây. Có hai raw malformed: một action SQL/JSON và một đề xuất sửa schema. Chúng không phải tái phát lỗi thiếu stop token.
- Chỉ một model call sửa schema, 10,67 giây theo trace preparation, khoảng 1% tổng thời gian. Bước SLM mô tả toàn doanh nghiệp vẫn tắt.

## Điểm yếu lớn nhất: giao tiếp chốt đáp án làm mất kết quả đúng

Có **31 câu** gặp lỗi `Tên cột chiếu không tồn tại hoặc query có tên cột trùng`, tổng 60 lần báo lỗi. Mười câu tự phục hồi để trả đúng, 21 câu còn lại abstain. Nhóm 31 câu dùng 108 calls và 353,61 giây; con số này gồm cả các phép tính hợp lệ trong nhóm, không phải toàn bộ là thời gian lãng phí do lỗi cột.

Trong 21 câu bị kẹt, **15 candidate đã đúng**, sáu candidate còn sai nghĩa/kỳ/thực thể. Ví dụ `q_f47b3edba33e` tính đúng tăng trưởng tháng 4/2026 là **8,25%**. Code trả cột tên:

```text
round_half_up((current.value-previous.value)*100.0/NULLIF(previous.value,0),?)
```

Model sinh:

```json
{"action":"final","type":"percent","query_id":"r1","columns":["8.25"]}
```

`columns` phải là tên cột, nhưng model đưa giá trị. Nó lặp đúng action này rồi bị dừng. Các biến thể khác chép giá trị số, boolean hoặc tự thay `?` trong tên biểu thức bằng số; đều không khớp tên thực. Phép sum thường có alias `value`, còn growth/ratio/day_count/boolean có tên biểu thức SQL dài, khiến giao diện trả kết quả không thống nhất.

Đây là lỗi thiết kế giao tiếp trong bản self_review đã triển khai, không phải công cụ không tính được phần trăm. Nó giải thích phần lớn sự suy giảm của nhóm percent: **24 câu chỉ ba câu trả đúng, 21 abstain**; trong đó 18 bị kẹt projection. Câu boolean duy nhất cũng có candidate true đúng nhưng không final được vì dùng `columns=["true"]`.

Probe cơ học áp dụng cùng một quy tắc cho mọi câu phù hợp: với abstain có candidate, query đầy đủ một cột và lỗi projection, bỏ tham số columns rồi chiếu chính các dòng đã ghi. Kết quả replay là **58 đúng / 26 sai / 16 abstain, PA 32%**. Đây chỉ là counterfactual CPU trên kết quả cũ, **không phải benchmark runtime đã sửa**. Sửa projection cũng mở đường cho sáu candidate sai, nên không được chỉ cộng 15 ca đúng rồi bỏ qua rủi ro đó.

## Self-review hiện chưa kiểm tra được ý nghĩa

84 câu có candidate tính được; một câu khác có candidate unanswerable. Sau candidate tính toán, **mọi action tiếp theo đều là final**: 124 action gồm cả retry. Không có plan, lookup hay phép tính độc lập mới sau candidate; 27 câu thay plan đều sửa trước khi có candidate hợp lệ.

Trong 84 candidate: 58 đúng, 26 sai. Có 43 câu đúng được chốt và 15 đúng bị mất; 20 candidate sai được chốt và sáu sai vô tình bị chặn bởi projection. Không quan sát được trường hợp review sửa một candidate sai thành đáp án đúng. Dữ liệu này không chứng minh review luôn vô ích, nhưng không cho thấy lợi ích kiểm tra ngữ nghĩa ở lượt này. Không được quy toàn bộ biến động so lượt trước cho review vì nhiều thành phần cùng thay đổi.

Runner xác nhận tính đúng theo plan nhưng chưa chứng minh plan khớp câu hỏi. `evidence_id=Q` trỏ cả câu hỏi không chứng minh được filter “bia” là cách hiểu hợp lệ của chữ “gỏi”. Model có thể bỏ phủ định, bỏ sản phẩm, chọn sai đại lượng hoặc kỳ rồi đồng ý với chính kết quả của mình.

## Phân nhóm 20 câu sai

Phân loại dưới đây dùng nguyên nhân chính; một câu có thể mắc nhiều lỗi:

| Nhóm chính | Số câu | Ví dụ đã đối chiếu |
|---|---:|---|
| Sai đại lượng/phép toán/cấp tổng hợp/đầu ra | 7 | Hỏi lãi gộp nhưng cộng doanh thu; đếm đơn bỏ ngưỡng; gộp hai phương thức thay hai số; hỏi tháng lớn nhất nhưng trả ba tổng; hỏi hàng không bán nhưng SQL đếm hàng có bán. |
| Sai ca, kỳ hoặc năm | 6 | Tuần này thành tuần trước; năm 2025 thành 2026; so toàn timestamp với chuỗi giờ; khoảng đầu–cuối tháng thay ca từng ngày. |
| Thiếu/sai thực thể, nhóm, thuộc tính | 4 | Mất tên hàng chỉ giữ size M; hỏi gỏi chọn bia; bỏ điều kiện không kê đơn; hỏi nghêu hấp sả nhưng cộng toàn shop. |
| Tham số sai kiểu hoặc tên thay khóa lọt guard | 3 | prescription string "1" trả 0; hai khách nợ bị lọc bằng tên đặt vào customer_key nên trả [0,0]. |

Một số lỗi đáng chú ý:

- **Lãi gộp tháng 12/2025:** trả **305.793.500** là doanh thu, đúng phải **215.776.976**. Chỉ đổi measure sang gross_profit, công cụ tính được đúng.
- **Đơn đạt tiền hàng từ 93.000:** trả **4.210** đơn, bỏ điều kiện sau tổng hợp. order_count/threshold trả **1.067**.
- **Chân gà ngâm sả tắc bán bao nhiêu đĩa:** lookup đã xác định đúng sản phẩm nhưng measure vẫn revenue; trả **15.470.000** thay **182**.
- **Ca chiều tháng 10/2025:** so timestamp ISO cả ngày với `14:00:00` và `21:59:59`, không lấy phần giờ. Trả **379.300.800** thay **175.175.700**. Dùng shift/afternoon tính đúng. Một câu ca sáng khác dùng khoảng từ sáng ngày đầu tới trưa ngày cuối, nên lấy gần cả tháng chứ không phải ca sáng từng ngày.
- **Thuốc phải kê đơn:** facts lưu prescription kiểu integer, filter JSON lại là string `"1"`. Probe trên cùng bảng cho text param khớp **0** dòng, integer param khớp **10.688** dòng; với đúng kỳ, số tiền đúng **8.954.800**. Validator không chuẩn hóa kiểu và guard giá trị không khớp chưa bao phủ trường này.
- **Nợ của Bạch Hằng Long / Vũ Cường Quân:** khóa thực là **KH0170 / KH0147**, không phải tên. Nhánh debt_flow chạy trước guard lọc đối tượng dùng chung nên cho 0 thay vì yêu cầu lookup. entities.customer giải đúng khóa và tính được **[23.000,15.332] / [30.000,10.000]**.

Đã cấp plan/SQL sửa tay và chạy lại **20/20** câu sai bằng công cụ hiện tại, đều khớp tham chiếu. Điều này xác nhận đường tính cho các recipe đã kiểm tra, không xác nhận model sẽ tự tạo được chúng. Không dùng các probe cấp tay này làm trajectory mới của model hoặc cộng vào PA.

## Kỳ thời gian và luồng sửa plan còn dễ làm mất ý câu hỏi

Có trường hợp gợi ý kỳ ban đầu đã đúng nhưng plan vẫn chạy kỳ khác:

- `q_d767d5ca414e`: gợi ý đúng 27/01–25/02 so 28/12–26/01; model chọn cả tháng 1 so cả tháng 12.
- `q_b46a38609042`: gợi ý đúng tháng 1/2026 so tháng 1/2025; model so với tháng 12/2025.
- `q_cadff21c3eac`: gợi ý đúng 01–06/12; model chọn 24–30/11.
- `q_9d3c67eadb66`: gợi ý tháng 7/2025; model chuyển thành tháng 7/2026.

Gợi ý hiện là advisory; khi nhận plan, runner thay expected_periods bằng kỳ của plan. Chưa có đối chiếu đủ các mốc/đơn vị/thực thể bắt buộc với câu gốc.

Một vấn đề khác: this_month kèm month/year bị từ chối dù giá trị ghi lặp đang khớp. Trong hai câu hỏi cùng kỳ tới hôm nay, model “sửa” thành month rồi month_offset toàn tháng, làm mất month-to-date. Mục đích không bỏ qua field là đúng, nhưng feedback và cấu trúc API chưa giúp SLM sửa cú pháp mà giữ nguyên nghĩa.

## Vì sao còn 37 abstain

- **21 câu:** kẹt final.columns như trên; không được hiểu là đều thiếu khả năng tính.
- **16 câu:** sai nguồn/population, contract operation/period/entity, nguồn nhóm khách chưa phủ khách vô danh, SQL công nợ/lô chưa biểu diễn đúng, hoặc không tận dụng bằng chứng thiếu dữ liệu. Toàn bộ từng câu ở phụ lục.

Ba câu tham chiếu unanswerable đều abstain. Câu giới tính nữ đã có e0 missing gender nhưng model tiếp tục ratio trên sai nguồn; câu thiếu nguồn công nợ cũng có e0 rồi vẫn thử nhiều plan và một lượt sửa schema không hợp lệ. Riêng `q_e308f9743d56`, model cuối cùng chốt unanswerable hợp lệ ở **bước 6**, runner lại yêu cầu review khi **remaining_steps=0** nên trả abstain mặc định. Luồng review chưa dự trù ngân sách cho lượt cuối.

Hai câu giá chọn đúng value/price nhưng không điền entities.product, vẫn dùng filter tên theo cách cũ. Hai câu đếm đơn hủy dùng population sales đã loại hủy. Tổng dư nợ/số khách nợ tại thời điểm chưa có operation native tương ứng với debt_flow; model cần SQL số dư được kiểm chứng. Hạn lô ở nguồn nhập typed, không có cột expiry_date trên mọi fact; model vẫn bịa cột trên sai bảng.

30 abstain dừng vì lặp, bảy vì hết số bước; không có câu hết 90 giây. Nhóm complex do script gắn nhãn có 20 câu: hai đúng, hai sai, 16 abstain. Nhóm không dấu có 29 câu: bảy đúng, 11 sai, 11 abstain. Các nhóm là heuristic, có chồng lấp; không phải thí nghiệm chỉ thay dấu/độ phức tạp.

## Những phần đã tốt hơn và giới hạn của kiểm thử trước

Ngày serial/POS, tên danh mục và lineage CTE đã mở được các đường tính trước bị chặn. Ví dụ ba khách mua nhiều nhất trên nguồn POS (`q_2ea34572153a`) từ abstain thành đúng. Câu nhóm ống nước (`q_717e5f91d40b`) cũng trả đúng tỷ lệ. Một số nhóm hàng, ngưỡng ngày và tính tăng liên tục đã cho candidate đúng.

Nhưng truy vấn chạy được không đồng nghĩa đúng nghĩa: câu hàng ế nay qua CTE guard nhưng SQL dùng tập hàng có bán, trả 55 thay vì năm hàng không bán. Lỗi chuyển từ kỹ thuật sang suy luận cần được phân biệt.

213 software tests và các phép CPU trước xác nhận các công cụ/luồng với action cấp trước. Chúng chưa phủ việc SLM tự điền final.columns sai, lọc boolean bằng chuỗi hoặc đặt tên vào customer_key của nhánh ledger. Audit context chỉ chứng minh token fit, không chứng minh model sẽ dùng đúng contract. Đây là khoảng trống kiểm thử trong bản đã triển khai, không phải bằng chứng mới rằng code cũ hoặc checkpoint khác đã được chạy.

## Thứ tự xử lý đề xuất

1. **Sửa giao tiếp kết quả trước.** Native output dùng tên cột ổn định; kết quả một cột đã có type/shape do code xác nhận thì final chỉ cần query_id hoặc một quyết định accept/revise. Không bắt SLM chép tên biểu thức SQL/giá trị vào columns. Ghi feedback sửa đúng field và tên cột được phép khi thật sự cần chọn cột. Replay toàn bộ raw final lỗi để kiểm tra, không chỉ sum/value.
2. **Đóng lỗ hổng tham số và zero sai.** Kiểm tra/coerce có căn cứ kiểu bool/number từ schema/domain; không đổi ID text tùy tiện. Mọi nhánh, gồm ledger, phải kiểm tra khóa tồn tại và entity không khớp trước khi COALESCE về 0. Phân biệt khách có thật không phát sinh với tên bị nhét sai vào key.
3. **Giảm gánh nặng API và giữ nghĩa lúc sửa.** Chỉ yêu cầu field cần cho từng operation; code giải source, key và projection khi có thông tin đủ. Tham số ghi lặp có thể được kiểm tra như assertion rồi chuẩn hóa khi thật sự khớp. Feedback nói cách giữ this_month, dùng shift thay khoảng timestamp và dùng entities thay key tên; không chỉ liệt kê lỗi enum. Giữ đường unanswerable theo bằng chứng, và dành bước cho review khi review là bắt buộc.
4. **Kiểm tra đủ mục tiêu câu hỏi trước tính/chốt.** Biểu diễn đại lượng, phủ định, thực thể, nhóm, đơn vị, mốc so, grain và số kết quả cùng đoạn bằng chứng cụ thể; đối chiếu với plan. Không dùng Q trỏ toàn câu như chứng minh ngữ nghĩa. Phép tổng hợp theo tháng/argmax, số dư nợ và anti-join hàng không bán nên được thể hiện bằng phép dùng chung hoặc SQL đã kiểm chứng, không tạo metric theo shop/ID.
5. **Đo lại review bằng ablation.** Sau khi giao tiếp và guard đúng, so single/self_review có chọn lọc trên cùng source, checkpoint, thư viện và mẫu. Review cần nhiệm vụ tìm sai khác cụ thể, không chỉ xác nhận candidate. Theo dõi cả wrong, lost-correct, calls và latency. Bổ sung ca TRAIN chưa dùng chẩn đoán/holdout theo doanh nghiệp; không chọn hướng sửa chỉ từ PA trên 100 câu đã xem nhiều lần.

Fine-tune có thể hỗ trợ cách điền intent/plan và quyết định sửa khi dữ liệu được curate bằng contract đã chốt. Nó không thay sửa lỗi projection/kiểu dữ liệu/khóa. Bộ Gemma SFT frozen hiện không bị thay trong phân tích này.

## Bằng chứng để kiểm tra lại

Tại `outputs/diagnosis-2026-11-07/qwen9b-shared-quality-review/`:

- `audit.json`: chấm lại, hash/raw/ZIP, cấu hình, timing và bảng chuyển kết quả.
- `questions.json`, `wrong.json`, `abstain.json`: toàn bộ plan, action, lỗi và candidate được trích từ log.
- `annotations.md` / `annotations.json`: chẩn đoán từng **20 sai + 37 abstain**.
- `manual-probes.json`: 20 recipe sửa tay, SQL, tham số và kết quả đối chiếu.
- `projection-replay.json`: quy tắc sửa cơ học không chọn theo nhãn, 21 case và kết quả counterfactual có ghi rõ phạm vi.
- `typed-filter-and-key-probes.json`, `period-hints-vs-execution.json`, `findings.json`: bằng chứng kiểu tham số, khóa, mốc thời gian và review.

Source ZIP inference hiện vẫn SHA256 `38407bd0bbdf97489f3de59d11c88c1f604a061f81b30914e8bba43ddbfbed64`. Hai ZIP seed/source Gemma frozen giữ nguyên hash đã bàn giao. Chưa sửa runtime hoặc đo PA sau sửa trong đợt phân tích này.
