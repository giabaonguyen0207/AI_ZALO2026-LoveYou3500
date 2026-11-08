# Qwen 3.5 9B · 100 câu sau sửa protocol · 06/11/2026

Nguồn: `outputs/diagnosis-2026-11-06/qwen35-9b-random100-protocol-fix-Nov6th-02/`.

**Lượt này chạy đúng bản sửa và hoàn tất đủ 100 câu: 49 đúng, 32 sai, 19 abstain, PA 17%. Tốc độ cải thiện rõ: 9,64 giây/câu; chất lượng còn thấp vì nhiều plan sai ý nghĩa vẫn qua kiểm tra và được tự chốt.** Không còn tình trạng mọi câu bị lỗi EOS/parser như các lượt Qwen trước.

## 1. Kết quả đã kiểm tra lại

| Chỉ số trên đúng cùng 100 ID/thứ tự | Qwen 9B protocol-fix | Gemma fast v5 | Gemma baseline v3 |
|---|---:|---:|---:|
| Đúng | **49** | 46 | 49 |
| Sai | **32** | 16 | 13 |
| Abstain | **19** | 38 | 38 |
| PA = (đúng − sai)/100 | **17%** | 30% | 36% |
| Tỷ lệ câu có đáp án | 81% | 62% | 62% |
| Đúng trong các câu đã trả lời | **60,49%** | 74,19% | 79,03% |
| Tổng thời gian các câu | **964,48 giây** | 2.318,89 giây | 1.825,54 giây |
| Trung bình | **9,64 giây** | 23,19 giây | 18,26 giây |
| Trung vị | **2,96 giây** | 17,95 giây | 12,33 giây |
| P95 | **30,85 giây** | 64,10 giây | 46,87 giây |
| Model calls, gồm sửa schema | **194** | 256 | 351 |
| Token đầu ra | **19.933** | 32.769 | 18.791 |
| Thời gian generation calls | **588,66 giây** | 1.940,91 giây | 1.459,81 giây |
| Thời gian còn lại | **375,82 giây** | 377,98 giây | 365,73 giây |

Qwen nhanh hơn Gemma fast v5 khoảng **58,4%** trong batch này. Cả ba số thời gian đều không gồm load model. Thời gian còn lại gồm đọc/chuẩn bị dữ liệu, tokenize, điều phối và I/O; log chưa tách đầy đủ từng phần.

Đây là so sánh các pipeline đã chạy, **không phải phép đo chỉ thay duy nhất model**: bản Qwen có prompt/validator/protocol mới, còn baseline v3 khác cả cơ chế chốt và ngân sách bước. GPU và software versions ghi trong các lượt Qwen/Gemma fast là cùng loại; prompt mới cũng dài hơn. Không dùng bảng này để khẳng định mức chênh lệch năng lực thuần của hai checkpoint.

So với Gemma fast v5, Qwen giữ đúng 37 câu, làm đúng thêm 12 câu, mất 9 câu từng đúng (8 sai, 1 abstain). Trong 38 câu Gemma từng abstain, Qwen trả lời 21 câu: **6 đúng, 15 sai**. Coverage tăng nhưng phần trả lời thêm gây mất điểm. Qwen vẫn có 49 câu đúng, nên kết luận “Qwen không chạy được” không còn phù hợp; pipeline Qwen hiện chốt quá nhiều đáp án sai.

Riêng bảy câu đầu: **6 đúng, 1 sai**, 83,51 giây và 7 calls. Mẫu bảy câu đủ kiểm tra giao tiếp nhưng đã cho cảm giác chất lượng tốt hơn kết quả toàn 100; không dùng nó thay benchmark đủ độ đa dạng.

| Type tham chiếu | Số câu | Đúng | Sai | Abstain |
|---|---:|---:|---:|---:|
| number | 54 | 33 | 14 | 7 |
| percent | 24 | 9 | 8 | 7 |
| list | 10 | 2 | 6 | 2 |
| date | 8 | 4 | 2 | 2 |
| unanswerable | 3 | 1 | 2 | 0 |
| boolean | 1 | 0 | 0 | 1 |

Nhóm không dấu có 29 câu: 10 đúng, 15 sai, 4 abstain, PA **−17,24%**. Nhóm gợi ý phức tạp có 20 câu: 5 đúng, 9 sai, 6 abstain, PA **−20%**. Các nhóm ngôn ngữ chồng nhau, không cộng thành hai tập độc lập. Câu nhiều thành phần và câu không dấu là ưu tiên kiểm tra giữ đầy đủ điều kiện; không chỉ ưu tiên làm JSON hợp lệ.

## 2. Xác nhận bản sửa EOS/protocol đã thực sự được dùng

- Manifest: pipeline **5**, diagnostic **7**; runtime: backend protocol **3**, effective EOS **[248044, 248046]**.
- 192 calls dừng ở EOS đều có last token **248046**, đúng token kết thúc lượt. Một call chạm 768 token; một call thất bại do context trước khi có output.
- 193 raw outputs được kiểm tra SHA256; có **một lỗi parse trong đường trả lời**, do JSON liệt kê nhiều ngày rồi bị cắt ở 768 token. Hai proposal sửa schema khác cũng không parse được; đây là ba output không hợp lệ, không phải lỗi hai marker cuối cũ.
- Chạy BF16, `quantization=none`, greedy, `enable_thinking=False`; tham số trên `cuda:0`, GPU RTX PRO 6000 Blackwell khoảng 94,97 GiB. Không thấy OOM/CPU offload trong bundle.
- Chuẩn bị/sửa schema bằng model chỉ 2 calls, **16,52 giây**, một cache hit; không phải nút thắt thời gian chính.

Đã kiểm tra fingerprint TRAIN được đăng ký, identity manifest, đúng nội dung câu hỏi/as_of_date, submission khớp logs, chấm lại nhãn local, CRC ZIP và 207 member khớp file xuất. Tất cả source hash runtime được manifest ghi nhận khớp source local hiện tại. Bundle chỉ cung cấp digest manifest model `f8b32a…`, không đính kèm chính manifest trọng số để kiểm tra độc lập toàn bộ weight hashes; không coi digest đó là bằng chứng đã so từng file trọng số với local.

## 3. Vì sao 32 câu sai?

Nhóm dưới đây dùng nguyên nhân chính để đếm không trùng; một câu có thể còn lỗi phụ.

| Nhóm | Số câu sai |
|---|---:|
| Sai phép tính, đại lượng hoặc cấp tổng hợp | **9** |
| Sai thực thể, nhóm hàng, đơn vị hoặc điều kiện | **9** |
| Sai kỳ thời gian | **5** |
| Sai dạng nhiều kết quả/thứ tự | **3** |
| Hiểu sai tồn kho/công nợ | **3** |
| Thiếu thông tin nhưng vẫn tính từ thuộc tính thay thế | **2** |
| Đúng giá trị ngày nhưng sai type | **1** |
| Tổng | **32** |

### 3.1. Validator còn chấp nhận trường mà compiler không thực thi

Đây là lỗi thiết kế hệ thống cần sửa trước khi quy toàn bộ lỗi cho SLM:

- `ratio` nhận hai kỳ nhưng chỉ tính kỳ đầu. Câu hiệu ứng tăng giá Lavie dùng ratio, lặp bộ lọc sản phẩm ở tử, nhận **100%** thay vì tăng trưởng **104,95%**. Kỳ thứ hai có trong log nhưng không được dùng.
- `sum` nhận `group_by/display/limit`, nhưng vẫn tính tổng. Câu top 5 sản phẩm trả **5.165**; câu trung bình/đơn trả **141.700.100** thay vì **72.444**.
- `numerator_filters` ngoài ratio không tạo đúng phép lọc mong muốn. Câu đếm đơn tiền hàng từ 93.000 đồng trở lên bỏ luôn ngưỡng tiền và trả **4.314** thay vì **1.067**.

Kiểm tra tĩnh các field này trên action đã được chốt đánh dấu **8 câu: 6 sai, 2 đúng**. Không suy rằng cứ chặn cả tám là điểm sẽ tăng: hai câu đúng có field dư vô ích. Hướng sửa là contract theo từng operation, báo rõ trường nào không được thực thi và cho model sửa, rồi đo lại cả hồi quy đúng và sai.

### 3.2. Có schema và SQL hợp lệ chưa chứng minh đúng câu hỏi

**79 câu được auto-final: 48 đúng, 31 sai.** Câu sai còn lại dùng final tường minh. Chỉ 60,76% auto-final đúng; đây là nơi cần ưu tiên kiểm soát.

`evidence_id="Q"` chỉ gắn lại nguyên văn câu hỏi. Nó không chứng minh model giữ đủ điều kiện hoặc chọn đúng đại lượng. Cờ `verified` của query biên dịch cho biết query đúng theo plan và kiểm tra nguồn; một plan sai vẫn tạo được query hợp lệ.

Ví dụ rõ:

- Hỏi **182 đĩa Chân gà ngâm sả tắc**, plan lại lấy toàn bộ **1.096.349.500 đồng** của shop.
- Hỏi **149 ly Trà sữa ô long (M)**, plan dùng revenue và mọi size M, trả **125.322.000 đồng**.
- Hỏi hai doanh thu COD và tiền mặt, plan bắt một đơn có payment_method bằng cả hai giá trị, chốt **0** thay vì list hai số.
- Hỏi ngày cao nhất, một câu tìm đúng `2025-05-11` nhưng rank xuất type string; reference là date nên vẫn sai.

Đối với SQL custom, guard có thể chỉ xác nhận nguồn/cột, không kiểm chứng toàn bộ công thức. Hai lỗi công nợ cho thấy rõ giới hạn này: “thu nợ” bị thay bằng dư nợ snapshot; “nợ thêm/trả được” bị thay bằng tổng tiền đơn một khách và chỉ trả một số.

### 3.3. Neo thời gian sai và dùng dữ liệu sau as_of_date

Có 5 lỗi thời gian chính. Các câu hỏi tháng cụ thể thường bị so với `previous_month_to_date`, vốn neo theo **as_of_date**, thay vì tháng trước của **tháng được hỏi**.

Ví dụ hỏi tháng 6/2026 khi as_of_date là 03/09/2026: model so tháng 6 với **01–03/08**, trả **762,19%** thay vì so tháng 5 và ra **−9,65%**. Câu “từ đầu tháng tới hôm nay” vào 29/05 lại lấy tháng đến 31/05, trả **10,97%** thay vì **2,83%**. “Bữa qua” bị lấy thành chính ngày hôm nay.

Các kỳ gợi ý trong request_contract hiện chỉ là advisory. Cần giữ rõ trong plan kỳ nào neo theo ngày hỏi, kỳ nào neo theo kỳ được nhắc; code xử lý lịch và quan hệ kỳ sau khi model xác định căn cứ.

### 3.4. Khớp tên, nhóm và đơn vị chưa được giữ đầy đủ

- Tên không dấu `thach ca phe`, `ngheu hap sa` không khớp tên có dấu, rồi SUM/COALESCE biến thành **0**. Lookup hiện có tìm không dấu; model chưa dùng nhất quán.
- “Nhóm chiên xào”, “nhóm cơm mì cháo” bị đổi thành AND nhiều từ trên product_name. Cột nhóm thật có trong danh mục.
- Câu hỏi Cappuccino (M) chỉ lọc hậu tố M nên tính nhiều sản phẩm; câu hỏi số ly không lọc base_unit=ly nên cộng thêm đơn vị khác.
- Câu nhóm trà sữa bị bỏ toàn bộ nhóm. Nguồn này có danh mục theo mã nhưng dòng bán **không có product_id**. Nối ID trực tiếp bị guard chặn là hợp lý. Probe nối tên chuẩn hóa duy nhất sang danh mục mới tính đúng **65.734.000**, thay vì tổng shop **302.714.000**. Chỉ tìm chữ trà sữa trong tên không bao phủ đúng nhóm.

Không sửa bằng cách chặn mọi kết quả 0, hoặc tự bỏ dấu/fuzzy-match mọi chuỗi. Cần biết trước bộ lọc đã khớp đúng bao nhiêu thực thể, có mơ hồ hay nhân bản join không; số 0 sau khi khớp đúng vẫn có thể hợp lệ.

### 3.5. Ngưỡng theo ngày/đơn bị áp sai cấp dữ liệu

Câu đếm ngày đạt 8,6 triệu đã dùng đúng day_count nhưng còn lọc từng **dòng bán** đạt 8,6 triệu trước khi cộng theo ngày, trả 0 thay vì 7. Câu 11 triệu dùng count các dòng bán lớn, trả 0 thay vì 14 ngày.

Ngưỡng của tổng theo ngày/đơn cần áp **sau khi tổng hợp theo ngày/đơn**, tương đương HAVING, không phải WHERE lên tiền từng dòng. Schema của plan cần biểu diễn rõ cấp tổng hợp và cấp áp điều kiện.

### 3.6. Thiếu thuộc tính bị thay bằng proxy không có căn cứ

Nguồn câu tỷ trọng khách nữ không có trường giới tính; model tìm `nu` trong customer_name rồi trả 0%. Nguồn câu so lý do trả hàng không ghi lý do; model đếm đơn completed theo channel. Cả hai phải unanswerable theo dữ liệu hiện có.

`check()` hiện chưa bao phủ hai nhu cầu này; khi có plan, nó chủ yếu lấy requirements do model khai báo. Model bỏ requirements thì kiểm tra đầu vào cũng không nhận ra phần cần thiết bị thiếu. Không suy giới tính từ tên hoặc lý do từ trạng thái; cần thuộc tính và nguồn có bằng chứng.

## 4. Vì sao còn 19 abstain?

Chúng dùng **90/194 calls**, mất **368,19 giây**, khoảng **38,2%** thời gian toàn batch. Lỗi chia thành schema/chuẩn hóa chưa đủ, plan sai hoặc chưa đúng giao diện công cụ, và guard SQL chặn nhầm.

**Ngày serial chưa được parser hỗ trợ:** `biz_111` có 16.594 đơn, POS `biz_123` có 8.437 đơn với ngày chuẩn hóa NULL. Raw có giá trị như `45658`; ở POS, trường thời gian cùng dòng là `2025-01-01 07:28:15`, khớp cách hiểu serial Excel. Cả `iso_date("45658")` và `iso_date(45658)` hiện đều lỗi. Các bảng này có dữ liệu, không phải không thể trả lời vì thiếu ngày gốc. Guard ngừng cộng thiếu dòng là cần thiết; phải sửa parser.

Excel có hai hệ ngày 1900/1904, nên bổ sung chuyển đổi cần dựa vào kiểu cột, epoch workbook hoặc đối chiếu nguồn, không chuyển mọi chuỗi số thành ngày. Xem [Microsoft: hệ ngày và serial trong Excel](https://support.microsoft.com/en-us/excel/date-systems-in-excel). Đây là hướng xử lý định dạng dữ liệu dùng chung, không gắn theo business ID.

**Schema POS chưa ánh xạ đủ:** `TongThanhToan`, `NgayGioThu`, `TenGoi`, `LoaiKhachHang`, `HanSuDung` có trong nguồn nhưng còn thiếu vai trò/quan hệ tương ứng. Các câu nợ quá khứ, khách nhóm sỉ và hạn lô chịu ảnh hưởng. Với tồn theo thẻ kho, từng dòng là trạng thái sau một giao dịch, phải lấy trạng thái cuối đúng thời điểm; không cộng cả bảng như snapshot phẳng.

**Công cụ có nhưng model không gọi đúng:** probe CPU gọi `continuous_growth` trả boolean true, `inventory` trả 3 hộp, `shift_growth` trả 40,45%. Hai câu khác chỉ sửa đơn vị phần hoặc dùng lookup tới tên thuốc đầy đủ đã tính được 12,5% và 123,46%. Đây là các probe gọi tay; không phải kết quả model mới.

**Lineage SQL còn chặn nhầm:** `WITH t AS (SELECT COUNT(*) FROM dm_orders_all) ...` thực sự đọc dữ liệu nhưng `direct_tables=[]`, bị báo như SELECT hằng số. Query subquery tương đương trên cùng bảng có cùng số đếm và qua guard. Với view dm_sales còn có trường hợp subquery cũng mất direct_tables. Cần giữ truy vết đọc ở mọi ngữ cảnh CTE/view/subquery thay vì chỉ xét một danh sách bảng trực tiếp. Xem [SQLite: authorizer callback và ngữ cảnh view/trigger](https://www.sqlite.org/c3ref/set_authorizer.html).

**Không có câu nào hết 90 giây**; câu lâu nhất 65,66 giây. Flag `step_or_time_budget=16` đang gồm 10 lý do dừng lặp có chữ “ngân sách” và 6 câu hết số bước. Không diễn giải nó thành 16 timeout. Có một lỗi context sau nhiều lượt schema/SQL, khác với chậm GPU.

## 5. Thứ tự xử lý đề xuất

1. **Sửa contract và điều kiện auto-final trước.** Mỗi operation có số kỳ, field được hỗ trợ và shape/type rõ; không âm thầm bỏ group_by, limit, numerator_filters hay kỳ thứ hai. Phát hiện các điều kiện EQ loại trừ nhau trên cùng cột. Kiểm tra plan giữ thực thể, đơn vị, số kết quả, ngưỡng và cấp tổng hợp đã xác định từ câu hỏi. Vi phạm thì feedback ngắn đúng trường, chỉ sửa có giới hạn hoặc abstain.
2. **Sửa chuẩn hóa ngày và bổ sung binding có căn cứ.** Nhận serial số/số dạng text trong cột ngày đã xác định, kiểm chứng epoch; ánh xạ các vai trò POS dựa trên README/profile/quan hệ khóa, kể cả ngày thu và tổng phải trả. Không tắt guard NULL/ngày chưa giải để cộng tổng một phần. Cache theo nguồn và phiên bản xử lý như hiện tại.
3. **Giải thực thể/nhóm/đơn vị trước tính.** Dùng lookup/key duy nhất; nhóm từ cột danh mục, nối bằng ID hoặc khóa tên đã đối chiếu khi không có ID; kiểm tra khóa đích duy nhất. Giữ rõ filter của shop và filter riêng tử số/nhánh series. Bổ sung bằng chứng thuộc tính cần thiết, không dùng tên/thành phần chữ làm proxy cho giới tính, nhóm hay lý do.
4. **Hoàn thiện phép toán dùng chung và định tuyến công cụ.** Thể hiện rõ WHERE/HAVING và cấp dòng/đơn/ngày, average order value, chuỗi kết quả có thứ tự, stock quotient/remainder, nợ thêm/trả nợ theo ledger. Tận dụng inventory/shift_growth/continuous_growth đã có. Không tạo metric mới theo từng doanh nghiệp hoặc ID; tham số, schema mapping và rulebook của split quyết định phép tính.
5. **Sửa lineage CTE/view/subquery, rồi thử cơ chế chốt có chọn lọc.** Giữ kiểm tra SQL chỉ đọc/nguồn thật và dữ liệu chất lượng. Có thể chạy một ablation chỉ đổi `auto_execute_plan=False` trên cùng mẫu để đo tác động của lượt model đọc kết quả trước final; chưa có bằng chứng tắt auto-final hoặc bật self-review cho tất cả sẽ tăng PA. Chỉ thêm kiểm tra SLM có chọn lọc sau khi công cụ đúng và đo chi phí. Không cần quay lại chuẩn bị schema toàn doanh nghiệp bằng SLM.
6. **Đo lại có kiểm soát.** Cố định code sau sửa, chạy cùng 100 làm hồi quy, thêm câu train chưa dùng chẩn đoán và bộ giả đổi tên file/cột/nhãn. So Qwen/Gemma trên cùng prompt/tools/budget mới để tách ảnh hưởng model; theo dõi wrong, PA và thời gian, không chỉ số câu hết abstain. Chưa cần tăng model 14B hoặc fine-tune để che lỗi parser/binding/compiler. Bộ Gemma SFT hiện giữ nguyên; nếu tạo trajectory sau này cần replay bằng contract/tool đã sửa rồi kiểm chứng.

Các đề xuất trên có nhánh kiểm tra API, kiểu dữ liệu, cấu trúc phép toán và tính nhất quán nguồn. Chúng không phải rule “gặp câu/ID/doanh nghiệp này thì trả đáp án kia”. Với doanh nghiệp mới, cần ánh xạ schema/thuộc tính và policy có căn cứ; không mặc định một metric hay nhãn cố định cho mọi split. Code vẫn không thể chứng minh tuyệt đối mọi diễn giải tiếng Việt của SLM.

## 6. Bằng chứng CPU và phạm vi kết luận

Đã tái hiện từng nhóm sai bằng kế hoạch/công cụ/SQL sửa tay: 29 biến thể đi qua đường tính/chiếu tương ứng khớp tham chiếu; một câu nhóm trà sữa cần nối danh mục theo tên vì không có ID, query kiểm tra riêng cho kết quả đúng; hai câu thiếu thuộc tính được kiểm tra schema/profile. Có thêm năm probe công cụ/điều kiện cho các câu abstain. Các query, tham số, output, lỗi và đối chiếu đều được lưu.

**Không quy các kết quả sửa tay này thành PA mới hoặc số câu Qwen chắc chắn sẽ sửa được.** Chưa sửa runtime/prompt/notebook hoặc source ZIP trong đợt phân tích này, chưa inference thêm hoặc fine-tune. Toàn bộ 100 thuộc TRAIN đã dùng nhiều lần phát triển; kết quả không phải holdout độc lập hay dự báo private test.

Artifact: [audit và so sánh](../outputs/diagnosis-2026-11-06/qwen9b-full-review/audit.json), [probe từng câu sai](../outputs/diagnosis-2026-11-06/qwen9b-full-review/probes.json), [kiểm tra nguồn/ngày/CTE](../outputs/diagnosis-2026-11-06/qwen9b-full-review/additional-probes.json), [nguyên nhân từng câu dạng JSON](../outputs/diagnosis-2026-11-06/qwen9b-full-review/wrong-causes.json). Các script tái hiện nằm cùng thư mục; verification.json ghi kiểm tra cuối và hash artifact.

## 7. Toàn bộ 32 câu sai

Số # là vị trí trong mẫu 100, ID chỉ dùng nối log chẩn đoán. Giá trị list dài được rút gọn tại bảng; JSON/probes giữ đầy đủ tên và thứ tự. `unanswerable` không phải 0.

| # mẫu / ID | Câu hỏi rút gọn | Model → tham chiếu | Nguyên nhân và hướng xử lý |
|---|---|---|---|
| 7 · `q_d767d5ca414e` | xem giup nuoc suoi lavie 1.5L doi gia tu 10.000đ len 10.500đ vao ngay 27/01/2026. … | percent: 100.0 → percent: 104.95 | Dùng ratio cho tăng trưởng hai kỳ; compiler ratio chỉ đọc kỳ đầu, lặp filter sản phẩm ở tử nên ra 100%. **Hướng xử lý:** Dùng growth với hai kỳ và một phạm vi sản phẩm; ratio phải từ chối hai kỳ thay vì bỏ kỳ thứ hai. |
| 9 · `q_1295f18bd2fe` | cho toi biet doanh thu bua qua | number: 36214000 → number: 29235200 | “Bữa qua” bị hiểu thành ngày as_of_date 07/10 thay vì 06/10. **Hướng xử lý:** Làm rõ cách hiểu cụm thời gian trước tính; dùng yesterday theo as_of_date. |
| 12 · `q_b602a57b38a4` | Tháng 7/2025 nhóm chiên xào chiếm bao nhiêu % doanh thu? | percent: 0.0 → percent: 17.4 | Tìm sản phẩm đồng thời kết thúc bằng (chiên) VÀ (xào); thay cột nhóm bằng hai điều kiện tên không đúng. **Hướng xử lý:** Nối danh mục, lọc nhóm chiên xào ở tử; mẫu số giữ doanh thu cả shop. |
| 13 · `q_ece3c46e24d3` | dthu tu dau thang nay den het hom nay tang hay giam bao nhieu % so voi cung ky tha… | percent: 10.97 → percent: 2.83 | Kỳ hiện tại lấy cả tháng 5 đến 31/05, dù as_of_date và câu hỏi chỉ đến 29/05. **Hướng xử lý:** Dùng this_month và previous_month_to_date, giữ cùng số ngày theo rulebook. |
| 14 · `q_f47b3edba33e` | So với tháng liền trước, doanh thu tháng 4 năm 2026 thay đổi bao nhiêu %? | percent: 3.91 → percent: 8.25 | So tháng 4/2026 với 01–29/04 theo as_of_date tháng 5; kỳ gốc phải là tháng 3/2026. **Hướng xử lý:** Neo tháng trước vào tháng được hỏi, không neo vào as_of_date cho câu hỏi tháng cụ thể. |
| 18 · `q_9f82b2f25fd6` | trong thang 5/2026, khach nu chiem bnhiu % doanh thu | percent: 0.0 → unanswerable: unanswerable | Tìm chữ nu trong tên khách để suy giới tính; nguồn không có trường giới tính. **Hướng xử lý:** Yêu cầu thuộc tính giới tính có căn cứ; thiếu thực sự thì unanswerable, không dùng tên khách làm proxy. |
| 19 · `q_41c306df830d` | trong thang 6/2025, bao nhiu don co tien hang dat 93.000 dong tro len vay | number: 4314 → number: 1067 | Bỏ ngưỡng 93.000 đồng, dùng mọi trạng thái; numerator_filters không được count áp dụng. **Hướng xử lý:** Cộng thành tiền từng đơn hợp lệ, HAVING tổng >= 93.000 rồi đếm; từ chối trường không dùng. |
| 20 · `q_72f43cc7a0bc` | ngayf nào trong tháng 4/2025 dthu cao nhất | number: 1488698100 → date: "2025-04-04" | Plan đầu liệt kê từng ngày thành filter AND và bị cắt 768 token; plan sửa vẫn dùng tổng tháng, không tìm ngày cao nhất. **Hướng xử lý:** Giữ mục tiêu/type qua retry; dùng best_day/date, không liệt kê mỗi ngày thành filter AND. |
| 23 · `q_3e69edbe4dc2` | cuoi ngay hom nay da 4x6 con may xe 4m3 chan, may m3 le | number: 0 → list: [9, 3] | Câu tồn Đá 4x6 và chia xe/lẻ bị giải thành doanh thu hôm nay; hai filter tên loại trừ nhau. **Hướng xử lý:** Đọc snapshot đúng ngày và hệ số xe 4m3; inventory_units hiện có trả [9,3]. |
| 24 · `q_6e13d0978b21` | tháng này khách bạch hằng long ghi nợ thêm bao nhiêu và trả nợ đươc bao nhiêu? ghi… | number: 1383200 → list: [23000, 15332] | Câu hỏi hai dòng tiền được trả thành tổng giá trị đơn một khách, bỏ cả ghi nợ và các lần trả sau ngày mua. **Hướng xử lý:** Theo rulebook: nợ thêm = chưa trả tại ngày mua; trả nợ = phiếu thu sau ngày mua trong kỳ; trả list đúng thứ tự. |
| 32 · `q_27eebcccdf1f` | coi gium dthu thang 6/2026 tang hay giam bao nhieu % so vs th truoc | percent: 762.19 → percent: -9.65 | Tháng 6/2026 bị so với 01–03/08 do as_of_date ở tháng 9, thay vì cả tháng 5. **Hướng xử lý:** Dùng tháng trước của kỳ hỏi; tính ngày bằng công cụ và kiểm tra quan hệ hai kỳ. |
| 33 · `q_095e5097bfb5` | doanh thu trung bình trên một đơn tháng 5/2025 là bnhiu hả | number: 141700100 → number: 72444 | Hỏi doanh thu trung bình/đơn nhưng chỉ SUM; group_by order_id bị compiler sum bỏ qua. **Hướng xử lý:** Tổng tiền dòng / COUNT DISTINCT đơn, làm tròn tiền theo rulebook; không âm thầm bỏ group_by. |
| 36 · `q_4bbd72655e8e` | thang 9/2025 mat hang Chan ga ngam sa tac ban duoc tong cong bao nhiu dia | number: 1096349500 → number: 182 | Hỏi số đĩa Chân gà ngâm sả tắc nhưng trả toàn bộ doanh thu cửa hàng, thiếu cả sản phẩm và số lượng. **Hướng xử lý:** Giữ measure quantity và đầy đủ thực thể/đơn vị; lookup tên không dấu có kiểm tra duy nhất. |
| 37 · `q_b3e5c2fb4852` | xem giúp tổng tiền bán các mặt hàng k kê đơn tháng 4/2026 là nhiêu | number: 134177600 → number: 129523500 | Bỏ điều kiện không kê đơn, thay bằng trạng thái completed. **Hướng xử lý:** Nối bảng thuốc và lọc cờ kê đơn theo README; trạng thái vẫn theo policy bán hàng. |
| 43 · `q_0db8d6ef55e9` | thang 5/2025 ngay nao cua hang ban duoc nhiu tien nhat | string: "2025-05-11" → date: "2025-05-11" | Tìm đúng ngày 11/05/2025 nhưng rank mặc định xuất string, sai type date. **Hướng xử lý:** Dùng best_day/date hoặc hỗ trợ type theo cột xếp hạng có kiểm tra; không chấm chuỗi ngày thành đúng type. |
| 48 · `q_ce030a6717ff` | doanh thu cua topping thach ca phe trong thang 12 nam 2025 la bao nhiu vay | number: 0 → number: 742000 | Tên thach ca phe không dấu không khớp tên có dấu; SUM COALESCE làm lỗi khớp tên thành 0. **Hướng xử lý:** Lookup tới key/tên chuẩn duy nhất; kiểm tra số đối tượng khớp trước khi cộng, không coi mọi 0 là sai. |
| 54 · `q_788ab61d79d1` | Liệt kê 5 mặt hàng bán chạy nhất tháng 9/2025. | number: 5165 → list: list 5 đối tượng | Hỏi top 5 mặt hàng nhưng operation sum/type number; group_by và limit 5 bị bỏ qua. **Hướng xử lý:** Dùng rank/list, product_key/product_name, limit 5; validator từ chối grouping/limit không được thực thi. |
| 56 · `q_9a3b9473390e` | tỉ trọng của nhóm cơm mì cháo trong dthu tháng 8/2025 là bnhiu % | percent: 0.0 → percent: 12.07 | Sau lỗi cột product_group, model đổi sang tên phải đồng thời chứa Cơm, Mì VÀ Cháo. **Hướng xử lý:** Dùng cột nhóm thật trong danh mục; không thay nhóm bằng AND các từ trong tên. |
| 58 · `q_a7563e07926b` | thang 3/2025, dthu cac don tra bang vi dien tu va cac don tra bang chuyen khoan mo… | number: 535004000 → list: [106656000, 420904000] | Gộp ví điện tử và chuyển khoản thành một SUM, còn đưa thêm debt ngoài câu hỏi. **Hướng xử lý:** Series/list với hai nhóm độc lập ewallet rồi transfer, giữ thứ tự và đủ điều kiện. |
| 60 · `q_e308f9743d56` | thang 3/2026, tra hang vi loi va tra vi mua nham, ben nao nhieu don hon | number: 2332 → unanswerable: unanswerable | Không có lý do trả hàng nhưng model đếm đơn completed và nhóm channel; không trả lời được so sánh lý do. **Hướng xử lý:** Kiểm tra thuộc tính lý do trong toàn schema/profile; thiếu thực sự thì unanswerable, không suy từ trạng thái/kênh. |
| 63 · `q_39698eb3a76b` | Tháng 1/2025 có bao nhiêu ngày đạt dthu từ 8.600.000 đồng trở lên? | number: 0 → number: 7 | Đúng day_count nhưng lọc dòng tiền >= 8,6 triệu trước khi cộng theo ngày, làm mất hầu hết giao dịch. **Hướng xử lý:** Cộng doanh thu theo ngày rồi áp threshold 8,6 triệu ở kết quả nhóm; bỏ điều kiện tiền từng dòng không được hỏi. |
| 64 · `q_0d5667212f41` | trong thang 2 nam 2025 cua hang ban duoc may ly tra sua o long (m) | number: 125322000 → number: 149 | Hỏi số ly Trà sữa ô long (M), plan dùng revenue và chỉ lọc size M, trả tiền nhiều sản phẩm. **Hướng xử lý:** Giữ quantity, đúng tên hàng và base_unit ly; tránh dùng size thay cho toàn bộ thực thể. |
| 69 · `q_8f42e0c8d88a` | Doanh thu các đơn trả bằng COD và doanh thu các đơn trả bằng ví điện tử tháng 10/2… | number: 0 → list: [46892500, 108264500] | Yêu cầu hai doanh thu nhưng bắt payment_method đồng thời là cod VÀ ewallet; ra 0. **Hướng xử lý:** Dùng hai nhánh series/list; điều kiện bằng nhau loại trừ trên cùng cột cần được báo mâu thuẫn. |
| 71 · `q_ef9ec0a860f8` | Tháng 6/2026 thu hồi được bao nhiêu tiền công nợ từ khách mua chịu? | number: 0 → number: 9792001 | Thu hồi công nợ bị tính bằng SUM(balance) trên đơn đặt trong tháng; đó là tồn dư nợ, không phải dòng phiếu thu. **Hướng xử lý:** Đọc dm_payments theo ngày thu, chỉ đơn ghi nợ không hủy và phiếu sau ngày mua, kể cả đơn kỳ trước. |
| 73 · `q_237a88f0c7e3` | So doanh thu các đơn trả bằng COD với doanh thu các đơn trả bằng tiền mặt tháng nà… | number: 0 → list: [21961500, 204504500] | Bắt cùng một đơn vừa cod vừa cash và trả scalar thay vì list; ra 0. **Hướng xử lý:** Dùng series_filters hai nhóm độc lập, một kỳ this_month, list đúng thứ tự. |
| 74 · `q_d3837b22e001` | ngay 21/08/2025 cappuccino (m) tang gia tu 45.000đ len 47.000đ. so ly ban duoc tro… | percent: 0.25 → percent: 53.85 | Chỉ lọc mọi sản phẩm size M, bỏ tên Cappuccino nên tăng trưởng là của cả nhóm size. **Hướng xử lý:** Lookup/filter đầy đủ Cappuccino (M), giữ số lượng ly và hai kỳ giá đã xác định đúng. |
| 75 · `q_7c0a4b5f5196` | Tháng 5/2026 có mấy ngày doanh thu đạt 11.000.000 đồng trở lên? | number: 0 → number: 14 | Dùng count dòng bán có line_amount >= 11 triệu thay cho đếm ngày có tổng doanh thu đạt ngưỡng. **Hướng xử lý:** Dùng day_count với threshold sau tổng hợp theo ngày; kiểm tra cấp dữ liệu của ngưỡng. |
| 81 · `q_13afb3a15b07` | Ngày 01/12/2025 cửa hàng bán Sinh tố bơ (M) giá bao nhiêu một ly nhỉ? | number: 38639.34426229508 → number: 47000 | AVG giá của mọi hàng size M, bỏ tên Sinh tố bơ; còn dùng sales đã lọc hủy/trả dù luật giá bán tính cả các trạng thái. **Hướng xử lý:** Chọn đúng sản phẩm/ngày/đơn vị cơ bản và population cho giá; kiểm tra số giá distinct, không lấy trung bình nhóm sản phẩm. |
| 84 · `q_b0ff97ca1db9` | tuan nay ban duoc bn ly ha | number: 883 → number: 771 | Cộng mọi đơn vị cơ bản thay vì chỉ mặt hàng có đơn vị ly. **Hướng xử lý:** Giữ base_unit=ly theo câu hỏi/rulebook; không cộng các đơn vị khác vào cùng số lượng. |
| 86 · `q_117496103d90` | thang 4/2026 mat hang ngheu hap sa thu ve bn tien | number: 0 → number: 19653000 | Tên ngheu hap sa không dấu không khớp dữ liệu Nghêu hấp sả; bị chốt 0. **Hướng xử lý:** Lookup/key duy nhất hoặc tìm không dấu có kiểm tra phạm vi, giữ tên gốc khi xuất đáp án. |
| 87 · `q_7dded87ba54e` | Doanh thu tháng 4/2026 hơn hay kém tháng trước bao nhiêu phần trăm? | percent: 136.11 → percent: 0.44 | Tháng 4/2026 so với 01–13/06 do as_of_date tháng 7; kỳ so đúng là tháng 3. **Hướng xử lý:** Neo quan hệ tháng trước vào tháng 4 được hỏi; kiểm tra kỳ hỏi và kỳ so độc lập. |
| 95 · `q_9d3c67eadb66` | thang 7/2025 cac mat hang nhom tra sua thu duoc tong cong bao nhieu tien | number: 302714000 → number: 65734000 | Bỏ hẳn nhóm trà sữa nên trả toàn shop. Dòng bán không có product_id, không thể nối danh mục bằng ID trực tiếp. **Hướng xử lý:** Ánh xạ tên hàng duy nhất sang danh mục rồi dùng cột nhóm; không chỉ tìm chữ trà sữa trong tên sản phẩm. |

## 8. Toàn bộ 19 câu abstain

| # mẫu / ID | Nguyên nhân |
|---|---|
| 8 · `q_2ea34572153a` | Plan xếp hạng hợp lý nhưng nguồn POS có ngày serial chưa parse; mapping tên/khách cũng chưa đầy đủ. Không bỏ guard ngày để cộng thiếu. |
| 15 · `q_aff1f9ce5753` | Bị chặn ngày serial POS; plan còn thay nhóm sỉ bằng customer_key not_null, không tương đương nhóm khách. |
| 16 · `q_26bd17df26e7` | Lọc đơn vị cái trong khi câu hỏi/nguồn dùng phần; không khớp đối tượng nên mẫu số 0, model suy thiếu dữ liệu. |
| 21 · `q_e210942807ec` | Câu boolean tăng liên tục bị lập series/list và SQL trên *_all; vòng sửa không giữ type mục tiêu. Công cụ continuous_growth đã có. |
| 22 · `q_adef8c6a2761` | Plan tỷ lệ hủy đúng cấu trúc nhưng 16.594 đơn có ngày serial chưa parse; fallback SQL cũng bị chặn chất lượng nguồn. |
| 25 · `q_717e5f91d40b` | Bịa group_name trên facts, lặp cùng lỗi; cần nối nhóm trong danh mục và giữ mẫu số toàn shop. |
| 38 · `q_a8978d09f36e` | Đưa cả SELECT vào value của op in thay vì list; fallback SQL dùng nguồn *_all không đúng scope bán hàng. |
| 47 · `q_43bd6fa219e8` | POS thiếu ánh xạ tổng phải trả và ngày phiếu thu, sửa schema thất bại; nhiều lượt schema/SQL làm vượt context 16K. Không phải timeout GPU. |
| 59 · `q_05b478fbe670` | Dùng eq với tên thuốc rút gọn, không khớp tên đầy đủ trong nguồn; mẫu số 0. Cần lookup/key thay vì suy unanswerable. |
| 61 · `q_cf239556b57d` | Công nợ POS chưa chuẩn hóa đủ ngày/tổng đơn/phiếu thu; balance NULL và source guard chặn. Không được thay NULL thành 0. |
| 67 · `q_250cf7d222c0` | Bịa cột han_dung/i.han_dung và bảng chuẩn không có; dữ liệu lô/expiry nằm trong nguồn POS chưa ánh xạ đủ. Cần quan hệ lô và hạn dùng thật. |
| 76 · `q_e2e41d53f803` | Hỏi tháng cao nhất nhưng plan xếp hạng khách, list/limit=1 mâu thuẫn và bỏ tháng giữa của khoảng. Cần tổng theo ba tháng và trả date YYYY-MM. |
| 79 · `q_e4f212c5a50f` | Thay nhóm đá xay bằng hậu tố tên (Đá xay), vừa lọc tử vừa mẫu số; không khớp nên tỷ lệ không xác định. |
| 83 · `q_4afebc6d1d4b` | Dùng customer_key="si" thay cho thuộc tính nhóm sỉ; khóa/nguồn không đúng, SQL sau đó lỗi aggregate. |
| 85 · `q_97378cee7c0e` | SQL trộn order_date/payment_date/amount trên cùng bảng đơn, thiếu join sổ thu; còn sai scalar thay vì hai dòng tiền. |
| 89 · `q_cea1b708d80a` | Plan inventory/population custom nhưng operation sum, lặp lỗi contract dù đã thêm SQL. Công cụ inventory snapshot đã có. |
| 90 · `q_4dc0ead903ab` | Nhóm gỏi bị hiểu thành channel="nhom hang"; thiếu role channel bị guard chặn và SQL NULL, không phải nhóm gỏi không có doanh thu. |
| 91 · `q_ddd87e1d24ed` | Hỏi số mặt hàng ế nhưng plan list/tên; SQL CTE bị lỗi lineage báo như SELECT hằng, và query tự chọn mốc lịch sử chưa giữ toàn bộ trước kỳ. |
| 100 · `q_542bdf8584fa` | Plan bịa numerator_filters_2/denominator_filters, sai ratio thay cho chênh lệch phần trăm so ca sáng. Công cụ shift_growth đã có. |
