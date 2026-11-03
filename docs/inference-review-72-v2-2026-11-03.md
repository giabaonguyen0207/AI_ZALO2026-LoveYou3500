# Phân tích inference lần 2 — Gemma 4 12B, pipeline v2

**Làm rõ sau báo cáo:** người dùng muốn xem đầu ra LLM, SQL và kết quả/lỗi từng bước. Những nội dung này đã nằm trong JSON trace; đã xuất thêm bản Markdown dễ đọc và bổ sung logging cho lượt chạy sau. Xem [hướng dẫn log từng bước](debug-step-logs.md). Kiểm tra source hash trong báo cáo này ghi nhận thời điểm trước thay đổi logging đó.

**Ngày nhận và kiểm tra:** 03/11/2026. **Nguồn:** thư mục người dùng cung cấp tại `outputs/diagnosis-2026-11-03/gemma4-12b-stress72-pipeline-v2-01/`. Đối chiếu với lượt trước tại `outputs/diagnosis-2026-11-02/diagnostic_bundle/`, mã local và rulebook train thực tế. Thời gian sửa file trong bundle không được dùng làm bằng chứng về ngày chạy.

**Kết luận:** lượt 2 đúng **56/72**, sai **1**, bỏ qua **15**, PA **76,39%**. Cải thiện trên cùng mẫu development đã được xác nhận bằng chấm lại. Pipeline phụ thuộc mạnh vào các công cụ và bộ nhận diện viết sẵn; một số bộ kiểm tra quá cứng đang chặn câu có thể tính được. Không tìm thấy nhánh tra đáp án theo question/business ID trong các môđun inference đã kiểm tra. Reasoning bằng văn bản bị tắt chủ động trong backend và prompt.

**Phạm vi công việc lần này:** phân tích, kiểm chứng CPU và cập nhật tài liệu. Không sửa mã inference, không gọi LLM, không fine-tune, không thay submission gốc. Các kết quả công cụ gọi lại bên dưới là phép chẩn đoán có tham số được người phân tích chỉnh, không phải điểm của agent sau sửa.

## 1. Kiểm tra tính nhất quán của bundle

- Đủ 72 ID duy nhất và 72 log. Submission, review, từng log và danh sách câu được chọn khớp nhau.
- Cùng 72 câu, cùng thứ tự, nội dung, business, `as_of_date`, reference và alias với baseline; cũng khớp dữ liệu train local.
- Fingerprint train: `5d598c5d603020c8077df7802cc6fb06b364af9914eb88425bd7664c639978ea`, khớp manifest sau khi tính lại.
- Hash của **33 môđun source** trong manifest đều khớp file local. Identity hash của manifest và SHA-256 của cả 198 đầu ra thô trong log đều hợp lệ.
- Chấm lại submission bằng local evaluator cho đúng kết quả summary. Đây chưa phải kết quả từ bộ chấm chính thức.
- Không có doanh nghiệp holdout trong lượt chạy. Giữ nguyên 6 doanh nghiệp đã dành riêng; không chạy công cụ trên dữ liệu của chúng trong lần phân tích này.
- Manifest model giống baseline. Cùng Gemma 4 12B, `single`, không quantization, BF16, tham số trên `cuda:0`; GPU ghi nhận là RTX PRO 6000 Blackwell Server Edition, khoảng 94,97 GiB VRAM.
- Có thay đổi môi trường: Transformers **5.21.0 → 5.22.0**. Vì vậy đây không phải thí nghiệm chỉ thay đúng một biến; không thể quy chính xác toàn bộ mức cải thiện cho riêng một sửa đổi.

Bằng chứng có cấu trúc: [audit.json](../outputs/diagnosis-2026-11-03/analysis-pipeline-v2/audit.json). Có thể chạy lại kiểm chứng bằng script trong cùng thư mục; script không nạp model.

## 2. Kết quả so với lượt 1

| Chỉ số | Lượt 1 | Lượt 2 |
|---|---:|---:|
| Đúng | 9/72 | **56/72** |
| Sai | 42 | **1** |
| Bỏ qua | 21 | **15** |
| Tỷ lệ đúng trên toàn mẫu | 12,50% | **77,78%** |
| PA = (đúng − sai)/72 | −45,83% | **76,39%** |
| Đúng trong số câu đã trả lời | 9/51 = 17,65% | **56/57 = 98,25%** |
| Lượt gọi model | 500 | **198** |
| Thời gian riêng batch, không gồm load model | 3.964,18 giây | **1.095,052 giây** |
| Trung bình/câu | 55,05 giây | **15,20 giây** |
| Trung vị/câu | 46,70 giây | **10,87 giây** |
| p95/câu | 139,01 giây | **32,65 giây** |

Batch mới mất khoảng **18 phút 15 giây**, nhanh khoảng **3,62 lần**. Đây là thời gian batch trong môi trường đã chạy, không phải thời gian Docker từ lúc khởi động trên phần cứng cuộc thi.

Chuyển trạng thái từng câu: 38 câu sai → đúng, 11 câu bỏ qua → đúng, 4 câu sai → bỏ qua, 7 câu đúng được giữ. **Hai câu đúng → bỏ qua** là `q_83ddbda300ed`, `q_80f40d2055e8`; một câu bỏ qua → sai là `q_9be15002fb02`.

| Type chuẩn | Số câu | Đúng | Sai | Bỏ qua |
|---|---:|---:|---:|---:|
| number | 10 | 10 | 0 | 0 |
| date | 10 | 10 | 0 | 0 |
| percent | 10 | 9 | 0 | 1 |
| string | 10 | 10 | 0 | 0 |
| list | 9 | 5 | 1 | 3 |
| table | 7 | 4 | 0 | 3 |
| boolean | 8 | 1 | 0 | 7 |
| unanswerable | 8 | 7 | 0 | 1 |

Đầu ra còn **67 sự kiện lỗi**: 54 ở metric, 9 ở parse action, 4 ở chiếu đáp án; xuất hiện trên 33 câu, bao gồm cả các câu đã phục hồi thành công. Có 11 câu bị dừng do lặp action/kết quả. Cả 198 lượt đều ghi nhận greedy decoding; 197 kết thúc bằng EOS, một lượt chạm 768 token. Input lớn nhất ghi nhận 13.210 token; không thấy lỗi vượt context hoặc hết VRAM trong bundle này.

## 3. Có hard-code rules không?

### 3.1. Đáp án theo ID: chưa tìm thấy trong đường inference đã rà soát

`Agent`, backend, schema binding và chuẩn hóa không chứa bảng đáp án theo `q_*`/`biz_*`. Business ID dùng để mở thư mục đúng doanh nghiệp; giá trị được lấy từ query/công cụ. Reference được thêm vào review sau khi dự đoán đã tồn tại. Script chẩn đoán có dùng **type chuẩn** để cân đối mẫu, nhưng không chuyển reference value vào agent.

`tools/verify_semantic_pipeline.py` có danh sách 28 question ID và action định sẵn để kiểm tra hồi quy trên CPU. Đây là mã kiểm chứng riêng, không được runner inference gọi. Cả 28 câu này nằm trong mẫu 72; ở lượt 2, 24 đúng và 4 bỏ qua. Trong 44 câu còn lại: 32 đúng, 1 sai, 11 bỏ qua. Như vậy cần phân biệt rõ kiểm chứng bằng action định sẵn với model tự chọn action khi inference.

Những kiểm tra trên hỗ trợ kết luận về mã và bundle đang có; không thay thế kiểm tra độc lập của ban tổ chức hay xác minh quy định chính thức.

### 3.2. Quy tắc và template viết sẵn: có, và đang chi phối mạnh

- `data/rules.py:37`: đọc rulebook bằng số mục và các chuỗi/regex đã biết. Ví dụ mục 1 cho doanh thu, mục 2 cho thời gian, mục 4 cho công nợ; đây chưa phải khả năng hiểu rulebook tùy ý.
- `data/binding.py`: nhiều alias tên cột và mẫu mô tả README được viết sẵn để gắn cột với vai trò.
- `agent/verification.py:25`: từ khóa quyết định metric/type. Ví dụ cùng xuất hiện “ca sáng/ca chiều” thì buộc `shift_growth/percent`; có doanh thu và “phần trăm” thì chọn `revenue_growth`. Có các nhánh cụ thể như “Zalo/quầy”, “từ 3/8/14 đơn”.
- `agent/tools.py:201`: SQL và công thức cho từng metric được lập trình trước.
- `agent/runner.py:81`: câu đã bị nhận diện vào một metric thì không được final từ SQL có cờ `verified=False`, kể cả SQL đọc facts chuẩn hóa và đã cho đúng số.

Trong **56 câu đúng**, **49 câu lấy kết quả từ metric viết sẵn**, 7 câu là `unanswerable` có bằng chứng. Không có câu đúng nào trong lượt này được giải bằng nhánh SQL tự do. Vì vậy điểm tăng cho thấy hệ thống gồm model, dữ liệu chuẩn hóa và công cụ hoạt động tốt hơn trên mẫu đã sửa; không chứng minh riêng năng lực suy luận tổng quát của Gemma tăng lên.

Viết công cụ tính xác định là lựa chọn thiết kế hợp lý: không cần để LLM tự cộng tiền hay tự đổi ngày. Phần gây vấn đề là dùng nhận diện từ khóa chưa đủ chắc chắn làm bộ phán quyết bắt buộc, đồng thời hạn chế cả bước tính trung gian và phương án SQL thay thế.

Đã thử các câu tổng hợp, không lấy từ test, chỉ qua `question_contract()`:

| Câu tổng hợp | Contract hiện tại | Vấn đề quan sát được |
|---|---|---|
| Tỷ trọng doanh thu mặt hàng A chiếm bao nhiêu phần trăm tổng doanh thu tháng này? | `revenue_growth/percent` | Tỷ trọng bị gán thành tăng trưởng |
| Ngày nào có số đơn cao nhất tháng này? | `best_day/date` | Công cụ best_day xếp ngày theo doanh thu, không theo số đơn |
| Ca chiều hơn ca sáng bao nhiêu tiền trong tuần trước? | `shift_growth/percent` | Câu hỏi tiền bị ép sang phần trăm |
| Nhân viên nào bán giỏi nhất theo số đơn trong tháng này? | `rank_employees/string` | Công cụ cố định theo doanh thu dù câu nêu tiêu chí số đơn |

Đây là bằng chứng trực tiếp về sự quá cứng của bộ nhận diện, chưa phải kết quả inference của các câu tổng hợp này. Rulebook train cũng ghi câu hỏi nói khác thì làm theo câu hỏi. Không thể bảo đảm khả năng tổng quát sang split khác chỉ bằng việc đã đưa toàn bộ rulebook vào prompt.

### 3.3. Giới hạn của điểm development

Cùng 72 câu đã được xem để chẩn đoán và sửa pipeline. Có thêm 28 ca được gọi công cụ trực tiếp trước đó. Mẫu còn được chủ động cân đối type và tín hiệu ngôn ngữ, không phải mẫu ngẫu nhiên đại diện toàn train. Vì thế **76,39% PA là điểm của lượt chạy trên mẫu development này**, không phải ước lượng private test hay bằng chứng không overfit.

## 4. Phân loại toàn bộ 16 câu chưa đúng

| Nhóm nguyên nhân | Số câu | ID |
|---|---:|---|
| Chuỗi nhiều tháng: `period/periods`, cấm metric trung gian, diễn giải kỳ sai | 7 bỏ qua | `q_e2ea7a9d48c2`, `q_c5639bf3d2a2`, `q_b4a4c7135e0a`, `q_a569fbc6b9fc`, `q_83ddbda300ed`, `q_9bc44da7a632`, `q_80f40d2055e8` |
| Công nợ tại ngày chốt bị kiểm tra như phát sinh riêng ngày đó | 4 bỏ qua | `q_8ae4326131f8`, `q_e9930d87abdd`, `q_faf7802261fb`, `q_8423c5cc4a42` |
| Danh sách doanh thu ba tháng và cổng chặn SQL final | 1 bỏ qua | `q_b622d7d743bc` |
| Đối tượng sản phẩm sai cấu trúc tham số/key | 1 bỏ qua | `q_dd6531374a2b` |
| Chưa triển khai tái dựng tồn theo lô | 2 bỏ qua | `q_fe14e521d3e0`, `q_3aae102f4eb2` |
| Sai nghĩa nợ phát sinh/trả nợ và sai type | 1 sai | `q_9be15002fb02` |

### 4.1. Xu hướng nhiều tháng: lỗi phối hợp model và validator

Model thường gửi danh sách vào `period`, trong khi công cụ cần `periods`. Validator tự bổ sung `periods` nhưng giữ nguyên `period` sai. Sau đó nó kiểm tra `period` như một object; nếu model đổi thành cả khoảng ba tháng thì lại so cả khoảng với **tháng đầu tiên** và từ chối.

`q_e2ea7a9d48c2`: câu “từ tháng 9 đến tháng 11” bị bộ parse kỳ chỉ trích tháng 9 và tháng 11, bỏ tháng 10. `q_83ddbda300ed`: câu nhắc lại các tháng trong phần giải thích bị lấy thành bảy kỳ có lặp. Hai trường hợp này còn khiến validator từ chối danh sách ba tháng đúng do người phân tích đưa vào.

Ngoài ra, để trả lời boolean, model thử `revenue_series` lấy các tổng trung gian nhưng bị từ chối vì contract chỉ cho `continuous_growth`. Cần phân biệt công cụ trung gian với loại kết luận cuối.

### 4.2. Công nợ: sai cách kiểm tra ngày chốt

Ví dụ `q_8ae4326131f8`: câu hỏi nợ đến hết 05/10/2025. Model gửi `date=2025-10-05`, kèm `period` từ đầu dữ liệu đến ngày đó. Validator so `period` này với riêng 05/10 và từ chối trước khi công cụ công nợ được chạy. Lỗi lặp tương tự ở bốn câu, bao gồm một câu mà preflight đã có bằng chứng thiếu dữ liệu để trả `unanswerable`.

Công cụ `debt_rank` thực tế nhận ngày chốt, không nhận period. Trường hợp này có cả tham số thừa từ model và thông báo/logic kiểm tra không đúng mục tiêu; không phải bằng chứng công cụ đã tính sai nợ.

### 4.3. SQL đúng số vẫn bị chặn

`q_b622d7d743bc`: SQL đã trả ba tháng đúng:

```text
2025-10: 1.186.031.600
2025-11: 1.153.886.300
2025-12: 1.219.569.100
```

Runner từ chối vì kết quả SQL không mang cờ verified của metric. Model cũng dùng `type=number` trong lần final này thay vì list. Sau lỗi, model kết luận nhầm công cụ không hỗ trợ nhiều kỳ, dù `revenue_series(periods=[...])` thực tế có hỗ trợ. Cờ verified là dấu hiệu đường tính của chương trình, không phải bộ chứng minh độc lập rằng mọi lựa chọn ngữ nghĩa đều đúng.

### 4.4. Tên sản phẩm và tồn kho

`q_dd6531374a2b`: model gửi `product={"product_name":...}` trong khi API nhận tên dạng chuỗi hoặc `{"key":...}`. Sau đó nó dùng tên làm key dù dữ liệu có mã riêng. Lookup chứa thêm từ “bình” cũng không tìm được ứng viên; SQL đã tìm thấy tên thật nhưng model chưa chuyển sang tham số đúng. Gọi lại bằng tên hợp lệ cho kết quả 121,74%, khớp nhãn.

Hai câu tồn kho có sổ nhập/lô nhưng công cụ hiện mới hỗ trợ snapshot. Bỏ qua ở đây phản ánh phần chức năng chưa triển khai; không có cơ sở đổi thành `unanswerable`.

### 4.5. Câu sai duy nhất: dùng tiền đã trả hiện tại để tính nợ phát sinh quá khứ

`q_9be15002fb02` hỏi hai số tháng 3/2025 của Tô Dũng Thảo. Chuẩn là `list: [74000, 37000]`; agent nộp `number: 0`.

Đã kiểm tra nguồn thực tế:

- Đơn ghi nợ `ORD005916`, ngày 23/03/2025, tổng phải trả 74.000.
- Trả 37.000 ngày 28/03/2025 và 37.000 ngày 07/04/2025.
- Không trả ngay ngày mua: nợ thêm trong tháng 3 là 74.000; trả nợ trong tháng 3 là 37.000.
- `paid=74000` trong bảng đơn là tổng đã trả về sau. Lấy `order_total-paid` để tính nợ thêm tháng 3 làm mất lịch sử và cho 0. Model cũng đã bỏ lọc chỉ đơn ghi nợ sau khi điều kiện nhãn raw không khớp canonical.
- `question_contract()` không nhận ra yêu cầu hai thành phần của luồng công nợ; kiểm tra SQL chỉ xét nguồn bảng/ngày/trạng thái ở mức hạn chế, không chứng minh đúng nghĩa nghiệp vụ hoặc đủ hai đầu ra.

Một query CPU nối sổ thanh toán theo ngày đã cho `[74000,37000]`, khớp nhãn. Query và các dòng bằng chứng lưu tại [debt-flow-probe.json](../outputs/diagnosis-2026-11-03/analysis-pipeline-v2/debt-flow-probe.json). Đây là phép kiểm chứng phân tích, không thay đổi dự đoán đã nộp.

## 5. Tại sao không có model reasoning trong logs?

Có ba yếu tố riêng biệt:

1. **Tắt chế độ sinh thinking:** `src/datamate/models/backends.py:92` truyền `enable_thinking=False` cho chat template.
2. **Prompt yêu cầu action thuần:** `src/datamate/agent/prompts.py:7` ghi “Mỗi lượt DUY NHẤT một JSON object; không reasoning, markdown, thought hay call: ngoài JSON.”
3. **Không có trường reasoning riêng:** log lưu `trace[].raw`, action, observation, generation và final. File CSV là bảng tóm tắt; không chứa toàn trace. JSON từng câu và `review.jsonl` mới có đầu ra thô.

Chat template trong chính `runtime.json` cho thấy khi thinking tắt, generation prompt chèn sẵn `<|channel>thought\n<channel|>`: một kênh **rỗng đã đóng**. Vì thế nhìn thấy từ thought trong `generation.prompt_tail` không có nghĩa model đã sinh một đoạn suy nghĩ bị mất khỏi log.

Ở v2, backend decode với `skip_special_tokens=False`; runner lưu raw trước khi parse. Parser có bỏ các envelope thinking hợp lệ để lấy action nhưng không ghi đè raw đã lưu. Do đó nguyên nhân chính là cách cấu hình và yêu cầu sinh, không phải bộ ghi log âm thầm xóa toàn bộ reasoning model đã phát ra.

Có ngoại lệ: ở bước 7 của `logs/q_9be15002fb02.json`, raw chứa đoạn model tự sửa bằng tiếng Anh và lặp nhiều JSON; đầu ra chạm 768 token và parse thất bại. Đoạn đó vẫn nằm trong log. Nó cho thấy văn bản giải thích nếu phát ra có được lưu, đồng thời cho thấy bật thinking không tự động giải quyết lỗi hợp đồng công cụ.

Để debug rõ hơn, nên tách **đầu ra reasoning do model sinh** khỏi **action máy thực thi** và lưu một bản tóm tắt quyết định: mục tiêu, nguồn/cột, kỳ, công thức, giả định và bằng chứng. Nếu thử bật thinking cho model local, cần tùy chọn cấu hình, parser hiểu channel, ngân sách riêng và so sánh trên cùng dữ liệu. Văn bản reasoning là nội dung model sinh; không phải bản đọc đầy đủ hoạt động nội bộ hay bằng chứng tự thân rằng đáp án đúng.

## 6. Kiểm chứng CPU và bước sửa nên ưu tiên

Đã gọi lại công cụ cho **13/15 câu bỏ qua**, với tham số được người phân tích chỉnh từ câu hỏi/log, không đưa reference value vào phép tính. **13/13 kết quả khớp nhãn**. Trong đó hai action ba tháng hợp nghĩa vẫn bị validator hiện tại từ chối; công cụ tính đúng khi được gọi trực tiếp. Hai câu tồn theo lô không nằm trong 13 probe này. Câu sai duy nhất được kiểm chứng riêng như phần trên.

Các probe chứng minh những đường tính cụ thể hiện có, không chứng minh model sẽ tự chọn đúng chúng sau sửa. Không cộng những kết quả này vào 56 câu đúng hoặc công bố PA giả định.

Ưu tiên đề xuất, chưa triển khai trong lượt phân tích này:

1. Thống nhất schema action, chỉ rõ trường không hợp lệ và ví dụ sửa đúng. Tách date chốt, period đơn và periods nhiều kỳ; xử lý khoảng tháng, nhắc lại tháng và năm chuyển tiếp đúng nghĩa.
2. Chuyển bộ nhận diện từ khóa thành gợi ý khi chưa đủ bằng chứng. Kiểm tra kết luận cuối theo câu hỏi; cho phép các bước công cụ trung gian hợp lệ và đường SQL được kiểm tra điều kiện/nguồn/phép tính rõ ràng.
3. Bổ sung kiểm tra cấu trúc đầu ra nhiều thành phần và luồng nợ theo sổ thanh toán; giữ riêng bài toán snapshot và phát sinh trong kỳ.
4. Ghi lại contract/policy đã áp cùng đoạn rulebook nguồn, kế hoạch/tóm tắt lựa chọn, action gốc và action sau bổ sung tham số, kết quả kiểm tra, lý do từ chối cụ thể. Hiện `event['action']` có thể bị sửa tại chỗ khi validator thêm tham số; raw mới giữ action model gốc.
5. Sau sửa, kiểm tra hồi quy CPU rồi chạy model trên câu development chưa dùng để sửa. Đóng băng thay đổi trước khi đánh giá tập holdout đã dành riêng; không dùng public/private test để huấn luyện hoặc thu đáp án.

Các yêu cầu cuộc thi tổng quát vẫn là bản tổng hợp do người dùng cung cấp. Báo cáo này chỉ xác nhận quan sát trong train, mã và bundle thực tế; không tự xác nhận quy định split khác hoặc hiệu năng Docker chính thức.
