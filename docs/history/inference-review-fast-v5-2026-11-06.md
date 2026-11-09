**Phân tích bundle Gemma fast v5 nhận ngày 06/11/2026**

Bundle mới: `outputs/diagnosis-2026-11-05/gemma4-12b-random100-fast-v5-Nov5th`.
Đối chiếu chính: `outputs/diagnosis-2026-11-03/gemma4-12b-stress100-baseline-v3`.
Đây là chấm local trên dữ liệu TRAIN đã dùng phát triển, không phải điểm thi hoặc ước lượng private test.

**Kết luận**

Fast v5 đã giải quyết chi phí chuẩn bị schema/rulebook của bản chạy dở bảy câu. Tuy nhiên, so với baseline v3 trên **đúng cùng 100 câu**, nó giảm PA từ **0,36 xuống 0,30**, đồng thời tăng thời gian trung bình **27,0%**. Vì vậy, tên “fast” chưa được kết quả thực tế xác nhận khi so với baseline v3.

Lỗi nằm ở cả cách model lập plan lẫn thiết kế hợp đồng, trình biên dịch và bước kiểm tra/chốt đáp án. Đặc biệt, code đang kiểm tra việc thực thi có phù hợp với plan, trong khi plan có thể đã diễn giải sai câu hỏi. Model không cần tự nhẩm sai: một plan sai được tính chính xác vẫn cho đáp án sai.

Chưa có bằng chứng lượt này dùng model fine-tuned. Manifest ghi cùng base checkpoint với lượt v3. Không có căn cứ quy kết suy giảm cho fine-tuning hoặc GPU nhỏ hơn.

**Những gì đã kiểm chứng**

- Đủ 100 câu, cùng ID và thứ tự với lượt v3, thuộc 25 doanh nghiệp; câu hỏi, business và `as_of_date` khớp TRAIN local. ID chỉ dùng ghép kết quả và chẩn đoán.
- Fingerprint dữ liệu khớp allowlist TRAIN: `5d598c5d603020c8077df7802cc6fb06b364af9914eb88425bd7664c639978ea`.
- Chấm lại `submission.jsonl` bằng evaluator local cho đúng số liệu của `summary.json`; submission khớp 100 log. ZIP qua kiểm tra CRC và khớp các file được xuất bên ngoài: 207 thành viên ở lượt mới, 206 ở lượt cũ.
- 255 đầu ra nguyên văn mới và 351 đầu ra cũ đều khớp SHA256 được ghi trong log. Lượt mới có thêm một lần generation thất bại vì vượt context, không có raw output.
- Các source hash inference được manifest mới ghi nhận khớp source local hiện tại. Chênh lệch duy nhất trong danh sách module được ghi nhận là `training.finetune`, không được dùng để trả lời trong run này.
- Hai runtime ghi cùng Python/dependency, GPU **NVIDIA RTX PRO 6000 Blackwell Server Edition, khoảng 95 GiB**, BF16, tham số trên `cuda:0`, greedy thực tế `do_sample=false`, `num_beams=1`. Model manifest cùng SHA256 `da6dc0a578c623c64e1ad7304e26f32b241fc30af248c2c71ada147271b71869`.
- Đã chạy lại các phép tính chẩn đoán trên CPU từ plan lưu thật và một số biến thể sửa thủ công. **Không chạy thêm SLM, không benchmark GPU mới, không sửa runtime và không đưa log mới vào SFT.**

**So sánh cùng 100 câu**

| Chỉ số | Baseline v3 | Fast v5 mới | Thay đổi |
|---|---:|---:|---:|
| Đúng / sai / bỏ qua | 49 / 13 / 38 | 46 / 16 / 38 | Ít đúng hơn, nhiều sai hơn |
| PA = (đúng − sai) / 100 | 0,36 | 0,30 | −6 điểm phần trăm |
| Tỷ lệ có trả lời | 62% | 62% | Không đổi |
| Trung bình giây/câu | 18,26 | 23,19 | +27,0% |
| Trung vị giây/câu | 12,33 | 17,95 | Tăng |
| P95 giây/câu | 46,87 | 64,10 | Tăng |
| Tổng thời gian các câu | 1.825,54 giây | 2.318,89 giây | Khoảng 30 phút 26 → 38 phút 39 |
| Lần gọi model | 351 | 256 | −27,1% |
| Tổng token đầu vào có log | 3.462.535 | 2.715.775 | −21,6% |
| Tổng token đầu ra | 18.791 | 32.769 | +74,4% |
| Token đầu ra trung bình/lần sinh có output | 53,54 | 128,51 | Khoảng 2,4 lần |
| Thời gian trong generation calls | 1.459,81 giây | 1.940,91 giây | +33,0% |

Thời gian trên gồm chuẩn bị doanh nghiệp, **không gồm load model**. Không dùng các số này để khẳng định đáp ứng ngân sách toàn private test.

Có **16 câu từng đúng nay mất**: 11 chuyển sang bỏ qua, 5 chuyển sang sai. Chiều ngược lại có **13 câu mới đúng**. Lượt mới có 14 câu sai vốn trước đó chưa sai: 5 từ đúng và 9 từ bỏ qua. Như vậy, không phải hệ thống chỉ thận trọng hơn; phạm vi trả lời giữ nguyên nhưng chất lượng trong phạm vi ấy giảm.

| Type chuẩn | V3: đúng / sai / bỏ qua | V5: đúng / sai / bỏ qua |
|---|---:|---:|
| number, 54 câu | 25 / 11 / 18 | 31 / 9 / 14 |
| percent, 24 câu | 14 / 1 / 9 | 10 / 5 / 9 |
| list, 10 câu | 3 / 0 / 7 | 0 / 1 / 9 |
| date, 8 câu | 5 / 1 / 2 | 5 / 0 / 3 |
| unanswerable, 3 câu | 1 / 0 / 2 | 0 / 1 / 2 |
| boolean, 1 câu | 1 / 0 / 0 | 0 / 0 / 1 |

Điểm tốt có thật: nhóm số đơn, doanh thu có điều kiện và một số kỳ thời gian đã cải thiện. Ví dụ lượt mới trả đúng số đơn huỷ 19/41 ở hai câu, doanh thu riêng ca chiều, lãi gộp và một số câu từ đầu năm từng lỗi. Không nên xoá toàn bộ phần tính bằng code chỉ vì tổng điểm giảm.

Nhưng nhóm phần trăm mất nhiều điểm, cả 10 câu list không có đáp án đúng, và boolean/unanswerable đều mất câu từng đúng. Nhóm 29 câu không dấu cũng giảm từ 13 đúng/4 sai xuống 11 đúng/8 sai; từ điển viết tắt hiện có chưa giải quyết được khớp thực thể.

**Nguyên nhân 1: plan sai nghĩa vẫn được tự động chốt**

61 câu được auto-project, gồm **45 đúng và toàn bộ 16 câu sai**. Một câu đúng còn lại qua `final` tường minh. Điều này không chứng minh tắt auto-project sẽ tự sửa đáp án, nhưng cho thấy chốt tự động hiện chưa có đủ tín hiệu phát hiện plan sai nghĩa.

Các trường hợp tiêu biểu:

- Ba câu tỷ lệ huỷ đặt `status=cancelled` vào **cả `filters` và `numerator_filters`**. Bộ tính lấy “đơn huỷ / đơn huỷ”, cho 100%. Khi chỉ sửa phạm vi mẫu số theo rulebook, phép tính CPU cho **2,42%; 1,37%; 2,18%**, khớp nhãn. Không cần đổi công thức số học hoặc checkpoint.
- Câu tỷ trọng nhóm gỏi đặt cùng danh sách sản phẩm vào cả hai phía, cũng ra 100% thay vì 10,95%. Danh sách này còn cần kiểm tra tính đầy đủ theo danh mục; xoá điều kiện mẫu số chưa chứng minh chọn đúng nhóm.
- Câu doanh thu khách nữ dùng `customer_name IS NOT NULL` làm điều kiện khách nữ, tự trả 100%. Dữ liệu không đủ thuộc tính cần hỏi và nhãn là `unanswerable`. Có tên khách không chứng minh giới tính.
- Câu “COD với tiền mặt, mỗi bên bao nhiêu” bị thu thành doanh thu COD, trả một `number` 21.961.500 thay vì list hai số `[21961500, 204504500]`. Query hợp lệ và số COD đúng vẫn không trả đủ câu hỏi.

`evidence_id="Q"` hiện chỉ gắn nguyên văn câu hỏi làm nguồn trích dẫn; nó không chứng minh phép diễn giải từ câu hỏi sang thuộc tính, điều kiện hoặc mẫu số là đúng. Tương tự, cờ `verified` của kết quả biên dịch xác nhận phạm vi kiểm tra công cụ, không phải chứng nhận hiểu đúng ý người dùng.

Hướng xử lý: thể hiện rõ phạm vi mẫu số, điều kiện tử số, số thành phần đáp án và thuộc tính nguồn cần có. Khi hai tập điều kiện trùng nhau hoặc thiếu thuộc tính được hỏi, phải kiểm tra/sửa plan có căn cứ. Không đặt luật “100% luôn sai”, “0 luôn sai”, hoặc tự xoá bộ lọc; các giá trị đó có thể hợp lệ ở câu khác.

**Nguyên nhân 2: bỏ qua bước gắn tên/nhãn với dữ liệu thật**

Lượt cũ có 26 action `lookup`, lượt mới chỉ còn 2. Trong các lỗi sai mới, **9 đáp án bằng 0** và **5 đáp án bằng 100**; đây là dấu hiệu để kiểm tra, không phải quy tắc quyết định đáp án.

Ví dụ đã kiểm chứng trên CPU:

| Câu | Plan mới / kết quả | Biến thể chẩn đoán từ câu hỏi và nguồn |
|---|---|---|
| Topping thạch cà phê, 12/2025 | So bằng `topping thach ca phe`, ra 0 | Khớp tên có xử lý dấu/hoa thường → 742.000 |
| Trà sữa ô long (M), 2/2025 | `contains` chuỗi không dấu, ra 0 | Khớp tên có xử lý dấu → 149 |
| Nghêu hấp sả, 4/2026 | `contains 'ngheu hap'`, ra 0 | Khớp tên có xử lý dấu → 19.653.000 |
| Đếm đơn huỷ, 3/2026 | Nhãn raw `Huỷ đơn`, `hủy` trên cột canonical, ra 0 | Giá trị canonical `cancelled` → 37 |
| “từ đầu năm tới heets hnay” | Biến `heets` thành bộ lọc sản phẩm, ra 0 | Bỏ bộ lọc sản phẩm không có căn cứ → 5.888.993.000 |

Hai câu tăng giá của Cappuccino và Magne-B6 từng đúng nay bỏ qua vì tên literal không khớp, mẫu số trở thành 0/NULL. Khớp tên có xử lý hoa thường/dấu phục hồi lần lượt **53,85% và 123,46%** trong phép thử công cụ.

Tuy nhiên, xử lý dấu không giải quyết được ý nghĩa nhóm hàng. Câu “nhóm trà sữa” đổi sang lọc tên không dấu chỉ đạt **51.842.000**, vẫn khác nhãn **65.734.000**: rulebook yêu cầu cột nhóm, không phải mọi tên có chuỗi “trà sữa”. Đây là kết quả âm cần giữ lại khi chọn giải pháp.

Hướng xử lý: resolver thực thể dựa vào toàn danh mục, ưu tiên mã/key khi có ứng viên duy nhất; giữ nguyên tên có dấu để xuất đáp án. Phân biệt tên sản phẩm với tên nhóm. Kiểm tra nhãn trạng thái/thanh toán theo enum của facts và ánh xạ đã có căn cứ. Khi bộ lọc không tìm thấy thực thể, cần phân biệt “tên không khớp” với “thực thể đúng nhưng không có giao dịch trong kỳ”. Không bật tìm gần đúng rồi nhận ứng viên đầu tiên.

**Nguyên nhân 3: hợp đồng plan và prompt còn quá dễ gây lỗi cho SLM**

- Prompt minh hoạ JSON bằng giá trị như `"measure":"revenue|quantity|orders|..."`. Có **4 câu thực sự chép cả chuỗi lựa chọn** vào trường `measure`; ba trong số đó còn chép chuỗi lựa chọn cho operation/type/population. Đây là bằng chứng thiết kế ví dụ chưa tốt, không chỉ giả thuyết về khả năng model.
- `output_order` chỉ hợp lệ cho `series`, nhưng bốn câu rank dùng nó và gặp tổng cộng 11 lỗi kiểm tra. Ba câu trong số đó từng đúng ở v3. `group_by`/`display` cần một tham chiếu cột, model lại dùng list; prompt hiện không cho ví dụ rank hoàn chỉnh đủ rõ về kiểu trường.
- Chỉ bỏ `output_order` và gỡ list một phần tử **chưa đủ**: thử lại top khách cho ra cả `khách lẻ`, `lẻ`, `kl` đứng đầu. Plan đang nhóm tên thay vì danh tính canonical theo rulebook. Dùng `customer_key` và display tên mới phục hồi top 5 đúng. Top mặt hàng còn có join dùng tên khoá không tồn tại; bỏ join không cần thiết và nhóm theo `product_key` mới phục hồi đủ top 5.
- Câu đếm ngày đạt 11 triệu bị lập `measure=revenue, operation=count`, lọc từng `line_amount>=11000000`. Trình biên dịch vẫn dùng SUM doanh thu vì chưa ràng buộc đủ cặp measure/operation; kết quả 0. Plan `day_count` với ngưỡng 11 triệu cho **14 ngày**. Đây cũng là thiếu sót kiểm tra tổ hợp phép toán trong code.
- Câu tăng liên tục ba tháng bám vào series và xuất bảng tháng/doanh thu thay vì boolean. Metric `continuous_growth` đã có nhưng không được model sử dụng. Không thể giải quyết chỉ bằng ép projection sang list.
- Câu tăng trưởng tháng 2/2025 có plan `growth` nhưng thêm inline SQL chỉ trả hai tổng doanh thu, không trả phần trăm; bước chiếu vì thế thất bại. Giữ nguyên plan và bỏ query để code tính trả **−8,64%**, khớp nhãn. Câu so ca chiều/sáng sinh `substr(order_timestamp,11,5)` bắt đầu ở khoảng trắng ngăn ngày và giờ; sửa sang vị trí 12 trong probe cho **40,45%**, khớp nhãn. Cho model viết lại SQL của phép tính đã hỗ trợ tạo thêm token và thêm cơ hội sai.

Hướng xử lý: ví dụ JSON phải là action hợp lệ thực sự; mô tả lựa chọn và kiểu trường tách khỏi giá trị ví dụ. Có ví dụ ngắn cho rank, ratio, nhiều thành phần và boolean; chỉ đưa trường liên quan. Kiểm tra cặp đại lượng/phép toán, nguồn/key, đơn vị và yêu cầu đầu ra. Lỗi phản hồi cần nêu trường nào phải sửa và dạng hợp lệ, thay vì chỉ “sửa plan theo lỗi”. Không nới lỏng validator để plan sai nghĩa đi qua.

**Nguyên nhân 4: có lỗi mặc định nguồn và theo dõi nguồn SQL trong code**

- `measure=inventory` không có `source` đang mặc định sang sales; khi population all thì thành `dm_sales_all`. Câu tồn Kem đánh răng P/S dùng bộ lọc `dm_inventory` bị báo “bảng chưa có trong joins” hai lần rồi bỏ qua. Giữ nguyên SQL và chỉ định nguồn `dm_inventory` trả **3**, khớp nhãn. Cần yêu cầu hoặc suy ra nguồn theo capability đã xác minh; không dùng mặc định sales cho mọi đại lượng.
- SQLite authorizer chỉ thêm bảng vào `direct_tables` khi tham số `trigger` là null. Đọc qua CTE có thể làm danh sách rỗng. Probe hai SQL tương đương `WITH t AS (SELECT COUNT(*) ... dm_orders_all) SELECT n FROM t` và derived table cùng trả **51.044**, nhưng CTE bị kết luận nhầm là SELECT hằng số, derived table được nhận. Đây là lỗi theo dõi phụ thuộc nguồn cần sửa.
- Trong log câu công nợ `q_6e13d0978b21`, truy vấn CTE thật cũng có `direct_tables=[]` và bị chặn như trên. Tuy nhiên SQL đó còn dùng nhãn/khách và phép tính nợ chưa đúng, trả NULL; **sửa lineage không đồng nghĩa cứu được đáp án câu này**.
- Một số chặn là cần thiết: nguồn raw chưa bảo đảm cấp đơn/trạng thái, snapshot paid/balance dùng cho nợ quá khứ, trạng thái/ngày chưa giải rõ, phép nối có thể nhân bản tiền. Giữ các kiểm tra đó và sửa chính xác lỗi nhận diện nguồn; không bỏ toàn bộ lớp bảo vệ SQL.

**Vì sao ít gọi model hơn mà vẫn chậm hơn?**

Chuẩn bị schema/rulebook không còn là nút thắt trong bundle mới: tổng **2,6298 giây**, một model call sửa schema. Thời gian tăng chủ yếu nằm trong generation calls: tăng khoảng **481 giây**, gần hết mức tăng tổng **493 giây**. Plan dài và việc model còn sinh inline SQL làm tổng token đầu ra tăng 74,4%; đây là nguyên nhân thiết kế có bằng chứng mạnh. Log chưa tách prefill/decode/kernel warmup, nên chưa quy toàn bộ chênh lệch tốc độ cho riêng một thành phần GPU.

38 câu bỏ qua dùng **162/256 model calls** và **1.434,87/2.318,89 giây**, tức **61,88% thời gian**; trung bình 37,76 giây/câu. Có 22 câu dừng vì lặp, 6 câu kết thúc bằng “Hết ngân sách bước hoặc thời gian”, 9 câu bỏ qua tường minh/khác và một lỗi xử lý context. Cả sáu câu cạn ngân sách đều đã gọi 6 lượt và kết thúc dưới 90 giây; không có bằng chứng timeout 90 giây là nguyên nhân chính.

Cần đọc cẩn thận `flag_counts.step_or_time_budget=28`: cách gắn cờ hiện tìm chữ “ngân sách” trong reason, nên tính cả 22 câu lặp có câu “tiết kiệm ngân sách”. Đây không phải 28 lần timeout độc lập. Các flag khác cũng có thể chồng lấp.

Trong 255 lần sinh có output, 254 kết thúc EOS và chỉ một chạm giới hạn 768 token. Vì vậy tăng `max_new_tokens` không phải giải pháp chính. Tăng số bước sẽ có thể cứu vài câu nhưng cũng kéo dài các vòng lặp; cần sửa hợp đồng/feedback trước khi thử tăng ngân sách.

Để tránh nhầm với bản chạy dở: đúng bảy câu đầu trước fast v5 từng mất **1.253,95 giây**, đều bỏ qua. Bảy câu ấy trong bundle mới mất **225,23 giây**, có 5 đúng/2 bỏ qua. Bảy câu baseline v3 mất **118,46 giây**, 6 đúng/1 sai. Phần chuẩn bị nặng đã được xử lý, nhưng chưa vượt baseline cả về số câu đúng và tốc độ trên nhóm nhỏ này. Không so trực tiếp trung bình của bảy câu với toàn bộ 100 câu để kết luận mức tăng tốc.

**Thứ tự xử lý đề xuất**

1. **Sửa tính đúng của giao diện trước.** JSON example hợp lệ; rõ kiểu `group_by/display`, trường riêng cho mỗi operation, danh sách enum; kiểm tra cặp measure/operation; nguồn inventory; feedback có lỗi theo trường. Sửa CTE lineage với kiểm tra vẫn chặn query hằng số. Chuẩn hoá reason code của diagnostics để tách timeout, hết bước và lặp.
2. **Gắn plan với dữ liệu trước khi tự chốt.** Resolve tên sang key có căn cứ, enum/status đúng nguồn, nhóm lấy từ danh mục, mẫu số/tử số rõ, đầy đủ các thành phần câu hỏi. Có số dòng khớp và trạng thái resolve trong log. Chỉ thêm lượt SLM kiểm tra khi còn mơ hồ; trường hợp không đủ thuộc tính thì thu thập bằng chứng thiếu hoặc abstain, không thế bằng cột gần giống.
3. **Giảm lượng model phải sinh.** Các phép tổng/đếm/xếp hạng/tăng trưởng đã biểu diễn được thì chỉ sinh plan ngắn để code biên dịch. Đưa schema/lookup liên quan vào một kết quả gọn; cache theo doanh nghiệp và nội dung nguồn. Dùng SQL cho phần thật sự chưa biểu diễn được. Lỗi cấu trúc có sửa được không mất nghĩa thì sửa có kiểm tra; lỗi nghiệp vụ giữ cho model giải quyết, không thay ý câu hỏi bằng heuristic.
4. **Đo từng thay đổi bằng model thật trước khi chọn kiến trúc cuối.** Dùng một nhóm nhỏ bao phủ lỗi để thử nhanh, cố định checkpoint/GPU/budget khi so sánh. Đối chiếu plan bắt buộc với đường gọi công cụ có kiểm tra và auto-project có điều kiện; baseline v3 là đối chứng. Sau đó chạy lại cùng 100 để kiểm tra hồi quy và dùng tập chưa tham gia sửa hệ thống để đo khả năng tổng quát. Theo dõi PA, wrong, list/boolean, token đầu ra, số lượt sửa và P95, không chỉ số model calls. Không khẳng định cấu hình mới nhanh/đúng hơn trước khi chạy.
5. **Fine-tune Gemma sau khi ổn định hợp đồng.** Seed đã chuẩn bị có thể dạy chọn operation, gọi công cụ, lookup và sửa lỗi; fine-tune không tự sửa lỗi CTE hay chứng minh plan đúng nghĩa. Nếu đổi contract/source runtime cần tái sinh hoặc tái kiểm chứng target, phát hành dataset/source manifest mới theo cơ chế hash hiện có. Không chép đáp án sai mới vào SFT; không đưa các câu giữ ngoài train vào train. Sau fine-tune, 100 câu này là development có chồng lấp nguồn seed, không được báo là đánh giá độc lập.

Các hướng trên đều là phép toán, hợp đồng và kiểm tra dùng chung, lấy schema/enum/quy tắc từ doanh nghiệp và split đang chạy. Không cần rule theo business ID/question ID hoặc thêm một metric riêng cho từng doanh nghiệp. Việc máy tính thực thi theo điều kiện cấu trúc là bình thường; điều cần tránh là cài sẵn đáp án hoặc quyết định nghĩa câu hỏi bằng các nhánh chỉ khớp bộ câu đang debug.

Phần triển khai trước đã kiểm chứng phần mềm và plan soạn tay, nhưng chưa có đo end-to-end với Gemma ở protocol mới. Bundle này cho thấy khoảng trống đó rõ ràng: công cụ tính tốt hơn ở một số câu số học, trong khi giao diện mới làm mất các luồng model từng xử lý được. Trách nhiệm sửa gồm cả code/prompt/đánh giá; chưa đủ cơ sở kết luận Gemma 12B không có khả năng thực hiện các nhiệm vụ này.

**Đối chiếu 16 đáp án sai mới**

ID dưới đây chỉ là mã tham chiếu log chẩn đoán, không phải điều kiện đưa vào agent.

| ID | Kết quả mới | Nhãn local | Nguyên nhân quan sát |
|---|---|---|---|
| q_9f82b2f25fd6 | percent 100 | unanswerable | Có tên khách bị coi là khách nữ |
| q_78928d7dfc95 | percent 100 | 10,95 | Lọc cùng nhóm cả tử/mẫu; nhóm cần kiểm chứng |
| q_46d72d387508 | percent 100 | 2,18 | Đơn huỷ chia đơn huỷ |
| q_a91735cf4993 | percent 100 | 2,42 | Đơn huỷ chia đơn huỷ; v3 từng đúng |
| q_f87ae1288ec0 | percent 100 | 1,37 | Đơn huỷ chia đơn huỷ; v3 từng đúng |
| q_2ca8fd946a09 | number 0 | 5.888.993.000 | “heets” bị coi là sản phẩm |
| q_ce030a6717ff | number 0 | 742.000 | Lọc tên không dấu literal; v3 từng đúng |
| q_0d5667212f41 | number 0 | 149 | Lọc tên không dấu literal; v3 từng đúng |
| q_117496103d90 | number 0 | 19.653.000 | Lọc tên không dấu literal; v3 từng đúng |
| q_255991af3110 | number 0 | 37 | Nhãn raw dùng trên status canonical |
| q_9a3b9473390e | percent 0 | 12,07 | Nhóm bị đổi thành tên chứa cơm AND mì AND cháo, mẫu số cũng bị thu hẹp |
| q_237a88f0c7e3 | number 21.961.500 | list [21.961.500, 204.504.500] | Bỏ mất thành phần tiền mặt và sai type |
| q_7c0a4b5f5196 | number 0 | 14 | Lọc tiền dòng thay vì cộng theo ngày rồi đếm |
| q_b0ff97ca1db9 | number 32.659.000 | 771 | Hỏi số ly nhưng plan chọn doanh thu |
| q_4dc0ead903ab | number 0 | 86.190.000 | Lọc product_key chứa “nhom hang”, mất đúng nhóm gỏi |
| q_9d3c67eadb66 | number 0 | 65.734.000 | Không dấu và nhầm nhóm danh mục thành chuỗi tên |

**Artifact và cách chạy lại**

Kết quả máy đọc tại `outputs/diagnosis-2026-11-06/fast-v5-review/`:

- `audit.json`: chấm lại, hash, runtime/settings, token/calls, lỗi và chuyển trạng thái từng cặp kết quả.
- `paired.json`: 100 câu; `wrong.json`: 16 sai; `lost.json`: 16 câu mất đáp án đúng; `gained.json`: 13 câu mới đúng.
- `probes.json`: 20 phép so sánh plan/biến thể trên 18 câu và hai query kiểm chứng CTE; lưu SQL, parameters, nguồn, kết quả/lỗi và đối chiếu nhãn. Đây là biến thể do người phân tích soạn để cô lập nguyên nhân, không phải trajectory model mới hoặc điểm PA sau sửa.
- Hai script phân tích chỉ ghi artifact/cache ngoài dataset; không sửa bundle đầu vào, model, runtime hoặc seed fine-tune.

```powershell
.\.venv\Scripts\python.exe -X utf8 outputs/diagnosis-2026-11-06/fast-v5-review/analyze_bundle.py
.\.venv\Scripts\python.exe -X utf8 outputs/diagnosis-2026-11-06/fast-v5-review/probe_causes.py
```

Log mới có `raw`/`llm_action`, `execution.sql`, `parameters`, kết quả/lỗi và `auto_final`. Phân biệt SQL trong `plan.query` do model sinh với SQL trong execution của plan không có query do code biên dịch. Không cần nội dung suy nghĩ ẩn để xác nhận các lỗi đã nêu; những lỗi này nhìn thấy trực tiếp ở action và kết quả công cụ.
