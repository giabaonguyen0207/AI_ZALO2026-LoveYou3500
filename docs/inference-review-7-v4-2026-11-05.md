# Lượt inference 4: chẩn đoán độ trễ sau bảy câu

Nguồn nhận ngày 05/11/2026 là output được lưu trong `notebooks/DataMate_Colab_Inference.ipynb`. Ô diagnostic ghi bảy câu hoàn thành và bị `KeyboardInterrupt` khi đang chạy câu thứ tám. Có summary của bảy câu và một trace đầy đủ của câu tương tác riêng tại `biz_101`. Chưa có ZIP diagnostic lượt 4 trong workspace, nên không suy đoán lỗi cụ thể của từng bước ở cả bảy câu.

## Kết quả đã xác nhận

| Chỉ số | Lượt 4, bảy câu hoàn thành | Lượt 3, đúng bảy ID tương ứng |
|---|---:|---:|
| Đúng / sai / bỏ qua | 0 / 0 / 7 | 6 / 1 / 0 |
| Tổng thời gian bảy câu | 1.253,95 giây | 118,46 giây |
| Trung bình mỗi câu | 179,14 giây | 16,92 giây |
| Lượt gọi model | 95 | 17 |

Thời gian ghi nhận của bảy câu mới dài hơn khoảng **10,59 lần**. Metadata hai lượt cùng Gemma 4 12B, BF16, Transformers 5.22.0, PyTorch 2.11.0+cu130 và RTX PRO 6000 Blackwell 94,97 GiB. Các tham số model được ghi trên `cuda:0`; chưa thấy bằng chứng offload CPU hoặc OOM trong output này.

| Doanh nghiệp | Lượt 4 | Lượt gọi model | Lượt 3 cùng câu |
|---|---:|---:|---:|
| biz_127 | 219,1 giây | 16 | 15,97 giây |
| biz_116 | 194,5 giây | 11 | 17,01 giây |
| biz_102 | 141,7 giây | 12 | 18,03 giây |
| biz_117 | 223,5 giây | 20 | 13,86 giây |
| biz_108 | 220,3 giây | 15 | 27,63 giây |
| biz_110 | 97,5 giây | 9 | 9,90 giây |
| biz_103 | 157,3 giây | 12 | 16,05 giây |

**Chuẩn bị doanh nghiệp chiếm 1.045,55 giây, tương đương 83,38% thời gian bảy câu**, với 53/95 lượt gọi model. Phần còn lại khoảng 208,39 giây, tức 29,77 giây/câu và sáu lượt gọi model/câu. Đây là phép tách thời gian trong log, chưa phải benchmark của một phiên bản đã tắt chuẩn bị.

Bảy câu thuộc bảy doanh nghiệp khác nhau nên đều chịu chi phí chuẩn bị ban đầu. Không được ngoại suy rằng mọi câu trong cùng doanh nghiệp sẽ tốn ba phút. Tuy nhiên, mọi câu đã hoàn thành đều bỏ qua và summary gắn cờ `preparation_validation` ở cả bảy câu. Cache chưa chứng minh được lợi ích về độ đúng ở lượt này.

Thời gian cả invocation là 1.484,44 giây, gồm phần làm việc trên câu thứ tám trước khi ngắt. Thời gian load/copy model 445,83 giây thuộc ô trước và không nằm trong thời gian mỗi câu diagnostic.

## Trace tương tác cho thấy lỗi thiết kế

Câu “Hôm nay cửa hàng bán được bao nhiêu tiền?”, `biz_101`, `as_of_date=2026-02-22`, mất **107,44 giây / chín lượt model** rồi `abstain`.

- Chuẩn bị mất **87,37 giây**. Bốn lượt sinh chuẩn bị tốn 84,99 giây và 2.103 output token.
- Hai lần đọc rulebook đều chọn đúng `week_start=0`, nhưng dẫn câu `Tuần bắt đầu thứ Hai.`. Trong nguồn, nội dung nằm trong bảng Markdown, có dấu `|` và `**`. Kiểm tra trích nguyên văn không chấp nhận cách viết đó và loại toàn bộ policy, kể cả các trường khác.
- Hai đề xuất cho bảng bán hàng tiếp tục bị loại do trích dẫn của thuộc tính “đơn vị tính” hoặc “số lượng quy đổi”. Kết quả cuối của chuẩn bị là `bindings=[]`, `sales_statuses=null`, `line_revenue=false`.
- Phần trả lời dùng thêm năm lượt model, khoảng 20,01 giây. Plan doanh thu hôm nay hợp lý nhưng `calculate` bị chặn vì nguồn chưa ánh xạ. Model truy vấn bảng facts rỗng, thử `unanswerable`, bị từ chối rồi bỏ qua với lý do thiếu dữ liệu.

Kiểm chứng bằng công cụ hiện có, dùng ánh xạ bằng code trên cùng nguồn, tính được doanh thu ngày đó là **13.479.900 đồng**. Đây là lời gọi CPU để chẩn đoán, không phải dự đoán LLM mới. Nó cho thấy nguồn có dữ liệu; sự cố nằm ở việc bản mô tả bị loại rồi trở thành bảng facts rỗng.

Thiết kế mới đã đặt việc tạo JSON dài và sao chép trích dẫn thành điều kiện phải vượt qua trước khi tính toán. Một thuộc tính phụ hoặc khác biệt định dạng có thể làm hỏng cả bản mô tả. Chi phí và mức phụ thuộc này chưa phù hợp với SLM hiện tại. `enable_thinking` vốn đã tắt; phần tốn thời gian là các lượt sinh JSON và sửa lỗi, không phải bật một kênh reasoning ẩn.

## Hướng sửa đề xuất theo thứ tự

### 1. Bỏ ánh xạ toàn bộ bằng SLM khỏi luồng mặc định

Ưu tiên adapter bằng code cho schema, kiểu dữ liệu và các quy tắc đã nhận diện được từ nguồn hiện tại. Cải thiện xử lý CamelCase, mô tả cột trong README và các quan hệ kiểm tra được từ toàn bộ dữ liệu. Giữ các sửa lỗi về đại lượng, thời gian, trạng thái, đơn vị, kiểu đầu ra và nguồn rỗng.

Đây không phải viết đáp án theo ID hoặc tạo metric riêng cho mỗi doanh nghiệp. Vai trò cột và phép toán dùng chung; nguồn không chắc vẫn phải để chưa rõ. Adapter không được âm thầm áp rulebook train cho split khác.

SLM chỉ được gọi bổ sung khi một câu thực sự cần trường còn thiếu, với vài cột/bảng liên quan và tối đa một lần thử bổ sung ban đầu. Không mô tả toàn bộ bảng kho, nhân viên, nhập hàng và danh mục khi câu chỉ hỏi doanh thu ngày.

Trong code hiện tại, `adaptive_preparation` đang điều khiển cả chuẩn bị schema lẫn bắt buộc plan. Vì vậy, không nên chỉ đổi cờ thành `false` rồi coi như đã hoàn thành tối ưu. Cần tách hai việc để tắt chuẩn bị bằng SLM mà vẫn giữ kiểm tra đại lượng và kế hoạch.

### 2. Bỏ các lượt model chỉ để điều khiển công cụ

Với phép tính được hỗ trợ, cho model gửi một yêu cầu gồm plan và cách lấy đầu ra. Code kiểm tra, biên dịch SQL, chạy công cụ và chiếu kết quả thành final. Không cần thêm một lượt model để nói `calculate` và một lượt để nói `final`.

Mục tiêu kiến trúc là **một đến hai lượt model cho câu đơn giản**, chỉ gọi lại khi thiếu thông tin hoặc có lỗi cần sửa. Cần benchmark sau triển khai, không quy đổi mục tiêu này thành cam kết số giây.

### 3. Chấp nhận từng phần và sửa cơ chế căn cứ

Một thuộc tính phụ sai không được xóa các vai trò cột đã hợp lệ. Trường policy chưa rõ chỉ chặn phép tính phụ thuộc nó; chẳng hạn ngày bắt đầu tuần chưa rõ không tự chặn doanh thu hôm nay.

Nếu còn dùng LLM để sửa ánh xạ, code nên đánh số đoạn README/rulebook và cột nguồn để model chọn `column_id`/`evidence_id`. Nếu dùng đoạn trích, cần xử lý định dạng Markdown một cách kiểm soát. Không yêu cầu model chép lại nhiều câu tiếng Việt chỉ để chứng minh đã đọc nguồn.

Giữ bản ánh xạ bằng code đã qua kiểm tra khi đề xuất mới thất bại. Phân biệt `preparation_failed`, `unsupported` và `data_missing`. Không chạy thêm vòng hỏi SQL trên facts rỗng vì chuẩn bị thất bại. Tránh lặp lại việc diễn giải cùng rulebook đã thất bại ở từng doanh nghiệp mới.

### 4. Giảm context và chỉ tối ưu engine sau khi sửa luồng

Prompt trả lời vẫn mang toàn bộ rulebook. Có thể cung cấp policy đã xác nhận cùng các đoạn luật liên quan, giữ nguồn đầy đủ cho công cụ truy xuất và log. Đưa bản mở rộng viết tắt vào gợi ý để SLM giảm việc giải nghĩa ngôn ngữ, giữ nguyên câu gốc cho kiểm tra.

Backend hiện bật `use_cache=True` trong một lần `generate`, chưa có cơ chế giữ/reuse KV giữa các lần gọi trong vòng agent. Prefix caching là hướng thử sau đó, với điều kiện phiên bản hỗ trợ checkpoint và phần cứng đang dùng. Nó giảm chi phí xử lý prefix, không giảm trực tiếp thời gian sinh hàng trăm token mới. [Tài liệu vLLM](https://docs.vllm.ai/en/v0.10.1/features/automatic_prefix_caching.html), [Transformers cache](https://huggingface.co/docs/transformers/v5.15.0/kv_cache).

Không ưu tiên tăng timeout, hạ token cap một cách cơ học hay đổi quantization để che vòng gọi quá dài. Các output chuẩn bị trong trace đều kết thúc EOS; hạ cap thấp hơn JSON đang cần có thể làm tăng lỗi parse.

## Phương án thực dụng cho lượt kế tiếp

Ưu tiên **code trước, SLM sửa phần thiếu theo nhu cầu**, cùng luồng một lần lập plan rồi code tự thực thi/chiếu đáp án. Có thể tắt hẳn sửa schema bằng SLM nếu thử nghiệm cho thấy phần đó không tăng điểm đủ để bù độ trễ; đổi lại, phải chấp nhận bỏ qua nhiều hơn ở schema thực sự chưa hỗ trợ.

Sau sửa, chạy lại đúng bảy câu này và đo riêng câu đầu doanh nghiệp với câu dùng lại cache. So sánh thời gian, số lượt model và đúng/sai/bỏ qua. Chỉ mở rộng sang 100 câu khi đường mới có kết quả thực tế tốt hơn. Chưa có phép đo tốc độ hoặc PA cho bất kỳ phương án sửa nào trong báo cáo này.

Evidence đã xuất tại [analysis.json](../outputs/diagnosis-2026-11-05/round4-notebook-analysis/analysis.json) và [trace tương tác](../outputs/diagnosis-2026-11-05/round4-notebook-analysis/interactive-biz101.json). Notebook gốc và code inference được giữ nguyên trong lần phân tích này.
