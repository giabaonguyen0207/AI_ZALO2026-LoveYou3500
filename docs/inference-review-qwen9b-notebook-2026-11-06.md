# Qwen 3.5 9B vẫn abstain trong notebook — 06/11/2026

**Nguyên nhân trực tiếp: Colab đang dùng agent/backend/prompt trước bản sửa EOS.** Notebook mới có ô diagnostic mới, nhưng object agent được nạp vẫn là pipeline 4. Chưa có kết quả chạy Qwen 9B bằng đầy đủ bản sửa pipeline 5 trong notebook nhận được; chưa thể quy lỗi cho chất lượng Qwen hoặc kết luận hệ thống chỉ hoạt động với Gemma.

Nguồn: output đang lưu trong `notebooks/DataMate_Colab_Inference.ipynb`, SHA256 `5d0d98637f89b2499d750c0b57e3c67b89c772797db68c320ff659bb12970df4`. Đợt này chỉ đọc notebook, trích log và chạy probe parser/period trên CPU; không sửa runtime, notebook hoặc source ZIP.

## Bằng chứng từ chính notebook

| Quan sát | Output thực tế | Bản sửa cần có |
|---|---|---|
| Ô load, execution 20 | `Pipeline: 4` | `Pipeline: 5` |
| Thông tin token dừng | `Generation tokens: None` | Dict có effective EOS `[248044, 248046]` |
| Prompt system của câu thử | Khớp nguyên văn prompt trong ZIP trước sửa, 6.371 ký tự | Prompt mới có ví dụ plan hợp lệ |
| Kết thúc mỗi generation | Last token 248044; raw chứa cả im_end và endoftext | Có thể dừng ngay ở im_end 248046 |
| Batch execution 22 | `Agent trong RAM là bản cũ...` | Chấp nhận agent/backend mới rồi chạy bảy câu |

Notebook có **một câu thử tương tác**, không phải bảy câu benchmark đã chạy: doanh nghiệp `biz_101`, `as_of_date=2026-02-22`, câu hỏi “Hôm nay cửa hàng bán được bao nhiêu tiền?”. Hai lần model sinh cùng action, cả hai bị `action_parse`; không action nào được nhận để thực thi SQL/plan trả lời. Chính agent xuất abstain sau khi dừng lặp; Qwen không sinh action final abstain ở đây.

Raw kết thúc:

```text
{...JSON plan đã đóng đủ...}<|im_end|>
<|endoftext|>
```

Đây là đúng trường hợp parser cũ chỉ bỏ một marker cuối rồi từ chối JSON vì còn marker dư. Parser hiện tại đọc được cả hai raw này. Hash raw khớp generation log ở cả hai bước.

Gói ZIP local được bàn giao có SHA256 **`7e584c8ad688f85e92ac1f7a27e8647ad6436c7d196a33109e93a0181a8dea15`**; kiểm tra CRC và nội dung xác nhận runner pipeline 5, backend protocol 3, prompt/parser mới. Các file runtime trong ZIP khớp source hiện tại. Notebook không cho biết đủ để phân biệt **ZIP trên Drive còn cũ** hay **Python đã giữ module cũ trong RAM sau khi thay file**. Có thể đồng thời có cả hai; không quy thành lỗi thao tác cụ thể khi chưa đọc live Colab.

## Plan Qwen có vấn đề gì ngoài lỗi parser?

Model chọn `measure="revenue"`, `operation="sum"`, `type="number"`, `population="sales"` và ngày 22/02/2026. Các trường đó thể hiện cách hiểu phù hợp với câu hỏi này. Model thêm filter ngày và trạng thái `completed/shipping`; đây là nhãn chuẩn có trong hệ thống.

Tuy nhiên, model dùng kỳ:

```json
{"kind":"day","date":"2026-02-22"}
```

`day` chưa thuộc giao diện period của hệ thống. Probe CPU xác nhận `resolve_period()` trả lỗi “period kind chưa hỗ trợ...”. Với câu hỏi này, biểu diễn được hỗ trợ là `{"kind":"today"}` hoặc range có hai ngày bằng nhau. Probe `today` giải đúng từ `as_of_date` thành 22/02/2026.

Đây là lỗi tuân thủ giao diện của output model dưới **prompt cũ**. Khi parser còn lỗi, Qwen chỉ nhận thông báo JSON chung nên không thấy lỗi period để sửa. Không tự đổi plan này thành câu trả lời đúng, không coi probe sửa kỳ là một lượt inference mới hoặc bằng chứng PA cải thiện.

## Có phải hệ thống bị gắn với Gemma?

Các lượt phát triển/đo chất lượng trước tập trung vào Gemma, nên chưa đủ benchmark để khẳng định prompt và ngân sách hiện tại phù hợp như nhau với Qwen. Đây là giới hạn kiểm chứng thực nghiệm cần thừa nhận.

Tuy vậy, đường chạy đang quan sát dùng key/thư mục model `qwen35-9b`, loader/tokenizer của checkpoint đó và cùng agent/plan/SQL dùng chung. Không có nhánh quyết định đáp án theo tên Gemma/Qwen trong agent. Nhánh assistant đặc biệt phục vụ chế độ speculative của Gemma; notebook này chạy single. Bộ dữ liệu seed dành cho fine-tune Gemma không được nạp thành adapter hoặc đáp án cho Qwen trong ô inference này.

Vì vậy, **lần abstain này là bằng chứng môi trường đang chạy bản cũ và lỗi giao tiếp, chưa phải phép so sánh năng lực Qwen–Gemma**. Sau khi nạp đúng bản sửa, vẫn cần đo tỷ lệ plan hợp lệ, sửa lỗi thành công và chất lượng tính toán; một lỗi `kind="day"` trong một câu không đủ kết luận model 9B hoạt động kém nói chung.

Về thời gian: câu tương tác mất **5,905 giây**, hai lượt generation lần lượt **2,9396 và 2,9347 giây**, tổng 206 token đầu ra. Ô load mất **326,77 giây**, trong đó **312,76 giây** chép cache 18 GiB từ Drive và **12,54 giây** kiểm tra/nạp. Không gộp thời gian chép model vào thời gian sinh đáp án, không suy tốc độ một câu có dữ liệu cache thành benchmark chung.

## Việc cần làm trước khi so model

1. Đưa đúng `outputs/datamate-colab-source.zip` local vào đường dẫn Drive **`MyDrive/DataMate/datamate-colab-source.zip`**. Có thể đối chiếu SHA256 với số trên để loại khả năng đang chọn nhầm bản ZIP.
2. Khởi động lại kernel trong cùng runtime rồi chạy các ô từ đầu. Việc chỉ đổi MODEL_KEY, chạy lại pip install hoặc tạo lại object agent không bảo đảm module Python đã import được thay mới. Giữ cache `/content/models` để dùng lại bản trọng số đã chép.
3. Trước khi hỏi hoặc chạy batch, kiểm tra đủ **pipeline 5**, **backend protocol 3**, `generation_tokens` có effective EOS **`[248044, 248046]`**. Nếu còn pipeline 4 hoặc `None`, chưa chạy bảy câu.
4. Dùng cùng mẫu bảy câu và profile đã bàn giao; sau khi xác nhận giao tiếp hoạt động mới so với Gemma hoặc mở rộng 100 câu. Giữ raw output/SQL và phân loại rõ lỗi parse, lỗi plan, lỗi tính toán.

Một thiếu sót của tiện ích Colab hiện tại là chặn agent cũ ở batch nhưng chưa chặn sớm tại ô load/hỏi tương tác; vì thế vẫn có thể tốn thời gian copy/load rồi mới phát hiện đang dùng code cũ. Nên bổ sung kiểm tra phiên bản trên đĩa và trong RAM **trước khi chép/nạp trọng số**, thay vì nới validator để che lỗi. Đợt phân tích này chưa triển khai thay đổi đó.

Bằng chứng trích xuất và script tái hiện: `outputs/diagnosis-2026-11-06/qwen9b-notebook-review/analysis.json`, `interactive-extracted.json`, `analyze_notebook.py`. Không tải/chạy checkpoint mới và không sửa gói Gemma SFT.
