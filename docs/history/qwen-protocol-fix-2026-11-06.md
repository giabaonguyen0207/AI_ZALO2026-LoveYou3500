# Sửa giao tiếp Qwen 3.5 9B và upscaled 14B — 06/11/2026

**Kết quả thực tế nhận sau bàn giao:** Qwen 9B đã hoàn tất 100 câu với đúng bản sửa, PA 17%, trung bình 9,64 giây/câu. Các giới hạn “chưa chạy GPU” bên dưới mô tả thời điểm triển khai; xem [báo cáo kết quả mới và các lỗi còn lại](inference-review-qwen9b-100-2026-11-06.md). Bản 14B chưa có lượt chạy sau sửa được nhận.

Đã triển khai các sửa lỗi EOS, parser, hướng dẫn plan và giới hạn retry sau báo cáo Qwen. Ưu tiên checkpoint **Qwen/Qwen3.5-9B chính thức**; bản Pinkstack vẫn dùng chung sửa lỗi, không bổ sung diễn giải kiến trúc dựa trên tài liệu chưa rõ. Không đổi revision hoặc file checkpoint đã seal.

Đã kiểm chứng cơ chế giao tiếp offline và **187 kiểm thử phần mềm đều qua**. Chưa chạy lại checkpoint đầy đủ trên GPU, vì vậy chưa xác nhận tốc độ, PA hoặc số câu hết abstain sau sửa. Việc model hiểu đúng câu hỏi/rulebook vẫn cần đo riêng.

## Những thay đổi đã thực hiện

- Backend hợp nhất EOS đang có với EOS tokenizer và token kết thúc lượt có trong chat template. Token kết thúc lượt được kiểm tra ID nằm trong vocabulary, là special token và mã hóa thành đúng một token. Không gán cố định số ID token trong code, không quyết định đáp án theo business/question ID. Các stop ID được truyền tường minh vào `generate()`, không sửa `generation_config.json` trên đĩa. Pad ID bằng 0 vẫn được giữ đúng cho Gemma.
- Parser bỏ được **chuỗi marker kết thúc liên tiếp ở cuối**, giữ nguyên raw log. Hai action, văn bản dư hoặc user/tool giả sau action vẫn bị từ chối; không cắt lấy JSON đầu tiên để chấp nhận phần còn lại.
- Prompt dùng enum riêng và năm ví dụ plan có JSON hợp lệ: tổng, xếp hạng khách, tỷ lệ đơn hủy, tăng trưởng, số lượng có điều kiện. Validator kiểm tra các ví dụ bằng cùng code inference. Ví dụ phải được model điều chỉnh theo schema/rulebook/câu hỏi thực tế.
- Lỗi plan chỉ rõ đường dẫn như `plan.filters[0].op`, `plan.periods[0]`, `plan.group_by` và các giá trị/kiểu được hỗ trợ. Không tự đổi `period` sang `periods`, bỏ filter, thay đại lượng hoặc tắt kiểm tra để lấy đáp án. Model nhận feedback để sửa chính action của nó.
- Hai lỗi parse liên tiếp sẽ dừng retry với mặc định `max_repeated_actions=2`, kể cả raw thay đổi. Parse thành công sẽ đặt lại bộ đếm. Điều này giới hạn chi phí định dạng; vẫn có thể abstain nếu model không đáp ứng giao diện.
- `configs/qwen35-9b.json` dùng cùng profile nhanh với 14B/Colab: 6 bước, 768 token/lượt, 90 giây/câu, context 16.384, code chuẩn bị trước, plan bắt buộc và tự tính/chiếu khi được kiểm tra. Không thêm bước SLM đọc toàn doanh nghiệp.
- Runtime ghi cấu hình EOS gốc và `generation_tokens` thực dùng; từng generation ghi effective EOS/pad, last token và stop reason. Pipeline code là **5**, backend protocol **3**, diagnostic **7**. Batch chặn agent/backend phiên bản cũ còn trong RAM.

Đây là sửa giao diện và kiểm tra dùng chung, không phải quy tắc quyết định đáp án theo ID, doanh nghiệp hoặc câu hỏi cụ thể. Các kiểm tra nguồn, dữ liệu thiếu, trạng thái, phép nối, SQL chỉ đọc và type vẫn áp dụng. Những lỗi nghiệp vụ khác trong [báo cáo Gemma fast v5](inference-review-fast-v5-2026-11-06.md) chưa được giải quyết đầy đủ bởi đợt sửa này.

## Kết quả kiểm chứng

| Checkpoint đã pin | EOS gốc trong config | EOS thực dùng | Pad thực dùng |
|---|---|---|---:|
| Qwen 3.5 9B | `[248044]` | **`[248044, 248046]`** | 248044 |
| Qwen upscaled 14B V2 | `[248044]` | **`[248044, 248046]`** | 248044 |
| Gemma 4 12B | `[1, 106]` | `[1, 106]` | 0 |

Với Qwen, 248046 là `<|im_end|>`, 248044 là `<|endoftext|>`. Các số trong bảng được đọc từ tokenizer/config thực tế, không gán cố định trong runtime. Tokenizer/config 9B được đối chiếu với manifest model đã seal. SHA256 các file tokenizer/config kiểm tra không thay đổi trong quá trình xác minh.

Phép thử dùng **Qwen3.5 rất nhỏ, trọng số ngẫu nhiên trên CPU, ép chuỗi token đầu ra** và gọi chính backend/`generate()` của Transformers. Chuỗi thử có JSON, `<|im_end|>`, rồi một user giả. Với tokenizer 9B và 14B, generation đều dừng đúng ở `<|im_end|>`, JSON được parse, phần user giả không được sinh tiếp. Đây là kiểm tra tích hợp stop token; logits ép sẵn không chứng minh năng lực của SLM 9B/14B.

Replay nguyên vẹn 35 raw output cũ: **21 parse được, 14 bị từ chối**, trong đó cả **13 output có hội thoại giả** đều bị từ chối. Hash raw khớp log. Không sửa plan hoặc chấm những action replay này thành đáp án mới.

Bảy prompt hiện tại cùng ID/thứ tự mẫu 100 cũ được dựng lại trên TRAIN đã đăng ký và đo bằng tokenizer thật:

- Qwen 9B/14B: **8.668–11.799 token đầu vào**; thêm 768 token đầu ra vẫn dưới 16.384.
- Gemma: **9.153–12.548 token đầu vào**; cùng giới hạn vẫn vừa.
- Hướng dẫn/ví dụ mới tăng khoảng **744 token Qwen** trên mỗi prompt trong bảy câu này. Đây là chi phí prefill cần cân đối với lợi ích giảm lỗi/retry trên GPU; chưa đo lợi ích tốc độ sau sửa. Không suy kiểm tra bảy prompt thành bảo đảm mọi lịch sử nhiều lượt đều vừa context.

Toàn bộ pytest: **187 passed**, một cảnh báo deprecation của thư viện FastAPI/Starlette. TestClient cần quyền loopback local trên Windows; ca API và toàn bộ suite đã chạy thành công khi cho phép kết nối nội bộ. Không gọi dịch vụ bên ngoài khi kiểm thử.

Bằng chứng: `outputs/verification-qwen-protocol-2026-11-06/verification.json`, `pytest.xml` và `release-verification.json`. Chạy lại kiểm tra offline từ workspace có tokenizer cache:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tools.verify_qwen_protocol
```

Transformers hỗ trợ truyền nhiều EOS trong generation; xem [tài liệu GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation). Nguồn để quyết định các ID ở đợt sửa này là artifact local đã pin, không phải suy từ tên model hoặc tài liệu Pinkstack chưa rõ.

## Chạy lại trên Colab Pro

1. Thay file Drive `DataMate/datamate-colab-source.zip` bằng **`outputs/datamate-colab-source.zip` mới**. Mở lại **`notebooks/DataMate_Colab_Inference.ipynb` đã cập nhật**; notebook có ô diagnostic nhúng nên cần cập nhật cả notebook và ZIP.
2. Khởi động lại kernel/runtime, ở ô cấu hình đầu đặt `MODEL_KEY = "qwen35-9b"` để thử model chính thức. Notebook giữ lựa chọn MODEL_KEY và QUANTIZATION đã lưu của người dùng; cần sửa ô đầu nếu đang chọn Pinkstack. Muốn thử lại 14B thì dùng `qwen35-upscaled14b-v2`. `MODEL_SOURCE_DIR` được tính từ key tại ô này.
3. Chạy từ đầu đến ô load. Kiểm tra dòng `Pipeline: 5` và `Generation tokens` có effective EOS **`[248044, 248046]`**. Không cần tải lại trọng số từ Internet nếu bản đã seal trên Drive còn đầy đủ; ô load vẫn kiểm tra hash bản local trước khi dùng.
4. Ô batch đã đặt **`RUN_LIMIT=7`** và suffix mới **`random100-protocol-fix-Nov6th-01`**, giữ sample 100/seed 20261103. Không resume thư mục lượt trước sửa vì source hash khác. `RUN_NAME` có tiền tố model, tách 9B và 14B.
5. Xem lỗi parse/plan, stop reason/last token, calls, token, thời gian và kết quả trên bảy câu. Sau đó đặt `RUN_LIMIT=None` để resume cùng lượt đến 100. `AUTO_DISCONNECT=True` đang được giữ theo lựa chọn đã lưu; muốn tiếp tục trong cùng runtime thì đổi thành False trước khi chạy.

Output lịch sử trong notebook được giữ lại và có bản sao trước sửa trong `outputs/notebook-backups/`; chúng không phải số đo bản mới. Gói source trước sửa được giữ ở `outputs/datamate-colab-source-before-qwen-protocol-2026-11-06.zip`.

Bộ data và source **Gemma SFT seed 72+100 giữ nguyên**. Khi fine-tune seed đó, tiếp tục dùng cặp `seed72_100_gemma_v1.zip` + `datamate-colab-gemma-seed-source.zip` theo [hướng dẫn Gemma](gemma-seed72-100-colab-pro.md), vì manifest khóa source/prompt đã chuẩn bị. Gói inference mới có prompt khác; đợt này chưa tái tạo dataset SFT hay train thêm Qwen.
