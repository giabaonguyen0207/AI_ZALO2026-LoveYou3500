**Phân tích độ trễ Qwen 3.5 upscaled 14B V2 — nhận ngày 06/11/2026**

Nguồn: `outputs/diagnosis-2026-11-05/qwen35-upscaled14b-v2-random100-Nov5th/`.

**Kết luận chính:** hệ thống đang vướng lỗi tương thích giữa token dừng của checkpoint, parser action và hợp đồng plan. Thời gian tăng chủ yếu vì các lần gọi lại sau lỗi parse và model sinh tiếp hội thoại giả. Chưa có cơ sở quy nguyên nhân chính cho thiếu VRAM hoặc riêng kích thước 14B.

Người dùng nhắc bảy câu, nhưng bundle nhận được có **10 câu hoàn tất, cả 10 abstain**, trong kế hoạch 100 câu. `requested_cumulative_limit=10`, `completed_questions=10`, còn 90 câu chưa chạy. `status="complete"` ở đây là hoàn thành giới hạn 10 câu yêu cầu, không phải đã chạy hết 100. Báo cáo đối chiếu cả bảy câu đầu và toàn bộ 10 câu đã nhận.

**Đối chiếu đúng cùng các câu**

| Chỉ số trên bảy câu đầu | Qwen upscaled 14B V2 | Gemma fast v5 | Gemma baseline v3 |
|---|---:|---:|---:|
| Đúng / sai / bỏ qua | 0 / 0 / 7 | 5 / 0 / 2 | 6 / 1 / 0 |
| Tổng thời gian, giây | 362,29 | 225,23 | 118,46 |
| Trung bình, giây/câu | **51,76** | 32,18 | 16,92 |
| Lần gọi model | 25 | 17 | 17 |
| Token đầu ra | **7.848** | 3.028 | 561 |
| Thời gian generation calls, giây | 295,72 | 158,16 | 52,74 |
| Phần thời gian còn lại, giây | 66,56 | 67,08 | 65,72 |

Qwen chậm hơn Gemma fast v5 khoảng **61%** trên đúng bảy câu này. Không so bảy câu Qwen với trung bình toàn bộ 100 câu Gemma.

| Chỉ số trên 10 câu đã nhận | Qwen upscaled 14B V2 | Gemma fast v5 cùng 10 ID |
|---|---:|---:|
| Đúng / sai / bỏ qua | 0 / 0 / 10 | 7 / 0 / 3 |
| Tổng thời gian, giây | **490,56** | 260,37 |
| Trung bình, giây/câu | **49,06** | 26,04 |
| Lần gọi model | **35** | 21 |
| Token đầu vào | 321.900 | 204.955 |
| Token đầu ra | **10.874** | 3.359 |
| Thời gian generation calls, giây | **410,93** | 179,94 |
| Phần thời gian còn lại, giây | **79,63** | 80,44 |

Thời gian không gồm load model. Số liệu là local development trên TRAIN, không phải điểm thi hoặc phép đo chất lượng tổng quát của hai model.

**1. Cấu hình dừng chưa nhận token kết thúc lượt trả lời**

Token map của đúng checkpoint đã pin:

| Token | ID |
|---|---:|
| `<\|endoftext\|>` | 248044 |
| `<\|im_start\|>` | 248045 |
| `<\|im_end\|>` | **248046** |

`tokenizer_config.json` khai báo EOS là `<|im_end|>`, nhưng `generation_config.json` chỉ có:

```json
{"eos_token_id": 248044}
```

Runtime thật cũng ghi `eos_token_id=248044`; backend hiện để `generate()` dùng cấu hình đó, chưa bổ sung token kết thúc lượt từ tokenizer. Khi model phát `<|im_end|>`, lượt generate chưa có điều kiện dừng tương ứng nên có thể sinh tiếp newline, endoftext hoặc một lượt user/assistant khác.

Đã kiểm tra token map/config local khớp blob/hash trong metadata của revision `f4fe3d2ad8f2e14799624cdbafa5d46219bc1fb2`, không lấy ID từ một model khác. Nguồn checkpoint: [generation config](https://huggingface.co/Pinkstackorg/Qwen-3.5-upscaled-14B-V2-noft/blob/f4fe3d2ad8f2e14799624cdbafa5d46219bc1fb2/generation_config.json), [tokenizer config](https://huggingface.co/Pinkstackorg/Qwen-3.5-upscaled-14B-V2-noft/blob/f4fe3d2ad8f2e14799624cdbafa5d46219bc1fb2/tokenizer_config.json).

**2. Parser chỉ bỏ một marker cuối nên từ chối cả JSON đã đóng đủ**

**22/35 output** có dạng kết thúc thực tế:

```text
{"action":"plan", ...}<|im_end|>
<|endoftext|>
```

Trong `src/datamate/agent/protocol.py`, regex nhận cả hai loại marker nhưng chỉ thay một lần ở cuối chuỗi. Nó bỏ `<|endoftext|>`, để lại `<|im_end|>` phía sau JSON; `json.loads()` vì thế báo có nội dung dư. Model nhận thông báo chung “phải trả đúng một JSON object”, dù lỗi một phần nằm ở cách xử lý ranh giới của hệ thống.

Probe tối thiểu trên CPU tái hiện chính xác:

- Một JSON hợp lệ + một `<|im_end|>`: parser nhận.
- Cùng JSON + `<|im_end|>\n<|endoftext|>`: parser hiện từ chối.
- Chỉ loại các marker kết thúc liên tiếp ở cuối: parser nhận.
- Hai JSON action hoặc có thêm lượt user giả: vẫn từ chối trong biến thể chẩn đoán an toàn.

Trên log thật, **35/35 output đều lỗi action_parse**, không action nào qua được parser để thực thi plan/SQL trả lời. Các câu hỏi vẫn có bước đọc/chuẩn bị dữ liệu và kiểm tra ban đầu bằng code; điều bị chặn là hành động trả lời do model đề xuất.

Việc tích hợp trước đã kiểm tra cấu trúc, shape tensor, tokenizer và một fixture có một EOS, nhưng **bỏ sót đối chiếu EOS của generation với EOS tokenizer và trường hợp hai marker liên tiếp**. Đây là thiếu sót ở loader/parser cần sửa, không phải lỗi thao tác của người dùng.

**3. Các lần sửa lỗi sinh tiếp hội thoại giả và tiêu hết ngân sách**

**13/35 output** chứa `<|im_start|>user` do model sinh, sau đó có `tool_result`, lỗi giả và lượt assistant khác. Những đoạn nằm trong `raw` này không phải lần gọi công cụ thật: trace chỉ ghi lỗi parse, không có action được nhận. Không được lấy các số, profile hoặc kết quả SQL nằm trong hội thoại model tự sinh làm bằng chứng dữ liệu.

35 lần gọi có:

- 22 lần dừng ở endoftext, nhưng các output này vẫn lỗi parser.
- **9 lần chạm 768 token**, tổng **6.912 token**, mất **241,40 giây**; trung bình khoảng **26,82 giây/lần**.
- 4 lần `time_or_other` ở lượt cuối của bốn câu kéo tới khoảng **90 giây/câu**.

Mười lần gọi đầu tiên của 10 câu chỉ mất **51,47 giây**. Các lần gọi lại mất **359,46/410,93 giây generation**, tức **87,47%**. Đây là phần chi phí nổi bật nhất. Các lần gọi sau còn gửi lại prompt khoảng 8–11K token cùng feedback; `use_cache=True` trong một lượt generate không có nghĩa agent đang dùng lại toàn bộ prefix cache giữa các lượt gọi.

Sáu câu dừng sớm vì lặp; bốn câu hết ngân sách thời gian. `step_or_time_budget=10` trong summary không có nghĩa cả 10 timeout: flag hiện tìm chữ “ngân sách”, nên tính cả reason “dừng để tiết kiệm ngân sách” của sáu câu lặp. Không câu nào trong run này đi tới sáu lượt model; chúng dừng theo lặp hoặc thời gian trước đó.

**4. Sửa ranh giới sẽ chưa giải quyết hết hợp đồng plan**

Tôi chỉ chạy probe parser/validator trên CPU, không sửa runtime và không chạy thêm model:

| Biến thể chẩn đoán trên 35 output đã lưu | Parse được action | Plan qua kiểm tra cấu trúc |
|---|---:|---:|
| Parser hiện tại, raw nguyên gốc | 0 | 0 |
| Bỏ riêng chuỗi marker liên tiếp ở cuối | **21** | **0** |
| Giả lập generate đã dừng ở `<\|im_end\|>` đầu tiên | **29** | **3** |

Dòng cuối chỉ dùng để phân tích tác động của dừng đúng lượt; đây không phải cách cắt text để đem vào production. Các output được cắt thuộc lịch sử retry của run lỗi; khi sửa EOS, lịch sử và các lần sinh tiếp sẽ thay đổi. Không quy số trên thành điểm PA hoặc số câu sẽ cứu được.

Các lỗi plan còn lại nhìn thấy trực tiếp:

- `rank` lồng trong plan thay vì `operation="rank"` cùng các trường được hỗ trợ; có câu top khách lại chọn `sum/number`.
- Dùng `period` thay vì `periods`, thêm `order`, `query_id` hoặc `denominator_filters` vào action plan.
- Dùng `between`, `<=` làm `op`, trong khi giao diện hiện hỗ trợ các giá trị như `ge`, `le`.
- Thiếu `evidence_id`/evidence ở plan hoặc ở từng filter.
- Câu đếm đơn huỷ chọn quantity và lọc `order_id` bằng nhãn cancelled, sai cả đại lượng lẫn cột.
- Câu tăng trưởng sau tăng giá chọn ratio thay vì growth với hai kỳ.

Vì lỗi parse che phía trước, model hiện chỉ nhận feedback định dạng chung, chưa nhận được lỗi cụ thể về các trường plan đó. Sau khi sửa EOS/parser, cần đo xem model có sửa được plan với feedback đúng tầng hay không.

**5. Những nguyên nhân khác đã kiểm tra**

- Runtime là GPU **NVIDIA RTX PRO 6000 Blackwell Server Edition ~94,97 GiB**, BF16, `quantization="none"`, tham số trên `cuda:0`; cùng GPU/software versions được ghi ở run Gemma. Không thấy bằng chứng CPU/disk offload, OOM hoặc giảm sang GPU nhỏ.
- `preparation_seconds=0`, `preparation_model_calls=0`, `schema_repair_attempts=0`. Đây không phải lỗi quay lại bước SLM đọc toàn schema/rulebook nặng như bản trước fast v5.
- Prompt tail có khối `<think>...</think>` rỗng đúng cấu hình `enable_thinking=False`. Các lượt đầu sinh JSON; nguyên nhân nổi bật là dừng/parse và hội thoại sinh dư, không phải bằng chứng thinking đã được bật lại.
- Phần thời gian ngoài generation của Qwen và Gemma gần như bằng nhau trên cùng 10 câu: **79,63 so với 80,44 giây**. Phần này còn gồm chuẩn bị facts, tokenize, điều phối và I/O; log chưa tách đủ để gán toàn bộ cho một bước. Gần hết chênh lệch tổng thời gian **230,19 giây** nằm trong generation calls.
- Qwen có 56 tầng và nhiều tham số hơn, nhưng run này chưa cô lập được chi phí phần cứng của riêng kiến trúc. Không có telemetry tách prefill/decode, GPU utilization hoặc kernel fallback. Không kết luận cần đổi GPU/engine từ số liệu này.
- Runtime dùng tokenizer class `Qwen3_5Tokenizer` ở Transformers 5.22.0, còn kiểm tra offline trước dùng `TokenizersBackend` ở 5.15.0. Bảy độ dài prompt đầu ghi trong runtime khớp các số đã đo offline. Khác tên class không tự chứng minh tokenizer sai; bất nhất EOS trong cấu hình đã có bằng chứng riêng.

**Thứ tự xử lý đề xuất**

1. **Kiểm tra và hợp nhất stop ID hợp lệ lúc load/generate.** Đối chiếu EOS từ tokenizer với các stop ID của checkpoint; trong trường hợp đã xác minh này cần dừng được ở cả endoftext và im_end. Transformers cho phép danh sách EOS. Ghi effective stop IDs, tokenizer EOS/pad và lớp tokenizer vào runtime/generation log. Thực hiện trên cấu hình trong bộ nhớ; giữ nguyên các file checkpoint đã pin và seal. Xem [GenerationConfig của Transformers](https://huggingface.co/docs/transformers/main_classes/text_generation).
2. **Sửa parser ranh giới cuối.** Cho phép chuỗi marker kết thúc liên tiếp đã biết, vẫn giữ raw nguyên gốc. Tiếp tục từ chối nhiều action/hội thoại giả; không tìm JSON đầu tiên rồi âm thầm bỏ phần còn lại, không chỉ xoá mọi special token trong toàn output.
3. **Làm rõ contract plan và feedback.** Ví dụ JSON thực sự hợp lệ, kiểu trường rõ, đủ evidence cho filter, lỗi nêu đúng trường và kiểu cần sửa. Giữ các kiểm tra về nguồn/điều kiện; không bỏ validator để làm hết abstain. Giới hạn retry lỗi định dạng dựa trên nguyên nhân, vì raw thay đổi liên tục có thể né cơ chế dừng lặp hiện tại.
4. **Chạy lại cùng bảy câu, cùng profile trước.** Theo dõi tỷ lệ parse, loại lỗi plan, token sau ranh giới kết thúc, số lần retry, thời gian và PA. Sau khi giao diện hoạt động mới đo được tương đối chất lượng và tốc độ checkpoint này. Không báo 21/35 hoặc 29/35 probe là kết quả sau sửa trên GPU.

Chưa cần thêm bước SLM ánh xạ toàn doanh nghiệp cho vấn đề này. Bộ fine-tune hiện tại vẫn chỉ dành cho Gemma; lượt Qwen mới không được tự đưa vào SFT.

**Thời gian từng câu Qwen**

| Thứ tự | ID | Giây | Calls | Token đầu ra | Kết thúc |
|---:|---|---:|---:|---:|---|
| 1 | q_1fa63c38c946 | 90,04 | 5 | 2.229 | Hết thời gian |
| 2 | q_3234281bf31d | 90,02 | 4 | 2.226 | Hết thời gian |
| 3 | q_37ecac46670e | 24,13 | 3 | 216 | Lặp lỗi parse |
| 4 | q_3811bb0ccc39 | 24,14 | 3 | 336 | Lặp lỗi parse |
| 5 | q_4362460c082a | 30,32 | 3 | 216 | Lặp lỗi parse |
| 6 | q_d96f45ce61be | 13,60 | 3 | 240 | Lặp lỗi parse |
| 7 | q_d767d5ca414e | 90,03 | 4 | 2.385 | Hết thời gian |
| 8 | q_2ea34572153a | 28,79 | 3 | 370 | Lặp lỗi parse |
| 9 | q_1295f18bd2fe | 90,03 | 5 | 2.496 | Hết thời gian |
| 10 | q_b5c41bf00e07 | 9,46 | 2 | 160 | Lặp lỗi parse |

**Bằng chứng và phạm vi thực hiện**

Đã kiểm tra identity manifest, TRAIN fingerprint, ID/nội dung câu hỏi, submission, chấm lại nhãn local, CRC/nội dung ZIP, source hash và SHA256 của **35 raw outputs**. Các source hash được manifest ghi nhận đều khớp local hiện tại. Bundle có đúng 100 câu kế hoạch, thực thi 10 câu đầu; đối chiếu sử dụng đúng cùng ID/as_of_date của Gemma.

Artifact tại `outputs/diagnosis-2026-11-06/qwen14b-review/`: `audit.json` có số liệu/kiểm tra/config và fixture, `questions.json` có từng câu và raw đầu, `parser-probes.json` có raw và kết quả từng biến thể. Script chạy lại:

```powershell
.\.venv\Scripts\python.exe -X utf8 outputs/diagnosis-2026-11-06/qwen14b-review/analyze.py
```

Đợt này chỉ phân tích, tạo probe CPU và tài liệu; **chưa sửa loader/parser/prompt, chưa thay source ZIP, chưa tải/chạy model mới và chưa đo kết quả sau sửa**.
