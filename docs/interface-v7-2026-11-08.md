# Sửa giao tiếp và kiểm tra tham số — pipeline 7, 08/11/2026

> **Trạng thái hiện tại:** v7 đã được hoàn tác theo yêu cầu người dùng ngày 08/11/2026. Tài liệu này giữ làm lịch sử; bản đang dùng là pipeline 6 theo `rollback-pipeline6-2026-11-08.md`.

Người dùng duyệt sửa các lỗi thuộc hệ thống sau lượt Qwen pipeline 6, để chạy lại cùng 100 câu bằng Qwen 9B và Gemma 12B. Đợt này không fine-tune, không thêm model, không sửa nhãn/dữ liệu và không thêm logic trả lời theo question/business ID. Các lỗi model chọn sai đại lượng, kỳ, nhóm hoặc thực thể không được thay bằng recipe theo từng câu.

## Những phần đã sửa

1. **Chốt bằng tham chiếu kết quả.** Model có thể gửi `{"action":"final","query_id":"r1"}`. Code lấy type từ plan/công cụ đã kiểm tra và giữ phép chiếu đã được chấp nhận, kể cả SQL nhiều cột có projection rõ. Các phép native growth/ratio/count/day_count/boolean/rank trả alias ổn định `value`. Không yêu cầu model chép tên biểu thức SQL hoặc số vào columns.
2. **Tương thích final cũ có kiểm soát.** Với query đầy đủ chỉ một cột và type scalar/list đã biết, một columns đơn ghi nhầm giá trị được chuẩn hóa về phép chiếu duy nhất; raw action và `projection_normalized` vẫn có trong log. Table giữ kiểm tra tên cột, query nhiều cột vẫn cần phép chiếu hợp lệ. Trường answer do model chép sai vẫn bị từ chối. Không lấy answer của model để thay số công cụ.
3. **Kiểu tham số và khóa.** Số/boolean được kiểm tra theo cột facts hoặc role typed; ví dụ prescription `"1"` thành integer 1. ID không bị parse như số tiền; chuỗi có số 0 đầu được giữ. Khóa product/customer phải thuộc domain khóa thật, kể cả ở nhánh debt_flow, trước khi phép cộng rỗng về 0. Tên khách phải qua lookup/entities, không được lặng lẽ dùng làm mã. Khách có thật nhưng không phát sinh vẫn có thể trả 0.
4. **Sửa biểu diễn API mà giữ nghĩa.** this_month/previous_month_to_date có thể ghi lặp month/year nếu khớp kỳ đã giải; mâu thuẫn vẫn báo lỗi. day/date + date ISO là bí danh ngày đơn. Timestamp có dấu cách được chuẩn hóa về ISO T; chuỗi chỉ giờ bị từ chối với hướng dẫn dùng shift, không tự đoán ca. Filter hủy/trả trên nguồn sales bị báo rõ nguồn đã lọc, thay vì báo như không có trạng thái. Exact product equality của value/price có thể được giải thành key duy nhất mà không yêu cầu model lặp lại entities; chỉ size/suffix vẫn không đủ. debt_flow chấp nhận payment_method=debt ghi lặp khi đúng điều kiện vốn được công cụ thực thi, không mở nhận filter tùy ý.
5. **Review và ngân sách.** Feedback đưa sẵn accept_action, giữ plan/candidate để model chọn final hoặc lập lại plan. SQL/tham số vẫn đầy đủ trong log; SQL native đã được kiểm chứng không bị lặp vào prompt review. Có `review_decision` accept/revise để đo hành vi. unanswerable chỉ chốt khi có bằng chứng missing và không cần thêm vòng review; do đó không mất kết quả hợp lệ ở bước cuối. Numeric candidate chưa review mà hết bước vẫn abstain với cờ rõ `review_budget_exhausted`; không tự bỏ yêu cầu review để chốt số.
6. **Giữ context cho cả hai model.** Phân bố profile dài được gửi cho model từng trang tối đa 20 giá trị, kèm next_offset/values_complete=False đúng thực tế. Log observation vẫn chứa toàn bộ trang công cụ đã trả, thống kê vẫn trên toàn cột. Không cắt README/rulebook hoặc âm thầm coi 20 giá trị là toàn dữ liệu. Rows kết quả tính toán không bị cắt bởi bước này.

Bản này cố ý không sửa plan theo từ khóa câu hỏi để ép đáp án đúng: hỏi lãi gộp mà model chọn revenue, chọn tuần trước thay tuần này, bỏ phủ định hoặc chọn nhầm nhóm vẫn là lỗi cần đo/fine-tune tiếp. Hints và ID không quyết định đáp án.

## Kiểm chứng đã chạy

- **241 software tests passed**, một cảnh báo deprecation từ thư viện TestClient. Có kiểm thử end-to-end cho final số/phần trăm/boolean/ngày/list, query reference không type, projection nhiều cột đã chấp nhận, không bỏ qua answer chép sai, table giữ tên cột, khóa có số 0 đầu, khách có thật nhưng không phát sinh, typed flag, ca giờ, kỳ tương đối và final unanswerable ở bước cuối.
- Replay nguyên raw action của lượt pipeline 6 cho **100 câu cũ**, dùng code mới trên CPU và `schema_mode=code_only` để không tạo lượt schema/model mới. Không cấp plan sửa tay theo nhãn. Đây là kiểm tra cơ chế với output cố định; model thật sẽ thấy feedback mới và có thể sinh action khác.
- Kiểm tra riêng **toàn bộ 31 câu từng lỗi projection**: giá trị công cụ đã ghi đều được giữ khi chốt theo phép chiếu một cột. Trong replay cố định, **15 candidate đúng trước đây bị mất** đều đã chốt được. Điều này không biến các candidate sai nghĩa thành đúng.
- Replay chặn hai tên khách đặt nhầm vào key, sửa kiểu prescription, và chốt unanswerable đã có bằng chứng ở bước sáu. Hai trường hợp price với eq tên đầy đủ được giải đúng identity; sai suffix/đại lượng vẫn không tự được sửa.
- Tokenizer pin offline của cả Qwen và Gemma: đủ 100 prompt đầu và hội thoại replay đều vừa context **16K**, đã chừa **768 token output**. Qwen prompt đầu 8.960–13.309, hội thoại tối đa 13.709; Gemma prompt đầu 9.478–14.149, hội thoại tối đa 14.570. Việc schema repair mới hoặc các action khác trong inference thật vẫn cần đo trong log.

Bằng chứng ở `outputs/verification-interface-v7-2026-11-08/`. Không chạy checkpoint thật/GPU trong đợt sửa này; không có PA hoặc thời gian/câu mới. Không cộng số case replay thành điểm inference mới.

## Chạy lại cùng 100 câu trên Colab

1. Thay `outputs/datamate-colab-source.zip` trên Drive bằng gói mới, mở `notebooks/DataMate_Colab_Inference.ipynb` mới, restart kernel rồi chạy từ ô đầu. Không cần xóa checkpoint đã seal/cache model local trong cùng runtime. Preflight phải hiện **pipeline 7 / backend protocol 3**; diagnostic là **9**. Không resume vào log pipeline 6.
2. Chọn `MODEL_KEY="qwen35-9b"`, `AGENT_MODE="self_review"`. Sau lượt Qwen, dùng kernel riêng và đổi MODEL_KEY thành `"gemma4-12b"`. Hai model dùng cùng API/prompt, context 16384, output 768, max_steps 6, timeout 90 giây, code_first và schema repair giới hạn 20 giây/512 token. Checkpoint/EOS protocol được giữ nguyên.
3. Diagnostic giữ **SAMPLE_SIZE=100**, **seed=20261103**, chọn ngẫu nhiên đều như cũ, **RUN_LIMIT=None**. Suffix mới **random100-interface-v7-Nov8th-01**; output có tiền tố model riêng. Có thể chạy bảy câu trước rồi None để resume khi code/config giữ nguyên. Lựa chọn AUTO_DISCONNECT và QUANTIZATION đã lưu được giữ lại; output cũ trong notebook là lịch sử.
4. AGENT_MODE có thể đổi `single` để đo riêng tác động review, nhưng phải dùng RUN_NAME_SUFFIX khác. Mặc định giữ self_review để hai lượt mới có chế độ giống lượt trước; đây không phải cam kết review sẽ cải thiện khả năng hiểu câu hỏi.

Log JSON/Markdown có raw LLM, plan gốc/plan được kiểm tra, SQL/tham số/kết quả, phép chiếu chuẩn hóa và quyết định review. Summary thêm số projection_normalizations và review accepts/revisions để phân tích tiếp.

## Những gì được giữ nguyên và còn cần đo

- Dữ liệu/rulebook/calendar/as_of_date, danh sách và thứ tự 100 câu giữ nguyên; không nới quy tắc chấm.
- Không thêm bộ dựng tồn theo lô hoặc đoán default nhóm khách vô danh trong đợt này. Các nguồn chưa hỗ trợ vẫn cần kế hoạch/SQL có căn cứ; không dùng thiếu công cụ để tự khẳng định unanswerable.
- Hai ZIP fine-tune Gemma frozen vẫn nguyên hash và đi với source đã khóa của chúng. Chưa tạo lại trajectory hoặc huấn luyện Qwen/Gemma. Muốn dùng runtime mới cho training phải curate/replay theo contract mới và đo trên tập giữ riêng.
- Đợt sửa giao tiếp ban đầu chỉ xây lại notebook 100 câu; các gói benchmark 200 câu và kit riêng khi đó được giữ nguyên. Sau yêu cầu tiếp theo đối chiếu chat, kit/notebook Colab 200 câu đã được đồng bộ pipeline 7 trong [đợt đóng gói riêng](team200-colab.md); mẫu 200 câu và gói SFT vẫn giữ nguyên.
- Source ZIP trước cập nhật được giữ ở `outputs/datamate-colab-source-before-interface-v7-2026-11-08.zip`; notebook trước cập nhật ở `outputs/notebook-backups/inference-before-interface-v7-2026-11-08.ipynb`.
