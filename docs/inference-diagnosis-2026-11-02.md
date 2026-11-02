# Chẩn đoán inference và kế hoạch kiểm tra · 02/11/2026

Trạng thái: đã phân tích trace Gemma lưu trong notebook và chạy truy vấn kiểm chứng trên train local. Chưa chạy lượt inference GPU mới hoặc fine-tune; người dùng sẽ execute batch trên GPU. Chỉ bổ sung ô chẩn đoán cuối notebook; giữ nguyên các ô cũ, prompt, runner, SQL tools và backend. Phương án cải thiện dưới đây cần người dùng duyệt sau khi xem báo cáo batch 72 câu.

## Bằng chứng hiện có

Nguồn: `notebooks/DataMate_Colab_Inference.ipynb` do người dùng lưu từ Colab; train nguyên bản và `rulebook.md` của train. Câu hỏi: “Hôm nay cửa hàng bán được bao nhiêu tiền?”, `biz_101`, `as_of_date=2026-02-22`. Câu này khớp một câu train (`q_1cc70c12aa27`); ID chỉ dùng ghép báo cáo và chọn mẫu, không dùng quyết định đáp án.

- Trace lưu: Gemma 4 12B, không quantization; 10 lượt model, 52,212 giây, cuối cùng trả `unanswerable`.
- Đáp án train: `number`, **13.479.900**.
- Kiểm chứng local: so ngày raw với ISO đếm 0 dòng; dùng `iso_date("Ngày tháng")` đếm **243 dòng** cho ngày đó. Trong đó **240 dòng `Xong`**, tổng `num_en("Thanh tien")` là **13.479.900**; **3 dòng `Đã hủy`**, tổng 56.000.
- Toàn file có các nhãn `Xong`, `Đã hủy`, `Đổi trả`, `Đang giao`. SQL trong trace lọc `NOT IN ('huỷ', 'trả hàng')` không loại đúng các nhãn hủy/trả thực tế của file này.
- `num_vi('560,000')` cho 560, còn `num_en('560,000')` cho 560.000. Ở tập dòng của ngày trên, dùng `num_vi` thực sự gây lỗi hàm SQL vì có chuỗi nhiều dấu phẩy.

Đây là phân tích một lỗi đã biết, chưa phải accuracy của model trên toàn train. Truy vấn kiểm chứng theo cột/nhãn của case này chỉ là bằng chứng; không được đưa nguyên thành logic chung cho mọi doanh nghiệp.

## Các lỗi đã xác định và phần còn phải đo

| Vấn đề | Bằng chứng | Hệ quả |
|---|---|---|
| Sai cách chuẩn hóa ngày | So cột raw với chuỗi ISO; có lần gọi `iso_date` với 2 tham số dù helper chỉ nhận 1 | Kết quả rỗng giả hoặc lỗi SQL |
| Sai locale số | Dùng `num_vi` với số có dấu phẩy phân cách nghìn | Sai hệ số hoặc lỗi hàm |
| Sai nhãn trạng thái | Nhãn trong file khác chuỗi điều kiện SQL | Có thể tính cả đơn hủy/trả nếu sửa riêng ngày/số |
| Lệch giao thức action | 3 lượt có `thought`, giải thích ngoài JSON hoặc nhiều JSON object trong cùng output | Không thực thi được SQL, mất 3/10 lượt |
| Chấp nhận kết luận thiếu dữ liệu quá sớm | Runner chỉ cần đã có query thành công; query sai nghĩa nhưng đúng cú pháp vẫn thỏa | Chấp nhận `unanswerable` sai; không phát hiện truy vấn rỗng giả |

Prompt hiện đã nói về helpers, ngày, raw data và rulebook. Bổ sung thêm một lời nhắc chung chưa có bằng chứng sẽ giải quyết được lỗi. Cần kiểm tra khả năng tuân thủ, thông tin schema/profile được cung cấp và điểm kiểm chứng sau SQL.

Chưa xác định: tần suất mỗi lỗi trên doanh nghiệp khác, tỉ lệ lỗi do thiếu quan hệ bảng/đơn vị/forward-fill/dedup, context quá dài, timeout và chất lượng đáp án dạng list/table. Cũng chưa kết luận chat template là nguyên nhân của `thought`; cần đọc template/tokenizer thực tế của phiên Colab. Harness lưu chúng để đối chiếu.

## Lượt batch 72 câu người dùng đã chọn

Người dùng yêu cầu gộp thành một lần chạy để tránh nhiều phiên GPU tốn Unit, và đã chọn 72 câu (3 câu/doanh nghiệp) và yêu cầu có câu không dấu, viết tắt và câu khó. Không cần chạy pilot rồi quay lại chạy lượt khác.

Đã thêm ô cuối **“Chẩn đoán batch: 72 câu train, 3 câu mỗi doanh nghiệp”** vào `notebooks/DataMate_Colab_Inference.ipynb`. Các ô và output cũ được giữ nguyên; có bản sao notebook trước khi thêm ô trong `outputs/diagnosis-2026-11-02/`. Mã độc lập nằm ở `tools/diagnose_train.py`; ô notebook chứa toàn bộ mã nên không cần upload script/source ZIP mới.

1. Dùng cùng kernel Colab đang có `agent`, `WORK`, `SAVE`. Nếu Gemma đã load thì **chỉ execute ô cuối**; không chạy lại ô download/load. Nếu VS Code chưa hiện ô mới, mở lại file từ ổ E:; mã cũng có thể copy trực tiếp từ file Python vào một ô mới.
2. Code kiểm tra train fingerprint và giữ nguyên model, prompt, SQL tools, decoding và cấu hình agent. Không fine-tune, không nạp model khác.
3. Chia doanh nghiệp seed 42 như pipeline cũ: 24 development; giữ riêng 6 holdout (`biz_107`, `biz_111`, `biz_115`, `biz_120`, `biz_123`, `biz_127`). Chọn **đúng 3 câu mỗi development business** (72 câu), trong đó mỗi doanh nghiệp có câu không dấu, có viết tắt và có tín hiệu xử lý nhiều bước; một câu có thể thuộc nhiều nhóm. Chọn từ câu train nguyên bản, không tự bỏ dấu hay thêm lỗi để giữ nguyên tính hợp lệ của nhãn. Tiêu chí chọn dùng question text và ID/type để cân đối; không đưa nhãn/profile vào prompt hoặc lời gọi agent. Đáp án chuẩn chỉ dùng chấm sau inference. Có thêm câu lỗi đã biết trong chính quota 3 của biz_101, không cộng thêm câu.
4. Giữ nguyên `as_of_date` từng câu. Lưu log ngay từng câu, đo thời gian lời gọi model. Báo cáo lưu cấu hình, phiên bản thư viện, thiết bị GPU/CPU, chat template, hash source và model manifest.
5. Một lần chạy xuất tất cả đáp án, PA (+1/−1/0), đúng/sai/bỏ qua theo type, type mismatch, false-unanswerable, JSON/SQL/context errors, số lượt model và latency. Flags có thể trùng; chưa phải kết luận nguyên nhân tự động. Mẫu cân bằng type **không đại diện tỷ lệ câu của toàn train**, nên PA này dùng chẩn đoán/so sánh trên cùng mẫu, không ngoại suy điểm thi.

Cấu hình đầu ô:

```python
SAMPLE_SIZE = 72
SAMPLING_MODE = "language_stress_3"
MAX_RUN_MINUTES = 90
AUTO_DISCONNECT = False
RUN_NAME_SUFFIX = "stress72-baseline-01"
```

Ngưỡng thời gian là 90 phút, tương đương khoảng 15 Unit theo mức 10 Unit/giờ người dùng cung cấp: kiểm tra trước câu mới và truyền ngân sách còn lại vào agent, không cưỡng chế ngắt một GPU forward pass. Nếu chưa đủ 72 câu sẽ xuất `status: partial` và số câu đã hoàn tất; không gọi đó là kết quả toàn mẫu. Đổi trần nếu muốn, trước khi execute. Không tính thời gian load model vì dùng model đã nạp.

`AUTO_DISCONNECT=False`: sau dòng REPORT, ngắt runtime GPU thủ công để tránh giữ phiên khi phân tích. Có thể đổi True trước khi execute: sau khi ZIP được ghi và kiểm tra CRC, code thử flush Drive (giới hạn 60 giây) rồi `runtime.unassign()`. Nếu môi trường VS Code/Colab không hỗ trợ hoặc flush lỗi, nó báo để ngắt thủ công. Chức năng này kết thúc runtime, nên các biến/model trong RAM không còn; file Drive vẫn dùng lại. Chưa thử trên runtime Colab của người dùng. Nguồn API: [Google Colab runtime.unassign](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py), [Drive flush_and_unmount](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py).

Kết quả mặc định:

```text
MyDrive/DataMate/runs/inference-01/diagnostics/gemma4-12b-stress72-baseline-01/
├── run_manifest.json
├── selected_questions.jsonl
├── runtime.json
├── logs/
├── submission.jsonl        # toàn bộ prediction đã chạy
├── review.csv              # từng câu: prediction, gold, đúng/sai, lỗi, thời gian
├── review.jsonl            # đầy đủ trace để phân tích
├── summary.json
└── diagnostic_bundle.zip
```

Nếu SAVE hoặc model khác, lấy đường dẫn REPORT in cuối ô. Tải **diagnostic_bundle.zip** về `outputs/diagnosis-2026-11-02/` trên E: rồi báo đã xong. Một gói này đủ để phân tích toàn lượt; có thể Ctrl+S notebook để giữ summary. Không cần gửi từng câu.

Chạy lại cùng RUN_NAME và identity sẽ bỏ qua câu đã có log. Nếu đổi model/config/source/data/sample hoặc re-extract train sang đường dẫn khác, công cụ có thể yêu cầu RUN_NAME mới để tránh trộn lượt. Đây là khả năng phục hồi khi bị ngắt; lượt mặc định chạy toàn bộ 120 một lần.

## Đề xuất sơ bộ để duyệt sau batch

1. **Sửa dữ liệu và công cụ trước.** Thêm profile cho từng cột: kiểu ngày, locale số, tỉ lệ parse được, khoảng ngày, các nhãn trạng thái thực tế. Tạo cột/view chuẩn hóa có provenance, giữ raw nguyên vẹn; trường mơ hồ cần báo rõ. Mapping dựa trên schema, README, rulebook và giá trị quan sát, không theo question ID/business ID. Xử lý forward-fill, trùng dòng, đơn vị và loại cả đơn trả ở đúng cấp khóa đơn; kiểm chứng riêng, không áp một cách làm mù cho mọi bảng.
2. **Ổn định giao thức model.** Kiểm tra chat template và token kết thúc của Gemma trong runtime; thử hướng dẫn/ví dụ tool ngắn và ràng buộc JSON phù hợp backend. Không tự chọn một JSON tùy ý trong output chứa nhiều action. Đo parse-error rate và số lượt sửa, giữ seed/config và cùng tập câu để so trước/sau.
3. **Bổ sung kiểm chứng kết luận.** `unanswerable` cần bằng chứng thiếu trường/khoảng ngày sau parse hoặc công cụ kiểm tra phạm vi độc lập. SQL ra rỗng không đủ để kết luận. Phân biệt kỳ có dữ liệu nhưng không phát sinh với kỳ hoàn toàn ngoài dữ liệu theo rulebook. Đáp án số cần provenance của query cuối và kiểm tra phạm vi/đơn vị/filter; SQL chạy được chưa chứng minh đúng nghiệp vụ.
4. **Đo lại từng thay đổi, rồi mới fine-tune.** Dùng lại cùng sample development; giữ 6 doanh nghiệp holdout cho đánh giá độc lập sau khi chọn phương án. Fine-tune chỉ khi pipeline dữ liệu/tool đã được kiểm chứng và lỗi còn lại thuộc lựa chọn SQL/quy tắc/định dạng của model. Trajectory cần lọc cả đáp án cuối và kiểm tra ngữ nghĩa SQL mẫu; không học lại trace sai trên đây. Chưa có cơ sở chọn model thắng, đổi model hay khẳng định LoRA sẽ cải thiện.

Hiệu năng nạp trọng số là việc riêng: kiểm tra hash lặp và đọc Drive có thể làm chậm startup. Nó không giải thích trực tiếp đáp án `unanswerable` sai đã xác nhận. Tối ưu startup sẽ được đo riêng, giữ nguyên kiểm tra tính toàn vẹn khi đưa vào Docker offline.

## Kiểm tra công cụ chẩn đoán

Đã qua 9 kiểm tra local: kế hoạch tái lập/cân bằng, chọn đúng 3 câu mỗi development business và đủ nhóm ngôn ngữ, loại holdout, không đưa nhãn vào prompt và chấm sau inference, tiếp tục không chạy lặp, khóa identity khi cấu hình đổi, lưu báo cáo dở và trả backend gốc khi ngắt, thứ tự flush/ngắt qua mock, cùng thực thi nguyên ô để xuất đầy đủ kết quả. Các kiểm tra dùng backend giả để xác minh harness, không phải benchmark LLM. Người dùng vẫn cần chạy batch 72 câu trên GPU thực.

Mẫu 72 thực tế đã kiểm tra trên train local: number 10, percent 10, string 10, date 10, boolean 8, list 9, table 7, unanswerable 8. Có **33 câu không dấu**, **33 câu có viết tắt**, **34 câu có complex_hint**; các nhóm chồng lấp và mỗi business đều phủ đủ ba nhóm. Đây là mẫu đã chuẩn bị, chưa có prediction GPU. CSV xem trước: `outputs/diagnosis-2026-11-02/selected-sample-72.csv`.

`complex_hint` được gắn theo tín hiệu từ câu hỏi: so sánh kỳ, tỷ lệ, gom nhóm rồi xếp hạng, điều kiện trên số đơn/khách, công nợ/giá vốn, quy đổi đơn vị và lịch. Đây là heuristic để chọn câu có khả năng khó; không phải độ khó đã đo hoặc xác nhận cần bao nhiêu lượt model. Nhóm viết tắt dựa vào token quan sát như `dthu`, `bn`, `bnhiu`, `dc`, `sp`, `hqua`, `trc`, `vs`; không tự sửa câu trước khi inference. Báo cáo có `by_question_group` và các cột tương ứng trong CSV để so đúng/sai, PA, thời gian theo nhóm. Do nhóm chồng lấp, không cộng các nhóm để ra tổng số câu.
