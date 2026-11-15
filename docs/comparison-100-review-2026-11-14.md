# Phân tích ba cấu hình100 câu —14/11/2026

Đã đọc đầy đủ ba submission,300 trace, timing từng câu, model calls, GPU samples, identity và cache liên quan. Source: `E:/Bảo Lập Trình AI/ZALO_AI_2026/outputs/comparison-100-20261114-101046`. Chấm **offline sau inference** bằng `train/train/answers.jsonl` và evaluator hiện tại; đây là điểm local, không phải kết quả BTC.

**Quan sát chính:** vLLM xử lý tập100 câu nhanh hơn Torch và đáp án cuối giống96/100 câu; RAG tăng điểm phạt sai từ57% lên62% nhưng tạo thêm các hồi quy. Chưa chốt cấu hình vLLM+RAG vì chưa có lượt kết hợp. Các giới hạn timing và trùng dữ liệu huấn luyện ở dưới phải đi cùng kết quả này.

## 1. Điều kiện đã kiểm

- Đủ cùng100 ID, cùng thứ tự trong identity; không mất/trùng ID ở bất kỳ submission nào.
- Cùng source hash toàn bộ Python, generator manifest, Python/common libraries, A100-SXM4-40GB cùng UUID, FP16, greedy/seed42, một worker, context16384, max_new_tokens768,90s/câu và6 bước. Initial baseline prompt hash khớp100/100; RAG thêm evidence có chủ đích.
- Cùng input hashes giữa ba run; kiểm lại **151 file local**, tất cả khớp. Gold SHA256 `0b5fe9d13edf5fae306bda9cafaf699c06b639846c8e1d9e41c39a86d1160b3d` khớp registry training_sources.
- Model inventory RAG tổng10.788.614.897 tham số, dưới15B theo inventory đã lưu. Không có lỗi CUDA OOM/model-call exception trong ba lượt100 câu này; lỗi SQL/plan khác với lỗi hạ tầng.
- **Giới hạn mẫu:** đây là100 câu đầu `questions.jsonl`, không phải mẫu random.82 thuộc phân hoạch shop train; kiểm ZIP dataset đúng hash cho thấy **80 ID xuất hiện thực sự trong train.jsonl**,18 trong development-validation.jsonl đã được xem trước;2 chưa có example. Không gọi đây là unseen/private benchmark.

## 2. Chất lượng theo gold

| Cấu hình | Đúng | Sai | Abstain | (Đúng−Sai)/100 | Coverage |
|---|---:|---:|---:|---:|---:|
| Torch baseline |66|9|25|57%|75%|
| vLLM |67|8|25|59%|75%|
| Torch + RAG |71|9|20|62%|80%|

Abstain=0 điểm, sai=−1, đúng=+1; type/value đều phải khớp theo local evaluator. Hai câu gold unanswerable được cả ba trả đúng. Không nhầm tỷ lệ đúng71% với điểm phạt sai62%.

**vLLM:**96/100 final type/value giống baseline (không so reason abstain).2 câu abstain→đúng,1 câu sai→abstain,1 câu đúng→abstain; không có đúng→giá trị sai. Chênh lệch này có cả ảnh hưởng deadline và chuỗi hành động SQL khác nhau; không chỉ là phép tính số khác.

**RAG:** so baseline có9 abstain→đúng,3 sai→đúng; đồng thời5 đúng→abstain,2 đúng→sai,2 abstain→sai,1 sai→abstain. Tổng cộng12 câu được cứu thành đúng nhưng mất7 câu vốn đúng; điểm ròng+5. Final type/value giống baseline78/100.

Riêng18 câu development-validation: baseline13 đúng/3 sai/2 abstain; vLLM giống baseline; RAG14 đúng/3 sai/1 abstain. Mẫu nhỏ và đã xem trước nên không suy rộng độ cải thiện.

## 3. Thời gian và bộ nhớ

Để không phải làm giả metrics của vLLM bị ngắt, dùng cùng mốc từ `question_timing.jsonl`: câu đầu bắt đầu → câu cuối kết thúc; và observer bắt đầu → câu cuối kết thúc. Có đủ100 record thật ở cả ba. Không loại câu chậm/sai/abstain.

| Đại lượng | Torch | vLLM | Torch+RAG |
|---|---:|---:|---:|
| Bắt đầu observer→bắt đầu câu1 |71,31s|391,11s|490,24s|
| Câu1→hết câu100 (bao gồm khoảng trống/I/O giữa câu) |3286,68s (54:46,68)|1479,86s (24:39,86)|2898,20s (48:18,20)|
| Tổng elapsed100 câu |3284,22s|1476,99s|2895,73s|
| Observer→hết câu100, chưa shutdown |3357,98s (55:57,98)|1870,97s (31:10,97)|3388,45s (56:28,45)|
| Tổng wall trong model calls |2501,38s|683,80s|2425,86s|
| Số model calls |278|281|262|
| Tổng output tokens |32382|33531|30572|
| Peak GPU được lấy mẫu |31,69GiB|30,58GiB|34,51GiB|

- vLLM nhanh **khoảng2,22 lần** trong giai đoạn xử lý100 câu, **khoảng1,79 lần** nếu tính từ observer start đến đáp án cuối. Đây là tỷ số deployment quan sát trên lượt này, không phải benchmark engine-only hoặc process-completion/cold-Docker đã xác nhận.
- Tổng model-call wall thấp hơn khoảng3,66 lần dù vLLM tạo nhiều hơn1149 output tokens. Nhưng input/output/hành động khác nhau sau một số bước nên không gọi đây là speedup kernel thuần. Phần công việc ngoài model gần như giữ nguyên: Torch khoảng782,84s, vLLM793,20s, giải thích vì sao toàn pipeline chỉ nhanh2,22 lần.
- RAG prepare/index/ingest ghi401,84s, retrieval100 câu tổng41,96s. Nó giảm số call16 nhưng thêm khoảng11% input tokens. Raw ingestion được làm trước trong prepare nên một phần chi phí chuyển từ từng câu sang startup; RAG có phase xử lý ngắn hơn nhưng tổng đến đáp án cuối **dài hơn baseline khoảng30,46s**, không chứng minh model chạy nhanh nhờ RAG. Cache RAG có26 index hợp lệ, một index mỗi business; không thấy build lặp raw/semantic như một phỏng đoán chưa kiểm chứng.
- So với bảng tạm từ cell, tổng elapsed vLLM1476,8s nay thay bằng1476,9915s từ raw timing; chênh lệch do làm tròn hiển thị.

**Vì sao strict gate vẫn fail:** baseline có2 lần `time_or_other`, RAG có1; duration xấp xỉ ngân sách còn lại và output không có EOS, phù hợp bị cắt theo max_time. `timeout_calls=0` không có nghĩa không chạm thời gian: backend Torch gán loại stop này thay vì exception timeout. vLLM có1 call chạm768 token (`length`). vLLM thiếu benchmark_metrics/run_summary do ngắt ở shutdown. Báo cáo baseline–RAG gốc fail `both_runs_valid` và `no_deadline_limited_questions`; mọi gate identity/phần cứng/data/source khác đều pass. Đã chấm lại RAG với gold trong artifact riêng, không sửa gate của báo cáo cũ.

## 4. Nguyên nhân sai đã xác nhận

Chi tiết **13 ID có ít nhất một cấu hình trả giá trị sai**, gồm câu hỏi, gold, đáp án cả ba, trace gốc và7 hồi quy RAG: xem artifact `error_cases.md` bên dưới. Các nhóm lỗi chính:

1. **Hiểu sai phép toán/cấp tổng hợp:** hỏi đếm đơn vượt ngưỡng lại SUM tiền trên từng dòng; hỏi tỷ lệ khách mua từ4 đơn lại dùng MIN(order_id); hàng ế bị tính bằng tập bán hiện tại trừ chính tập đó. SQL vẫn chạy nhưng trả lời sai ý.
2. **Kiểu dữ liệu tiền chưa chuẩn hóa:** cột `Cần trả NCC` còn TEXT như `56,356,700`; SUM của SQLite không đọc đủ số. Kiểm chỉ đọc trên chính DB cache:1633→39480100 ở một shop và87→89526400 ở shop khác khi dùng parse_number. Không phải vLLM sai số học. Tên clean không bảo đảm mọi cột đã được chuẩn hóa khi binding thiếu role.
3. **Sai cột/domain/entity:** `payment_method=phone` thay vì channel; channel thật có store nhưng model dùng quay/counter; nhãn thân thiết không chuẩn hóa; size M bị coi là tên sản phẩm; product key tự tạo thay vì lookup.0 do không khớp bị nhận làm đáp án thay vì kiểm chứng lại thực thể.
4. **Ràng buộc SQL không có trong câu hỏi:** RAG đếm người còn nợ nhưng thêm EXISTS khoản trả, loại người chưa từng trả. Bỏ riêng điều kiện đó cho35 đúng gold thay vì21.
5. **RAG làm trôi đối tượng:** nhóm tiêu hóa bị thay bằng dị ứng; model chấp nhận0. Retrieval context chọn nhiều header hóa đơn, không cung cấp nhóm cần hỏi. Đây là lỗi thay tham số quan sát được; chưa đủ để quy kết riêng embedding/reranker.
6. **Vòng sửa lỗi không tiến triển:** requirements quantity_ranking không thuộc capability, alias SQL không tồn tại, SUM sai arity, tham số ngày sai, chọn nguồn khác plan, type percent/number. Nhiều vòng lặp kết thúc abstain. Có guard đang chặn SQL sai; không nên bỏ guard chỉ để tăng coverage.

Gate SQL custom hiện chỉ kiểm nguồn/cột bắt buộc, trả `semantic_meaning_proven=false`; nó chưa chứng minh phép toán/cấp đếm/giá trị filter giữ đúng câu hỏi. Đó là lý do SQL sai ý vẫn có thể qua self-review và xuất final.

## 5. RAG: phần tốt và lỗi cần sửa

-100/100 retrieval báo ok, không timeout/fatal; nguồn chọn không lẫn business khác.394 chunk thực sự đưa vào prompt:173 rulebook,94 README,127 schema. Log initial_messages đã có text/score/provenance để review.
- RAG giúp chọn đúng channel, search_key nhãn/tên và sửa một số SQL. Tuy vậy tên file/README/rulebook có sẵn nguyên bản trong baseline, nên retrieval đang nhấn mạnh/chọn lại thông tin và thêm schema; không có bằng chứng truy xuất thêm dữ liệu giao dịch để tự tính kết quả.
- **Lỗi code đã xác nhận:**26 chunk_id trong index trỏ tới nhiều text/schema khác nhau;28 câu có ID trùng trong context được chọn. sections() tạo ID từ file hash, section và offset nhưng thiếu table/sheet/text; các mô tả sheet khác nhau cùng độ dài bị trùng ID. Điều này làm mơ hồ provenance và cản dedup/diversity; chưa chứng minh vector đã bị hoán nhầm hoặc mọi câu sai là do collision.
- Ví dụ câu nhóm tiêu hóa nhận4 header của cùng workbook hóa đơn, gồm tháng không liên quan, còn câu Lifebuoy nhận cả đoạn luật nhà thuốc. Rerank vẫn trả top-k dù ngữ cảnh yếu. Điểm reranker là logit, không suy ra xác suất hay tự đặt ngưỡng0; cần hiệu chỉnh relevance trên tập phát triển hợp lệ.

## 6. Đề xuất theo thứ tự

1. Siết contract câu hỏi→plan về đại lượng, cấp tổng hợp, thực thể, kỳ và ngưỡng. Ưu tiên plan biên dịch/calculate cho operation đã hỗ trợ; custom SQL phải kiểm cả nghĩa, không chỉ có đọc đúng cột.
2. Chuẩn hóa typed purchase/header amount bằng nguồn/schema và parser đã có; chặn aggregate trên TEXT chưa có căn cứ parse. Kiểm domain channel/group và bắt buộc lookup khi entity chưa giải quyết. Giữ đúng rulebook mỗi split; không sửa theo ID/shop cụ thể.
3. Sửa chunk_id để có table/sheet hoặc hash text, đổi cache fingerprint; chọn chunk có đa dạng/relevance và bám đúng đối tượng/kỳ. Đánh giá RAG lại trước bật mặc định.
4. Cải thiện sửa lỗi có hướng: báo rõ capability hợp lệ, phân biệt sai type/SQL/entity, tránh lặp query không thay đổi; ưu tiên fallback calculate khi đã có plan typed. Không tăng token/bước vô điều kiện để che lỗi hiểu sai.
5. Hoàn tất shutdown vLLM và ghi bền kết quả/telemetry trước bước teardown; không làm giả status complete. Theo dõi time stop Torch tách khỏi EOS/length. Chạy lượt đo trọn vẹn sau sửa để có process wall/strict gate.
6. Giữ vLLM là ứng viên tăng tốc; RAG tiếp tục là tùy chọn. Chạy holdout phù hợp sau khi sửa; không train trên validation/private/public test và không kết luận từ80 câu đã có SFT. Combo vLLM+RAG cần một phép đo riêng.
7. Rehearsal image offline trên GPU mục tiêu: RAG peak34,51GiB ở A100 nên chưa bảo đảm fit32GiB/GPU nếu đó là cấu hình BTC được xác minh. Tổng model parameters đạt giới hạn không đồng nghĩa đủ VRAM. Tính cả startup/index/JIT vào6giờ.

## 7. Artifact và giới hạn

Artifact local tại `outputs/runs/Bảo/comparison100-analysis-20261114/` (không qua Git):

- `diagnostic_summary.json`: số đo, score, transitions, giới hạn.
- `per_question_comparison.jsonl`: đủ100 câu ghép gold/prediction/timing và error đã quan sát.
- `error_cases.md`:13 câu sai và các hồi quy có link trace gốc.
- `verified_readonly_checks.json`: các truy vấn chẩn đoán số tiền, count đơn, công nợ và nhãn channel; không sửa dataset/cache.
- `rag_index_review.json`: ID collision, số context có trùng và kiểm scope.
- `training_overlap.json`: ID thực sự có trong train/validation JSONL của ZIP đã kiểm hash.
- `rag_vs_baseline.with_gold.json`: chạy lại comparator với gold; vẫn giữ strict gates fail đúng thực tế.
- `analysis_snapshot.json`: snapshot phân tích từ các file cung cấp.

Chưa sửa hành vi runtime/Agent/RAG trong lần review này. Kết quả là bằng chứng trên một tập phát triển100 câu, một checkpoint và một GPU, không phải kết quả nộp BTC hay cam kết chất lượng private test.
