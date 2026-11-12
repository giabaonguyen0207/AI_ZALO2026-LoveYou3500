# Chuẩn bị SQL SFT cho Qwen3.5-9B — 09/11/2026

> Bản mới cho kernel tăng tốc và tự resume: [v1.3 — hướng dẫn ngày 10/11/2026](training-sql-speed-resume-2026-11-10.md). Giữ các gói v1/v1.1/v1.2 để truy vết; pilot A100 v1.2 đã hoàn tất nhưng chậm do fallback kernels. Dùng cùng bộ source–dataset–notebook v1.3 khi bắt đầu run mới.

> Cập nhật 10/11/2026: dùng notebook `DataMate_Colab_Qwen35_9B_SQL_SFT_v1_1.ipynb` cùng hai ZIP `qwen35-9b-sql-sft-v1.1.zip` và `datamate-qwen35-9b-sql-source-v1.1.zip`. Bản v1 truyền `warmup_ratio` đã bị Transformers 5 loại bỏ; v1.1 dùng `warmup_steps=0.03`, vẫn warmup 3%. Dữ liệu học, tokenizer và checkpoint giữ nguyên; provenance và hash tương ứng được phát hành riêng, không ghi đè v1. Khởi động runtime mới, chạy notebook v1.1 từ đầu; model base trên Drive được tái sử dụng. Notebook chọn RUN_TAG v1.1 mới để tránh thư mục smoke đã tạo dở. Đây là sửa tương thích API; chưa xác nhận training vừa VRAM L4 bằng GPU thực tế.

Nguồn yêu cầu: chủ dự án ngày 09/11/2026, cùng đề xuất trong chat [Phân tích kết quả inference lần 2](codex://threads/01a1a0a2-9d9a-7c11-8487-caea75948690). Đây là bộ dữ liệu mới và notebook mới; các artifact Gemma frozen trong `archive/2026-11-09/` được giữ nguyên.

## Dữ liệu này dạy model làm gì?

Model nhận câu hỏi, `as_of_date`, README, rulebook và catalog giống runtime. Model sinh action JSON chứa `plan.query.sql` và `parameters`; khi cần, model gọi `schema`, `lookup` hoặc SQL khám phá trước. Runtime thực thi SQL và tạo đáp án đúng kiểu từ kết quả. Với trường hợp thật sự thiếu dữ liệu, model học `final` loại `unanswerable` dựa trên evidence công cụ.

Ví dụ giản lược, chỉ để hiểu cấu trúc:

```json
{"action":"plan","measure":"revenue","operation":"sum","type":"number","population":"sales","evidence_id":"Q","periods":[{"kind":"range","start":"2026-05-01","end":"2026-05-31"}],"filters":[],"query":{"sql":"SELECT SUM(revenue) AS value FROM dm_sales WHERE order_date BETWEEN ? AND ?","parameters":["2026-05-01","2026-05-31"],"columns":["value"]}}
```

Không nạp ví dụ giản lược này vào train riêng. Mẫu thật dùng schema, quy tắc doanh thu/trạng thái/đơn vị và SQL đã được thực thi trên doanh nghiệp tương ứng. Không đưa gold hoặc question ID vào request. Các ID ở metadata dùng để ghép nhãn, tách tập và truy vết.

Định dạng là JSONL hội thoại với `messages` gồm `role/content`, kèm metadata và `trainable_assistant_indices`. Trainer đi kèm chỉ tính loss trên các lượt assistant được chọn và token kết thúc lượt. Không tính loss trên system, câu hỏi hay kết quả tool. Ở mẫu `unanswerable`, plan chỉ phục vụ kiểm tra thiếu dữ liệu được giữ trong context nhưng mask khỏi loss; chỉ học quyết định có evidence thật và các bước khám phá hợp lệ.

## Kết quả chuẩn bị và phạm vi kiểm chứng

Đã xử lý đủ **3.000 câu/30 doanh nghiệp của TRAIN**, kiểm fingerprint theo `configs/training_sources.json`. Không dùng public/private test. Nguồn là `train/train`; file Gemini ngoài Downloads chỉ được audit và không được nhập nguyên làm nhãn.

Kết quả phát hành sau replay và kiểm tokenizer: **2.965 mẫu được nhận, 35 câu vào hàng chờ sửa**. Chia theo doanh nghiệp seed 42: 2.347 mẫu train ở 24 doanh nghiệp và 618 mẫu validation ở 6 doanh nghiệp. Có 2.898 mẫu chứa SQL, 67 mẫu `unanswerable` có evidence; 3.008 quyết định assistant được giám sát. Số liệu và hash cuối cùng nằm trong `outputs/packages/qwen35-9b-sql-sft-v1.release.json` và manifest dataset.

Độ dài sequence sau chat template: min 8.817, median 10.119, P95 13.298, max 14.154 token; không loại mẫu nào vì quá 16.384 token. Prefix dài nhất 13.261 token; target dài nhất 1.115 token. Đây là độ dài của dữ liệu đã biên tập, không bảo đảm mọi trajectory model tự sinh đều có cùng độ dài.

35 câu chưa được nhận gồm 21 ca bị guard chất lượng dữ liệu chặn, 11 trajectory có lỗi công cụ/mapping, 1 ca chưa xác định được nguồn kênh bán và 2 ca diễn giải lỗi gõ chưa hỗ trợ. Những câu này không được đưa vào gradient hoặc đổi nhãn thành `unanswerable` cho đủ số lượng.

Các lớp kiểm tra đã thực hiện:

- Đọc schema, README, rulebook và dữ liệu thực tế; biên tập theo nội dung câu hỏi và catalog, không chọn SQL/đáp án bằng question ID hoặc business ID.
- Sinh và cố định toàn bộ proposal/execution của pass rồi mới mở gold để chấm. Các vòng sửa trước có dựa trên lỗi đã chấm trên TRAIN, bao gồm validation phát triển; đây không phải quá trình mù gold từ đầu đến cuối.
- Replay qua `Agent.run` và tool thật trên CPU. Không nhận trajectory có lỗi trung gian, literal đáp án cuối trong target, hoặc kết quả sai so reference.
- Kiểm các họ truy vấn và ca biên: thời gian theo `as_of_date`, lịch Tết từ calendar, trạng thái đơn, doanh thu/số lượng/số đơn, điều kiện sản phẩm/khách/kênh, đơn vị, tồn kho theo số dư mới nhất, tiền nhập/giá nhập, lợi nhuận và nợ.
- Tokenize bằng tokenizer/chat template đúng revision Qwen; kiểm assistant loss mask và token kết thúc lượt, không cắt ngầm prompt.
- Khóa hash source, dataset, tokenizer và ZIP; xác minh lại trước training.

Bộ mới là kết quả biên tập bằng code theo câu hỏi/schema rồi thực thi công cụ, **không phải 3.000 trajectory mới do Qwen hoặc Gemini inference**. Khớp gold khi replay không phải PA của model; vẫn có thể còn lỗi ngữ nghĩa không bị bộ chấm phát hiện. Test CPU của source sau thay đổi: **242 passed**; chưa có training GPU hoặc kết quả PA/tốc độ sau fine-tune.

Validation đã được xem khi audit và sửa bộ biên tập. Có thể dùng nó để theo dõi loss và so sánh phiên bản trong quá trình phát triển, nhưng không gọi nó là tập đánh giá mù. `validation_ids.json` giữ đủ **630 câu**, kể cả 12 câu chưa có nhãn được nhận; chạy inference trên toàn bộ 630 câu để tránh chỉ đo trên phần đã curate thành công. Tương tự, `train_ids.json` giữ đủ 2.370 câu thuộc partition train.

## Đối chiếu đề xuất trong chat cũ

| Đề xuất trước đây | Áp dụng trong release mới |
|---|---|
| Giữ nguồn, fingerprint và cách chia theo doanh nghiệp | Giữ nguyên TRAIN allowlist và split seed 42; ghi rõ validation đã được xem. |
| Đúng đáp án cuối chưa đủ để gọi reasoning đúng | Kiểm ý định/filter/thời gian/đơn vị, SQL thực thi và lỗi trung gian; thêm test ca biên. Không tuyên bố khớp gold là chứng minh đầy đủ. |
| Mẫu cũ phải theo prompt/tool protocol hiện hành | Tạo lại messages bằng Agent pipeline v6, `mode=single`, `schema_mode=code_only`. Không đổi metadata trên log cũ để coi là trajectory mới. |
| Sửa ca sai bằng nguồn thực, không học action sai | Có `repair_queue.jsonl` riêng; không nhập các lỗi trung gian hoặc ép ca answerable thành unanswerable. |
| Chỉ loss trên assistant, đúng tokenizer và đủ context | Kiểm tokenizer Qwen đã pin, loss mask/EOT và giới hạn 16.384 token; trainer tôn trọng `trainable_assistant_indices`. |
| Có train/validation/queue/report rồi mới chạy GPU | Có bộ phát hành với manifest, inventory đủ 3.000 câu, thống kê token và notebook pilot riêng. |
| So chất lượng agent trước/sau trên cùng tập | Chưa đo GPU. Sau train cần baseline và adapter cùng code/prompt/mode/IDs/budget; không lấy loss hoặc PA `self_review` cũ làm kết luận. |

Đề xuất cũ ưu tiên plan ngắn để tiết kiệm thời gian inference; yêu cầu lần này là học sinh SQL, nên target mới chứa SQL rõ ràng. SQL dài hơn có thể làm tăng token/thời gian; phải đo bằng inference sau train. Không áp dụng nguyên notebook/seed Gemma 136 câu cho Qwen9B.

## Checkpoint và cấu hình

Base: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Apache-2.0, 9.653.104.368 tham số theo registry/evidence của dự án. Revision trọng số được ghi nhận công khai ngày 02/03/2026, trước mốc 01/10/2026; xem `docs/model-evidence-2026-11-01.json`, [commit checkpoint](https://huggingface.co/Qwen/Qwen3.5-9B/commit/c202236235762e1c871ad0ccb60c8ee5ba337b9a) và [model card đúng revision](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md). Không tự thay bằng `main`. Download/seal toàn bộ weights sẽ thực hiện ở Colab; đợt chuẩn bị local này chỉ dùng tokenizer và metadata, không load full model.

Mặc định QLoRA: base 4-bit NF4, double quantization, LoRA r=16/alpha=32/dropout=0,05, chỉ linear layers thuộc language model; LR=5e-5, batch=1, gradient accumulation=8, 1 epoch, seed 42. Có profile LoRA thường dùng cùng base/dataset. QLoRA dùng ít bộ nhớ lưu base hơn, nhưng activation và context vẫn tốn VRAM; cấu hình này chưa được đo trên GPU. Tham khảo [hướng dẫn PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization) và [Qwen3.5 trong Transformers](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

Notebook yêu cầu tối thiểu 22 GiB cho profile QLoRA, 38 GiB cho LoRA. Đây là ngưỡng preflight, không phải bảo đảm vừa VRAM. Chọn A100 nếu có; L4 phải vượt bước smoke/pilot. T4 16 GB không chạy được profile hiện tại qua guard. Không giảm context bằng cách cắt rulebook hoặc SQL; nếu cần GPU nhỏ hơn, chuẩn bị một profile/dataset mới có kiểm chứng.

## Chạy trên Colab

1. Chép hai file sau vào `MyDrive/DataMateSQLSFT/`:
   - `outputs/packages/qwen35-9b-sql-sft-v1.zip`
   - `outputs/packages/datamate-qwen35-9b-sql-source-v1.zip`
2. Mở `notebooks/training/DataMate_Colab_Qwen35_9B_SQL_SFT.ipynb` trên Colab. Chọn GPU, High RAM và chạy trong runtime mới.
3. Giữ `METHOD="qlora"`, `STAGE="pilot"` và đặt `RUN_TAG` riêng. Notebook kiểm hash ZIP/source/dataset, giữ Torch đã có và tải đúng base revision nếu chưa stage.
4. Notebook chạy một bước train trên mẫu train dài nhất trong process riêng để phát hiện lỗi bộ nhớ/cấu hình sớm, rồi pilot 5 optimizer steps. Kiểm report/log trên Drive. Pilot 5 bước chưa đủ chu kỳ eval mặc định 50 bước, nên chưa chứng minh bộ nhớ evaluation hoặc chất lượng agent.
5. Sau khi pilot đạt, đổi `STAGE="full"` và `RUN_TAG` mới, chạy lại từ base. Full chạy 1 epoch. Chọn `METHOD="lora"` khi dùng GPU đủ bộ nhớ. Không ghi đè hoặc resume lẫn dataset/base/config khác.

Kết quả nằm ở `MyDrive/DataMateSQLSFT/runs/<RUN_TAG>/`: adapter, training report, cấu hình, môi trường/pip freeze và kết quả smoke. Notebook không merge tự động. Adapter pilot chỉ để kiểm quy trình; lượt full bắt đầu lại từ cùng base.

Sau full, merge trên GPU đủ bộ nhớ bằng CLI và seal artifact:

```bash
python -m datamate merge --model qwen35-9b --base /path/to/qwen35-9b-c202236 --adapter /path/to/full-run/adapter --output /path/to/qwen35-9b-sql-sft-v1
```

Đo baseline và bản merge trên cùng 630 validation IDs; giữ `single`, `code_only`, require-plan/auto-execute và ngân sách tương đương. Dùng điểm khởi đầu `max_new_tokens=1536`, `max_context_tokens=16384`, `max_steps=8`, `max_result_rows=1000`, SQL timeout 15 giây như bước tạo dữ liệu; prefix dài nhất đã đo cộng 1.536 token vẫn nằm trong context. Cấu hình cũ với 768 token có thể cắt các SQL dài. Giữ hai bản baseline/adapter cùng budget; nếu cần đổi, đổi cả hai và đo lại. Đánh giá PA, số sai, abstain, unanswerable, số calls, thời gian và lỗi parser/SQL. Chỉ sau bước này mới quyết định adapter có cải thiện hệ thống hay không.

## File và khả năng tái tạo

- Pass đã replay: `outputs/runs/review/sql-sft-qwen35-9b-20261109/release-pass-v1/` gồm execution đầy đủ, summary và source hash lúc tạo.
- Dataset phát hành: `outputs/runs/review/sql-sft-qwen35-9b-20261109/dataset-v1/` gồm train/validation, queue, inventory, IDs, báo cáo và manifest.
- `tools/prepare_sql_training.py`: biên tập/replay TRAIN, freeze proposal trước khi mở gold; chạy CPU.
- `tools/release_sql_training.py`: kiểm source, tokenizer/mask/độ dài rồi seal dataset và tạo bộ Colab.
- `tools/build_sql_colab.py`: tạo notebook riêng với hash source ZIP được khóa.

Giữ cặp source–dataset–notebook–release JSON cùng phiên bản. Khi đổi prompt/schema/tool, chạy lại các ca liên quan và phát hành phiên bản mới; không sửa hash để bỏ guard. Data/model/output/cache/archive không đưa vào Git. Chưa thay entrypoint runtime hoặc các notebook inference hiện hữu.
