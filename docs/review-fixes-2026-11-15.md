# Bản sửa inference sau rà soát Public1000 — 15/11/2026

Branch làm việc: `check15_11`. Không chuyển branch, merge, stage hay commit. Hướng dẫn cho người chạy lần đầu: [huogndan.md](../../huogndan.md).

Chủ dự án bổ sung: public đã chạy A100; Docker đã được đội thử qua cổng BTC và không gặp lỗi môi trường theo thông tin cung cấp. Bản sửa giữ stack/interface đang chạy được; không xử lý lỗi chất lượng bằng thay CUDA/vLLM/Torch/model.

## Thay đổi đã thực hiện

| File | Hành vi mới | Giới hạn/guard |
|---|---|---|
| `src/datamate/agent/prompts.py` | Chia sẻ mô tả source schema giống nhau bằng `same_schema_as` | Giữ toàn bộ tên bảng raw/clean; không gộp hay bỏ dữ liệu; giữ question/rulebook/README |
| `src/datamate/agent/context.py`, `runner.py` | Kiểm token trước generation; chỉ khi overflow history mới checkpoint plan/results/evidence, catalog updates sau repair và trao đổi cuối | Giữ input gốc và output reserve; không fit thì báo lỗi, không cắt luật. Demo/scripted backend không cần tokenizer. Tính lại deadline sau bước packing |
| `src/datamate/data/binding.py` | Hiểu file template trong heading README; thêm alias phổ thông; STT thành mã đơn chỉ với định nghĩa README; nhận diện note-derived status có evidence | Không áp định nghĩa notes thuộc file khác; cột status thật có ưu tiên; bare STT vẫn là số dòng |
| `src/datamate/data/normalization.py` | Diễn giải notes đã được tài liệu hóa; dùng mặc định completed chỉ khi split quy định; năm filename khi sheet thiếu năm và filename có đúng một năm; bỏ adjustment phi hàng đã khai báo khỏi sales facts | Không biến bảng thiếu status thành completed; phân biệt phủ định/mâu thuẫn và tên Huy; adjustment lỗi parse không bị bỏ qua; raw/clean giữ nguyên; chuẩn hóa rulebook một lần thay vì mỗi dòng |
| `src/datamate/agent/schema_repair.py` | Truyền allowed table/policy scope; retry một lần khi parse/action envelope lỗi; cache lỗi tạm trong worker thay vì seal lỗi xuống disk | Giữ tổng20s/512token mỗi call; không retry fatal engine; giữ citation/validation và baseline; từng attempt có raw trace/model call riêng. Proposal hợp lệ vẫn được kiểm/cached theo logic an toàn sẵn có |
| `runner.py::_repair` | Khi unanswerable chưa có evidence, chỉ rõ cần action check và khác biệt missing/unsupported | Không bỏ guard hay đổi query rỗng thành bằng chứng thiếu dữ liệu |
| `tools/verify_submission.py` | Kiểm số dòng sau resume bằng written + previously_completed | Vẫn đòi đủ ID/log, answer khớp và summary sạch lỗi |
| `tools/colab_review.py` | Stage ZIP/folder một split, hash input, không copy gold vào split inference; export theo thứ tự input, diagnostics | Public không cho LIMIT; từ chối archive traversal và export đè đáp án khác |
| `tools/create_review_notebook.py`, `notebooks/DataMate_Colab_Review.ipynb` | Notebook source hiện tại, source ZIP+manifest gate, freeze gate, model cache, CLI/supervisor và export | Không sửa notebook Public1000 cũ; pin đúng generator manifest; inference offline; không tự submit |
| `docker/Dockerfile.source-update` | Overlay source trên chính image đội đã thử, không có RUN/pip/apt | Kế thừa model/libs/entrypoint/config; người giữ image cần kiểm đường import và test image mới |
| `docs/competition-context.md` | Ghi cập nhật người dùng về Docker đã chạy thử | Phân biệt xác nhận của chủ dự án với log/image digest chưa nhận |

Không đổi `pyproject.toml`, requirements/lock, `configs/submission.json`, constraints, CLI/Settings/Answer schema, `predict.sh`, Dockerfile competition gốc hoặc vLLM backend. Không thay weights, model registry, dtype, quantization, output tokens, số bước hay timeout mặc định.

Notebook default giữ cấu hình Colab đã dùng: một GPU, RAGcuda:0, vLLM fraction0,75. Các profile concurrency2/4, prefix và context20k là tùy chọn riêng, không tự ghi đè config Docker.

## Đọc lại luồng sau khi code

Đã đối chiếu source sau sửa theo chuỗi:

1. Notebook kiểm source/freeze và tách input khỏi gold; ba model vẫn là sealed artifacts cũ. Venv mới tái tạo freeze cũ, hoặc dùng Python venv đã có mà không pip install.
2. Notebook gọi cùng CLI `python -m datamate run --config`; CLI đi qua `execution.run_isolated` → `application.run_batch` → `create_agent`. Docker vẫn đi từ `/code/predict.sh` vào cùng đường này.
3. RAG chuẩn bị raw catalog/index trước generator như cũ; corpus không lấy answers. Bản sửa schema không làm thay đổi tập giao dịch do retrieval chọn.
4. Ingest vẫn giữ raw; semantic cache có version hash của các module đã đổi nên không dùng lại facts cũ sai. Notes chỉ được diễn giải qua binding có evidence; status thật/missing không bị áp nhầm mặc định notes.
5. Prompt compact chỉ giảm lặp mô tả. Context checkpoint giữ plan, query IDs/results, evidence IDs và catalog updates sau repair, kể cả qua nhiều lần checkpoint; guard output/type/SQL/answerability/self-review vẫn chạy. Nếu không đủ chỗ thì abstain rõ thay vì sinh SQL bị cắt.
6. Schema repair vẫn dùng cùng generator; retry nằm trong cùng question/repair deadline, mỗi call xuất hiện riêng trong trace/count. Không lưu parse error vào persistent cache để nó không đầu độc các run sau.
7. Batching chia sẻ engine, có writer output duy nhất như cũ. Mặc định1worker; các thử nghiệm concurrency phải đổi cả worker và engine max_num_seqs.
8. Output ghi từng câu, không sửa giá trị khi export. Resume phải khớp source/config/input/models; verifier kiểm tổng câu cũ+mới.

Trong lượt đọc lại đã phát hiện và xử lý hai lỗi triển khai trước khi hoàn tất: so sánh hai giá trị None có thể kích hoạt nhầm default notes cho bảng không status; việc normalize nguyên rulebook trong mỗi lần xử lý một dòng có thể làm tăng CPU time. Đã thêm regression cho trường hợp thứ nhất và chuyển default policy thành giá trị tính một lần.

Không thay thuật toán header nhiều tầng một cách suy đoán: XML của workbook mẫu có dòng group-header rồi một dòng leaf-header đầy đủ, reader hiện chọn được dòng leaf. Bộ alias/README mapping được cải thiện; không tuyên bố reader xử lý được mọi kiểu merged header khác.

## Kiểm chứng của lần sửa đầu

**154 tests passed, 8,11 giây**, bằng `.venv/Scripts/python.exe`, fixture CPU, không nạp weights/LLM/Torch/vLLM/Docker:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .cache/pytest-review-20261115-e --tb=short `
  tests/test_review_regressions.py tests/test_colab_review.py `
  tests/test_pipeline_fast.py tests/test_pipeline_v3.py `
  tests/test_ingestion_sql.py tests/test_semantic_pipeline.py `
  tests/test_shared_quality.py tests/test_concurrent_batch.py `
  tests/test_verify_submission.py tests/test_colab_model_cache.py tests/test_colab_bootstrap.py
```

Để tự chạy lại dùng **basetemp mới** thay hậu tố `-e`, tránh đụng output test trước. Các test kiểm preservation của prompt/state/catalog updates, schema family, note scope/negation/missing, phi hàng, SQL read-only/quality, plan compiler/self-review, repair retry/cache, batching/resume và notebook/export.

Lần thử có `test_rag_integration.py` không collect được vì môi trường nhẹ chưa cài NumPy; không cài thêm theo yêu cầu giữ môi trường. Bộ test cuối phía trên không chứa file đó. Lỗi quyền thư mục temp mặc định của Windows được tránh bằng basetemp nằm trong workspace. Đây không phải full GPU test hoặc full test suite.

Đã parse syntax notebook mới, kiểm notebook khớp builder, `git diff --check`, và so diff nhóm file dependency/interface được bảo vệ: không đổi.

## Những việc chưa được khẳng định

- Chưa chạy model trên Colab/5090; chưa đo PA, latency, peakVRAM hoặc prefix-cache hits cho source mới. **Không khẳng định test CPU qua sẽ tăng PA bao nhiêu.**
- Chưa build/chạy image overlay. Image cũ chạy được không chứng minh hành vi model/source mới giữ đúng score; cần chạy thử theo hướng dẫn.
- Chưa làm chứng minh ngữ nghĩa tổng quát cho mọi customSQL, scope STT phức tạp, ledger/trả góp/bảo hành hay mọi ngành mới. Guard còn có thể abstain khi chưa hỗ trợ.
- Không thêm spell-correction model, không đổi model lớn hơn hoặc quantization. Các phương án đó cần benchmark và budget model riêng.
- Checkpoint history có thể làm model cần gọi lại schema/profile cũ; kết quả và evidence giữ để không giả lập chứng cứ. Nếu checkpoint vẫn vượt context, dừng có chủ đích.

Gói source được tạo từ working tree bằng `tools/package_colab.py`, bao gồm thay đổi chưa commit và manifest hashes. Gói không chứa data/weights/venv/cache. Đóng gói source không phải chứng nhận Docker hay kết quả thi.

## Rà soát tiếp theo cùng ngày, sau khi chủ dự án xác nhận A100/2×5090

Đã đọc lại đường Colab và Docker từ đầu đến cuối; báo cáo hiện tại: [kiemtra_pipeline_2026-11-15.md](../../kiemtra_pipeline_2026-11-15.md). Hướng dẫn đóng từ các file hiện tại, không cần image hoàn chỉnh cũ: [huogndan_vast_docker.md](../../huogndan_vast_docker.md).

Sửa thêm matcher filename để chỉ `<...>` là placeholder; giới hạn bằng chứng dòng phi hàng ở đúng dòng README; gửi catalog mới về model cả khi repair thành công rồi plan/calculation lỗi; ưu tiên status_map đã validate hơn note default; giữ PYTHONPATH cũ khi dùng overlay. Tool chuẩn bị release có tùy chọn `--runtime-freeze` để khóa toàn bộ package versions và tái dùng training evidence đã kèm wheelhouse. Dockerfile competition, installer, predict.sh, config inference và dependency declarations vẫn nguyên.

Bộ kiểm tra cuối mở rộng đến artifact/build preparation, training evidence, mock export, fake vLLM lifecycle: **221 tests passed trong12,15s**, không GPU/model/Docker thật. Config public mặc định khớp JSON lượt A100 cũ sau chuẩn hóa đường dẫn. Đây thay trạng thái154test của lần đầu; không có PA hoặc GPU benchmark mới.

## Tích hợp tiếp correction/hybrid/concurrency theo yêu cầu mới

Chủ dự án đã làm rõ giữ interface/library versions nhưng cho phép đổi pipeline. Đã tích hợp correction checkpoint pin/hash vào inventory/staging/runtime/bundle/Docker, hybrid lexical+dense retrieval, profile optimized worker2/prefix/recovery2. Notebook Review cập nhật tại chỗ; Docker chọn configs/submission-optimized.json, config cũ là baseline. Báo cáo mới nhất: [pipeline_tichhop_2026-11-15.md](../../pipeline_tichhop_2026-11-15.md). **270 CPU fixture tests passed trong16,61s**, không model/GPU/Docker thật. Chưa có số PA hoặc speedup mới.
