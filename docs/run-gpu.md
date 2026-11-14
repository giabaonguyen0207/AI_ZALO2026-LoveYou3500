# Chạy DataMate trên GPU

Dùng cùng source và CLI trên Colab, Kaggle hoặc GPU thuê. Không copy vòng agent sang notebook. Đọc [README](../README.md) trước.

## Chuẩn bị

1. Nhận dataset đúng split và checkpoint đã seal từ kho artifact riêng; chúng không có trong Git.
2. Nếu chưa có checkpoint, chạy stage-model online từ revision trong registry, trước khi đo inference.
3. Cài bộ thư viện tương thích GPU thực tế. Torch dùng extra `llm`, vLLM dùng `vllm`; thêm `rag` khi bật retrieval. vLLM có thể yêu cầu đổi PyTorch; notebook phát hiện gói đã import bị đổi và yêu cầu restart trước inference. Dải dependency là ứng viên, chưa phải lock GPU đã xác nhận.
4. Lưu `pip freeze`, tên GPU, driver, source/config/model manifest và bộ câu cho mỗi phép so.

Lệnh trên Linux GPU sau khi clone hoặc giải nén source, chọn đúng Python environment:

```bash
python -m pip install -e '.[llm]'
python -m datamate models
python -m datamate stage-model --model qwen35-9b --output models/qwen35-9b
python -m datamate run --config configs/qwen35-9b.json --split-root /work/data/train --result-dir outputs/runs/hai/baseline-01
```

Preset dùng đường dẫn model mặc định `models/qwen35-9b` tương đối project. Nếu checkpoint ở `/work/models`, sửa `model_dir` trong một preset riêng (không sửa preset baseline sau khi đo), hoặc dùng notebook để resolve đường dẫn. Không coi việc cài được package là đã xác nhận model fit VRAM.

## Notebook chính

`notebooks/DataMate_Colab_Inference.ipynb` có bốn ô code: cấu hình/input → cài dependency → chuẩn bị checkpoint/config → chạy batch. Chọn `PLATFORM` rồi sửa đường dẫn theo máy:

| Nền tảng | Đường dẫn mẫu | Lưu ý |
|---|---|---|
| Colab | Source/model trên Drive; workspace /content/datamate | Có thể chép model vào scratch để giảm I/O Drive |
| Kaggle | Input dưới /kaggle/input; output /kaggle/working | Tên dataset là ví dụ; tải kết quả trước khi kết thúc phiên |
| GPU thuê | /work/datamate, /work/data, /work/models | Dùng ổ persistent cho model và kết quả |

`TRAIN_ROOT` là tên biến tương thích, có thể trỏ tới bất kỳ split/benchmark phù hợp; cần data/, rulebook.md, calendar.csv và questions.jsonl. `TRAIN_ZIP` tùy chọn chỉ dùng helper xác minh gói TRAIN gốc; bộ benchmark khác cần giải nén trước.

`MODEL_SOURCE_DIR` phải chứa artifact_manifest.json. Không tải model ngầm trong inference. `MODEL_KEY` mặc định qwen35-9b; đổi Gemma thì chọn preset tương ứng. `IDS_FILE` có thể là JSON array ID để chọn tập; `RUN_LIMIT` để thử ít câu. Bộ đo và số câu do từng người chọn.

`BACKEND=None` và `RAG_MODE=None` lấy lựa chọn từ config/env. Gán `BACKEND="torch"|"vllm"`, `RAG_MODE="none"|"dense"|"dense_rerank"` để ghi đè. Hai ô cài đặt và chuẩn bị dùng cùng Settings, tránh cài Torch nhưng chạy config vLLM. RAG dùng `EMBEDDING_SOURCE_DIR`/`RERANKER_SOURCE_DIR`, có thể sửa theo nơi đã upload; notebook kiểm/copy mọi checkpoint đang bật, không chỉ generator.

Đổi code trên Drive/Kaggle: upload ZIP mới, restart kernel và chạy lại từ đầu. Notebook giải nén vào workspace theo SHA256 ZIP; nếu kernel còn module từ source khác thì dừng và yêu cầu restart. Không trộn module Python cũ với ZIP mới.

## Đo baseline và bản tích hợp

Chạy baseline trước, rồi bản tích hợp trên đúng bộ câu đó. Giữ generator/checkpoint và điều kiện đo tương đương; vLLM cần nhanh hơn mà không giảm chất lượng, RAG cần cải thiện đáp án. Không bắt buộc Team200.

Batch đo từ trước hash/load model đến flush output; thời gian chép model từ kho vào scratch trước batch phải ghi riêng và cộng khi so tổng thời gian. Khởi tạo/index/compile trong inference đều tính chi phí. So cold/warm cùng chính sách, không gọi số đo giữa hai GPU khác nhau là speedup thuần của code.

Mỗi run có submission, logs, run_identity và run_summary. Dùng scorer trên nhãn phù hợp sau khi generation kết thúc:

```bash
python -m datamate score --predictions outputs/runs/hai/baseline-01/submission.jsonl --references /work/data/train/answers.jsonl --output outputs/runs/hai/baseline-01/score.json
```

Không cho nhãn vào prompt/index. Resume chỉ khi input/source/config/model giữ nguyên; bản tích hợp dùng thư mục kết quả mới. Chi tiết RAG/vLLM ở [hai plan](plans/).

## So sánh tự động ba cấu hình

Chạy trên **một environment GPU sạch**, cài và kiểm dependency chung trước khi đo. Không đổi phiên bản thư viện giữa các arm. Ví dụ (các đường dẫn là chỗ đội tự đặt):

```bash
python -m pip install -e '.[llm,vllm,rag]'
python -m pip check
python -m datamate stage-model --model vietnamese-embedding --output /work/models/vietnamese-embedding
python -m datamate stage-model --model vietnamese-reranker --output /work/models/vietnamese-reranker
python tools/compare_system.py --config configs/comparison/baseline.json --ids /work/benchmark/ids.json --split-root /work/data/train --model-dir /work/models/qwen35-9b --embedding-dir /work/models/vietnamese-embedding --reranker-dir /work/models/vietnamese-reranker --output outputs/runs/team/compare-01
```

Generator cũng phải đã stage/seal; có thể dùng checkpoint fine-tune đã merge/seal đúng lineage, dùng cùng artifact cho cả ba arm. Chỉ stage khi chưa có bản đúng; không ghi đè checkpoint fine-tune bằng base. Thêm `--references /work/benchmark/references.jsonl` nếu có nhãn hợp lệ: nhãn chỉ đọc sau khi cả ba arm đã inference xong. Không có nhãn thì báo đồng thuận/khác biệt đáp án, **không kết luận RAG tăng chất lượng**.

Mặc định cùng FP16, một question worker, mỗi arm tiến trình và cache riêng, prefix cache tắt. Nếu phần cứng cần chia model, truyền `--vllm-tensor-parallel-size 2`; chọn số GPU theo thực tế, không coi cấu hình này là bảo đảm fit VRAM. `--vllm-workers N` là thử nghiệm concurrency riêng, cần ghi rõ khi khác 1. Có thể dùng `--dtype bfloat16` cho **tất cả** arm nếu GPU hỗ trợ. Ngân sách startup vLLM trong preset là 1200s; thời gian này vẫn nằm trong wall budget, không được trừ khi tính tổng.

Output gồm `selected_ids.json`, ba config resolved, log từng arm, thư mục `baseline/`, `vllm/`, `rag/`, `process_timing.json` và `comparisons.json`. Trong từng arm:

- `benchmark_identity.json`: source hashes, full inventory/manifest, config và dependency.
- `benchmark_metrics.json`: startup, inference, tổng trong batch, token, timeout, VRAM/RAM lấy mẫu; VRAM là mức dùng toàn GPU, cần GPU không có workload khác.
- `question_timing.jsonl`, `model_calls.jsonl`, `logs/`: độ trễ, hash prompt và nguồn RAG.
- `process_timing.json` bên ngoài arm tính cả bootstrap/import/ghi metadata/dọn tiến trình. `cold_total_seconds` bên trong batch không bao gồm cài đặt/tải/copy model và không kiểm soát OS filesystem cache.

Comparator bác bỏ thay đổi model/dữ liệu/ngân sách/mã nghiệp vụ ngoài arm cho phép. So backend cần cùng cấu hình RAG và prompt đầu; so RAG cho phép context thay đổi nhưng kiểm hash prompt nền và cùng source. Câu timeout vẫn được giữ trong báo cáo; không xác nhận speedup sạch khi có lỗi/deadline. Các ratio là tốc độ **pipeline**, không phải tốc độ riêng kernel vLLM. Chạy thêm lượt/đổi thứ tự có kiểm soát nếu chênh lệch nhỏ vì OS cache có thể có lợi cho lượt sau.

`tools/kaggle_workflow.py` và notebook benchmark cũ của Hải được giữ để tái hiện phép đo lịch sử hai source. Dùng công cụ ba arm ở trên cho bản hợp nhất hiện tại.

## Lifecycle RAG và deadline

`none` giữ nguyên prompt baseline; `dense` dùng encoder; `dense_rerank` thêm chấm cặp. Corpus hiện gồm rulebook, README và schema raw trong đúng split/business; chưa index cell value/đáp án/log. SQL vẫn tính trên toàn bộ dòng. Chunk theo tokenizer embedding, context giới hạn bằng tokenizer generator, giảm nguyên chunk trước mỗi generate khi history tăng.

Model/index được chuẩn bị trước câu đầu trong wall budget. Cache index có checksum/identity, nằm ngoài dataset; RAG chậm/lỗi truy vấn phục hồi thì có log fallback, lỗi load model/OOM làm lượt chạy thất bại rõ ràng. RAG được load trước generator để engine thấy VRAM còn lại; vẫn phải đo VRAM theo cấu hình thực tế. Hai encoder thêm tổng 1.135.510.529 tham số; Qwen 9B + cả hai là 10.788.614.897, dưới 15B; không ghép hai encoder vào model upscaled 14B vượt cap.

CLI `run` và notebook dùng worker process có watchdog startup/wall; trên Linux dọn cả process group vLLM. API gọi trực tiếp `application.run_batch/create_agent` trong một process chỉ có timeout phối hợp với SDK, không thể cưỡng bức dừng một hàm SDK đang chặn thread. Dùng entrypoint được giám sát khi chạy batch đánh giá/nộp bài.
