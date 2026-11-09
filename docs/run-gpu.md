# Chạy DataMate trên GPU

Dùng cùng source và CLI trên Colab, Kaggle hoặc GPU thuê. Không copy vòng agent sang notebook. Đọc [README](../README.md) trước.

## Chuẩn bị

1. Nhận dataset đúng split và checkpoint đã seal từ kho artifact riêng; chúng không có trong Git.
2. Nếu chưa có checkpoint, chạy stage-model online từ revision trong registry, trước khi đo inference.
3. Cài bộ thư viện tương thích GPU thực tế. Với Colab/Kaggle đã có PyTorch CUDA, không tự thay nó khi chưa kiểm tra; notebook tạo constraints theo bản PyTorch hiện có.
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

Đổi code trên Drive/Kaggle: upload ZIP mới, restart kernel và chạy lại từ đầu. Notebook giải nén vào workspace theo SHA256 ZIP; nếu kernel còn module từ source khác thì dừng và yêu cầu restart. Không trộn module Python cũ với ZIP mới.

## Đo baseline và bản tích hợp

Chạy baseline trước, rồi bản tích hợp trên đúng bộ câu đó. Giữ generator/checkpoint và điều kiện đo tương đương; vLLM cần nhanh hơn mà không giảm chất lượng, RAG cần cải thiện đáp án. Không bắt buộc Team200.

Batch đo từ trước hash/load model đến flush output; thời gian chép model từ kho vào scratch trước batch phải ghi riêng và cộng khi so tổng thời gian. Khởi tạo/index/compile trong inference đều tính chi phí. So cold/warm cùng chính sách, không gọi số đo giữa hai GPU khác nhau là speedup thuần của code.

Mỗi run có submission, logs, run_identity và run_summary. Dùng scorer trên nhãn phù hợp sau khi generation kết thúc:

```bash
python -m datamate score --predictions outputs/runs/hai/baseline-01/submission.jsonl --references /work/data/train/answers.jsonl --output outputs/runs/hai/baseline-01/score.json
```

Không cho nhãn vào prompt/index. Resume chỉ khi input/source/config/model giữ nguyên; bản tích hợp dùng thư mục kết quả mới. Chi tiết RAG/vLLM ở [hai plan](plans/).
