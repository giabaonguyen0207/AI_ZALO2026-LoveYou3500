# DataMate · Làm việc nhóm trên GPU

DataMate là ứng dụng Python dùng chung cho **Colab, Kaggle, GPU thuê và Docker**. Máy cá nhân dùng để viết code và kiểm thử CPU. Baseline hiện là **pipeline 6 / Transformers**; vLLM và RAG đang được tích hợp, chưa có số đo mới từ đợt dọn bố cục.

| Người | Nhánh riêng | Phần phụ trách / kế hoạch |
|---|---|---|
| Hải | `codex/hai-vllm` | Backend vLLM — [plan kỹ thuật](docs/plans/plan-integrate-vllm-2026-11-09.md) |
| Thịnh | `codex/thinh-rag` | Retrieval/reranker — [plan kỹ thuật](docs/plans/plan-integrate-rag-2026-11-09.md) |

Kiến trúc đã chọn: **Adapter + Strategy** trong modular monolith. Mỗi người tự chọn benchmark, chạy baseline trước rồi đo bản tích hợp trên **cùng bộ đã chọn**. [Kiến trúc](docs/architecture.md) · [Ngữ cảnh cuộc thi](docs/competition-context.md).

## 1. Bố cục

```text
src/datamate/       Agent, data/SQL, models, evaluation, training, CLI
tests/              Kiểm thử bằng fixture; không cần full weights
configs/            Preset model và cấu hình submission
notebooks/
  DataMate_Colab_Inference.ipynb  Notebook GPU chính, chọn nền tảng ở ô đầu
  DataMate_Colab_Team200.ipynb    Benchmark 200 câu tùy chọn
  training/                     Notebook training
tools/              Tạo notebook, kiểm dữ liệu, đánh giá và đóng gói
docker/             Docker dev và Docker competition
docs/               Hướng dẫn hiện hành, plans/, history/, references/, research/
outputs/
  runs/hai/         Kết quả của Hải, mỗi run một thư mục
  runs/thinh/       Kết quả của Thịnh
  packages/         ZIP source, build context và artifact mới
models/             Staging checkpoint khi cần build Docker
train/, public_test/ Dữ liệu gốc
archive/            Bản lưu local, không đưa lên Git hoặc gói source
```

Dữ liệu, trọng số, output, cache và môi trường Python **không đi qua Git**. Người mới pull source vẫn cần nhận dataset/checkpoint riêng: xem [artifact và lưu trữ](docs/artifacts.md). Không còn Qwen 4B, preset laptop hay web local. Transformers và DemoBackend được giữ để làm baseline và kiểm thử.

## 2. Bắt đầu trên máy cá nhân

Yêu cầu Python 3.11+ và Git. Trong PowerShell tại thư mục dự án:

```powershell
.\tools\setup.ps1
.\.venv\Scripts\python.exe -m pytest -q
```

Để chạy cả kiểm thử tokenizer/model ngẫu nhiên nhỏ bằng **CPU**, cài thêm:

```powershell
.\tools\setup.ps1 -ProtocolTests
.\.venv\Scripts\python.exe -m pytest --basetemp .cache/pytest -q
```

Không dùng môi trường này cho full-checkpoint inference. Dependency GPU được cài và ghi phiên bản trên máy GPU thực tế; `requirements-dev.lock.txt` chỉ khóa môi trường phát triển nhẹ.

## 3. Chạy trên Colab/Kaggle/GPU thuê

Đọc [hướng dẫn GPU](docs/run-gpu.md). Tạo gói source trên máy cá nhân:

```powershell
.\.venv\Scripts\python.exe tools/package_colab.py
```

Gói mới ở `outputs/packages/datamate-colab-source.zip`, kèm manifest SHA256; không chứa dữ liệu, weights hoặc archive. Upload notebook `notebooks/DataMate_Colab_Inference.ipynb`, chọn `PLATFORM`, rồi sửa đường dẫn source/data/model ở ô đầu. Trên máy GPU đã clone repo có thể dùng CLI trực tiếp.

Notebook và CLI cùng gọi `application.run_batch()`: kiểm checkpoint, load model, chạy từng câu, lưu `submission.jsonl` và `logs/`. Qwen 9B/Gemma 12B dùng checkpoint đã pin trong registry; [model và inventory](docs/model-comparison.md). Bước tải/stage model tách khỏi inference offline.

Mỗi cấu hình dùng run name mới. Không resume kết quả cũ khi code/model/config/bộ câu đổi. Benchmark Team200 là lựa chọn có sẵn, xem [cách dùng](docs/team200-colab.md).

## 4. Làm việc qua GitHub

Hải dùng `codex/hai-vllm`, Thịnh dùng `codex/thinh-rag`, mỗi người một working copy/worktree riêng và PR vào `main`. Hai plan ghi rõ file cần tạo/sửa, contract, người phụ trách từng file và thứ tự PR nền trước khi phát triển tính năng song song; không cùng chuyển branch trong một checkout. Đọc [hướng dẫn cập nhật main và cho nhóm pull](docs/team-workflow.md).

Sau khi main được cập nhật:

```powershell
git status
git switch main
git pull --ff-only origin main
.\tools\setup.ps1
```

Nếu có code chưa commit, lưu vào nhánh của mình trước khi chuyển nhánh. Pull không xóa cache/model cũ trên máy đồng đội và không tự thay notebook/source ZIP trên Drive/Kaggle.

## 5. Docker nộp cuối

Giữ `docker/Dockerfile.competition`, `configs/submission.json` và `predict.sh`. Dùng `tools/package_solution.py` để tạo build context chỉ chứa source, checkpoint được chọn và wheelhouse.

[Hướng dẫn build, chạy offline và xuất image](docs/docker.md). Cần đối chiếu base image/guideline BTC và kiểm trên GPU thật trước ngày nộp; ngày cuối chốt image/checksum. Đợt dọn này **chưa build image thi hoặc đo inference GPU**.

Ràng buộc: tổng model inference ≤15B; checkpoint công khai trước 01/10/2026 và giấy phép phù hợp; `/data` chỉ đọc; kết quả `/result/submission.jsonl`, log `/result/logs/`; toàn private test ≤6 giờ gồm load. Không dùng public/private test để training.

## Tài liệu khác

- [Training trên GPU](docs/training.md) · [Đọc log](docs/debug-step-logs.md).
- [Rà soát hai model RAG](docs/rag-model-review-2026-11-09.md).
- [Báo cáo dọn bố cục](docs/team-layout-2026-11-09.md).
- Các báo cáo cũ ở `docs/history/`; kết quả lịch sử không thay cho baseline của tính năng mới.
