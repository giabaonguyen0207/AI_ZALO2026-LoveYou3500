# DataMate · Làm việc nhóm trên GPU

DataMate là ứng dụng Python dùng chung cho **Colab, Kaggle, GPU thuê và Docker**. Máy cá nhân dùng để viết code và kiểm thử CPU. **Pipeline 6** giữ Transformers làm baseline; đã ghép adapter vLLM và RAG, chọn độc lập qua config/CLI. Đã có [đối chiếu GPU100 câu phát triển ngày14/11](docs/comparison-100-review-2026-11-14.md); báo cáo nêu rõ training overlap, deadline và metrics thiếu. Chưa xác nhận image thi/private-test.

| Người | Nhánh riêng | Phần phụ trách / kế hoạch |
|---|---|---|
| Hải | `codex/hai-vllm` | Backend vLLM — [plan kỹ thuật](docs/plans/plan-integrate-vllm-2026-11-09.md) |
| Thịnh | `rag` (nhánh nhận ngày 14/11) | Retrieval/reranker — [plan kỹ thuật](docs/plans/plan-integrate-rag-2026-11-09.md) |

Kiến trúc đã chọn: **Adapter + Strategy** trong modular monolith. Mỗi người tự chọn benchmark, chạy baseline trước rồi đo bản tích hợp trên **cùng bộ đã chọn**. [Kiến trúc](docs/architecture.md) · [Ngữ cảnh cuộc thi](docs/competition-context.md).

## 1. Bố cục

```text
src/datamate/       Agent, data/SQL, models, evaluation, training, CLI
tests/              Kiểm thử bằng fixture; không cần full weights
configs/            Preset model và cấu hình submission
notebooks/
  DataMate_Colab_Inference.ipynb  Notebook GPU chính, chọn nền tảng ở ô đầu
  DataMate_Colab_Finetuned_Compare.ipynb  Bảo: model stage2 hoàn tất, ba cấu hình đối chứng
  DataMate_Colab_Team200.ipynb    Benchmark 200 câu tùy chọn
  training/                     Ghi chú training; notebook thử cũ giữ trong artifact frozen
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

**Bảo chạy Colab Pro với bản fine-tune stage 2 hoàn tất (1 + 0,5 epoch):** dùng [notebook đối chứng](notebooks/DataMate_Colab_Finetuned_Compare.ipynb) và [hướng dẫn đầy đủ](docs/colab-finetuned-compare.md). Nguồn là `qwen35-9b-sql-stage2-v1-full-01/adapter`, giữ report hoàn tất gốc. Merge một lần trên Colab với đúng base đã train; cả baseline/vLLM/RAG dùng cùng full model đã seal. Kết quả100 câu và giới hạn được lưu trong báo cáo ngày14/11 phía trên. Notebook này thay bản checkpoint198 để có một đường chạy chính.

Notebook fine-tuned dùng **venv inference riêng**, tránh xung đột requests/Pillow giữa vLLM và thư viện kernel Colab. Kết quả `pip check` và freeze được lưu trên Drive; tất cả process inference dùng cùng Python trong venv.

Nếu baseline đã xong nhưng vLLM lỗi `preprocessor_config.json`, xem [cách chạy tiếp trên source cũ](docs/colab-finetuned-compare.md#nếu-baseline-xong-nhưng-vllm-lỗi-image-processor): tool recovery bật chế độ chỉ xử lý văn bản, giữ baseline và chạy riêng vLLM/RAG trên cùng tập.

```powershell
.\.venv\Scripts\python.exe tools/package_colab.py
```

Gói mới ở `outputs/packages/datamate-colab-source.zip`, kèm manifest SHA256; không chứa dữ liệu, weights hoặc archive. Upload notebook `notebooks/DataMate_Colab_Inference.ipynb`, chọn `PLATFORM`, rồi sửa đường dẫn source/data/model ở ô đầu. Trên máy GPU đã clone repo có thể dùng CLI trực tiếp.

Notebook và CLI chạy `application.run_batch()` trong tiến trình được giám sát deadline: kiểm checkpoint, chuẩn bị RAG, load generator, chạy từng câu và lưu `submission.jsonl`/`logs/`. Qwen 9B/Gemma 12B dùng checkpoint đã pin; [model và inventory](docs/model-comparison.md). Bước tải/stage model tách khỏi inference offline.

Chọn backend bằng `--backend torch` (alias của `transformers`) hoặc `--backend vllm`; RAG bằng `--rag-mode none|dense|dense_rerank`. Ví dụ trên GPU với các checkpoint đã chuẩn bị:

```bash
python -m datamate run --config configs/comparison/baseline.json --backend torch --rag-mode none --split-root /work/data/train --result-dir outputs/runs/hai/baseline-01
python -m datamate run --config configs/comparison/vllm.json --backend vllm --rag-mode none --split-root /work/data/train --result-dir outputs/runs/hai/vllm-01
python -m datamate run --config configs/comparison/rag.json --backend torch --rag-mode dense_rerank --split-root /work/data/train --result-dir outputs/runs/thinh/rag-01
```

Có thể chọn `--backend vllm --rag-mode dense_rerank` để bật cả hai. Cài extras tương ứng `.[llm]`, `.[vllm]`, `.[llm,rag]` hoặc `.[vllm,rag]`; phải kiểm stack/VRAM trên GPU thật.

Để tự chạy **baseline → baseline + vLLM → baseline + RAG** trên cùng tập, dùng `tools/compare_system.py` theo [hướng dẫn GPU](docs/run-gpu.md#so-sánh-tự-động-ba-cấu-hình). Công cụ cố định source/IDs/dtype, lưu thời gian startup/inference/VRAM và chỉ chấm nhãn sau inference. [Báo cáo hợp nhất 14/11](docs/integration-vllm-rag-2026-11-14.md) nêu phần giữ lại, lỗi đã sửa và giới hạn kiểm chứng.

Mỗi cấu hình dùng run name mới. Không resume kết quả cũ khi code/model/config/bộ câu đổi. Benchmark Team200 là lựa chọn có sẵn, xem [cách dùng](docs/team200-colab.md).

## 4. Làm việc qua GitHub

Hải/Thịnh tiếp tục dùng nhánh và working copy/worktree riêng; nhánh RAG nhận thực tế là `rag`, tên dự kiến cũ trong plan là `codex/thinh-rag`. Bản ghép vào `codex/team-gpu-layout`; đưa lên `main` qua PR sau khi chủ dự án chọn. Hai plan ghi quyền sửa file/contract để tránh đè code; không cùng chuyển branch trong một checkout. Đọc [hướng dẫn cập nhật main và cho nhóm pull](docs/team-workflow.md).

Sau khi main được cập nhật:

```powershell
git status
git switch main
git pull --ff-only origin main
.\tools\setup.ps1
```

Nếu có code chưa commit, lưu vào nhánh của mình trước khi chuyển nhánh. Pull không xóa cache/model cũ trên máy đồng đội và không tự thay notebook/source ZIP trên Drive/Kaggle.

## 5. Docker nộp cuối

Đã đối chiếu PDF Docker BTC nhận14/11/2026: `/code/predict.sh` không đối số, máy **2×RTX5090/32GB mỗi card**, RAM512GB,6giờ gồm load, image Linuxamd64, TAR chưa nén<=100GB và nộp **TAR.GZ + MD5 của file nén**. Image phải có cả training source/data/log/config/checkpoints và môi trường training offline.

[Hướng dẫn Docker đầy đủ](docs/docker.md) · [Thuê GPU Vast và chuẩn bị trước giờ tính phí](docs/run-vast.md). `configs/submission.json` chốt ứng viên **vLLM + RAG**, generator GPU0/RAG GPU1. Full merged model đã verify đúng bản fine-tune stage2; không dùng base chưa fine-tune. `docker/prepare_release.py` chuẩn bị wheelhouse Linux và context có chọn lọc; `package_solution.py` từ chối release thiếu training evidence. `docker/rehearse.sh` dùng lệnh BTC cho hai lượt offline; `verify_submission.py` kiểm output/log; `export_solution.py` kiểm TAR/GZIP và tạo MD5.

**Chưa có image thi được chứng nhận GPU.** Base vLLM0.33 đã inspect dùng CUDA13, trong khi guideline chỉ cam kết R570+; phải giải quyết tương thích driver hoặc xác nhận qua Test Solution trước khi chốt. CPU/tiny smoke không thay full9B+RAG trên5090. Thư viện/weights được cài sẵn trong image vẫn cần import/nạp GPU/JIT lúc khởi động.

[Nhật ký lỗi runtime 14/11 và kiểm tra trước khi nộp](docs/runtime-issues-2026-11-14.md) lưu các lỗi dependency, text-only, Ninja/PATH, OOM và cảnh báo benchmark để tránh lặp lại khi đóng Docker.

Ràng buộc: tổng model inference ≤15B; checkpoint công khai trước 01/10/2026 và giấy phép phù hợp; `/data` chỉ đọc; kết quả `/result/submission.jsonl`, log `/result/logs/`; toàn private test ≤6 giờ gồm load. Không dùng public/private test để training.

## Tài liệu khác

- [Training trên GPU](docs/training.md) · [Đọc log](docs/debug-step-logs.md).
- [Rà soát hai model RAG](docs/rag-model-review-2026-11-09.md).
- [Báo cáo dọn bố cục](docs/team-layout-2026-11-09.md).
- Các báo cáo cũ ở `docs/history/`; kết quả lịch sử không thay cho baseline của tính năng mới.
