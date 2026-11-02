# Docker và kiểm tra offline

`docker/Dockerfile.dev` là image CPU nhẹ để kiểm tra packaging/ingestion/JSONL; mặc định demo, không có trọng số. Base Python 3.13 slim pin digest sau lần build thật ngày 01/11/2026. Không dùng image này để tuyên bố accuracy/thời gian LLM.

## Smoke test

```powershell
docker build -f docker/Dockerfile.dev -t datamate-dev:local .
.\tools\docker-smoke.ps1
```

Helper chọn đúng 3 câu đầu train và chép dữ liệu các doanh nghiệp liên quan vào `outputs/docker-input-<suffix>`. Khi Docker Desktop không nhận bind mount Unicode, dùng Docker cp và volume riêng:

1. Chỉ chép subset nguồn của dự án, không mount toàn ổ E:.
2. Inference dùng `--network none`, `--read-only`, `/data` volume chỉ đọc, `/work` và `/tmp` là tmpfs.
3. Ghi submission/log rồi chép về `result/docker-smoke-<suffix>` trên E:.
4. Kiểm tra exit code, JSONL, ID, đáp án demo và log; dọn đúng container/volume vừa tạo.

Docker Desktop quản lý storage nội bộ theo cấu hình engine. Source và kết quả xuất về E:; helper không đổi global settings.

## Image thi

Chưa có Docker guideline chính thức, chưa xác nhận base image, entrypoint đầy đủ, CUDA/PyTorch/Blackwell. `docker/Dockerfile.competition` yêu cầu `--build-arg BASE_IMAGE=<official-image>`, không có mặc định giả làm ảnh ban tổ chức.

Sau khi nhận guideline:

- Đối chiếu Python, CUDA, PyTorch, native support model/assistant, bitsandbytes nếu dùng và RTX 5090.
- Dùng Linux builder tương thích tạo wheelhouse đầy đủ, gồm setuptools/wheel. Root `.dockerignore` cho phép wheelhouse/models trong build thi; dev có ignore riêng để không gửi trọng số lớn.
- Chuẩn bị model chính, assistant nếu chọn, merge adapter và seal artifact.
- Sửa `configs/submission.json`: `/data`, `/result`, `/work/datamate`, model trong `/app/models`.
- Build offline từ wheelhouse với base chính thức. `predict.sh` dùng `timeout`; xác nhận có công cụ đó.
- Chạy toàn split offline, đo wall time từ trước load, kiểm tra JSONL và log. Không dùng test/log test để training.

Ví dụ sau khi có đủ tài liệu, wheelhouse và model (chưa thực hiện):

```text
docker build -f docker/Dockerfile.competition --build-arg BASE_IMAGE=<official-base-image> -t datamate-solution:validated .
```

Lần kiểm tra đóng gói cuối dùng trọng số có sẵn trong image, mạng tắt, `/data` chỉ đọc. Batch ghi `/result/submission.jsonl` và `/result/logs/`; dừng ở 21.000 giây, watchdog ngoài dừng dưới 21.600 giây. Câu đã flush được giữ, còn thiếu tính abstain theo ngữ cảnh cuộc thi. Đây là cơ chế giới hạn, chưa bảo đảm throughput hoặc chất lượng.
