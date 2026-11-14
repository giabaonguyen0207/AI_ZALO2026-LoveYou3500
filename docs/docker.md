# Đóng gói Docker cuối

Chuẩn bị và chạy rehearsal trước ngày nộp; ngày cuối chốt image đã kiểm tra, xuất TAR/checksum và nộp theo guideline BTC. Hiện chưa xác nhận base image chính thức trong workspace. Không dùng Docker dev để tuyên bố model chạy đúng/tốc độ trên GPU thi.

## 1. Chuẩn bị trên Linux GPU

- Chốt commit/config và các checkpoint thực sự dùng; tổng tham số ≤15B, có revision/hash/license.
- Nếu dùng fine-tuned model, merge/seal và giữ training source/config/provenance cần bàn giao.
- Đối chiếu base image, Python/CUDA/GPU với guideline chính thức. Chuẩn bị image base và wheelhouse phù hợp trước bước build offline.
- Lưu dependency GPU đã chạy thực tế vào artifact của release. requirements-dev.lock.txt là khóa CPU nhẹ, không dùng làm chứng nhận stack GPU.
- Bản ghép đã stage theo full inventory: generator/assistant và embedding/reranker đang bật. Profile dependency là `llm|vllm` cộng `rag`, không dùng wheelhouse Torch-only để đóng vLLM hoặc RAG.

## 2. Tạo build context có chọn lọc

```bash
python tools/package_solution.py --config configs/submission.json --wheelhouse /work/wheelhouse --model-source qwen35-9b=/work/models/qwen35-9b --output outputs/packages/release-01/context
```

Trên workspace Windows hiện tại, Qwen 9B đã lưu nguyên bản ở archive/2026-11-09/models/qwen35-9b; có thể dùng đường dẫn đó khi stage. Khi chạy trên Linux, thay bằng vị trí checkpoint local tương ứng. Assistant nếu chọn cần thêm --model-source cho khóa assistant.

Helper kiểm manifest/hash, chỉ chép source, config submission, checkpoint được khai báo và wheelhouse; giữ training source trong src/datamate/training. Không chép train/public_test, archive, logs, notebook hay toàn bộ models/. Thư mục output phải rỗng; tạo release mới nếu đã tồn tại. solution_manifest.json ghi inventory/hash và trạng thái chưa build/chưa xác nhận GPU.

Công cụ staging không chọn base image thay BTC, không xác nhận wheel tương thích chỉ bằng tên file và không tự tải tài nguyên.

Nếu bật RAG, phải cung cấp cả checkpoint encoder/reranker (chỉ encoder khi `dense`). Cách tránh bỏ sót là chuẩn bị **bundle** trên Linux/Python/CUDA tương thích image mục tiêu:

```bash
python tools/prepare_runtime.py --config configs/comparison/rag.json --model-dir /work/models/qwen35-9b --rag-embedding-dir /work/models/vietnamese-embedding --rag-reranker-dir /work/models/vietnamese-reranker --artifact-root /work/artifacts/torch-rag --requirements /work/locks/torch-rag.txt
python tools/prepare_runtime.py --artifact-root /work/artifacts/torch-rag --offline
python tools/package_solution.py --bundle /work/artifacts/torch-rag/bundle.json --output outputs/packages/release-rag/context --base-image TEAM_VERIFIED_IMAGE_WITH_DIGEST
```

Đổi config/backend sang vLLM nếu chọn vLLM; tạo bundle/profile riêng cho lựa chọn cuối. `--requirements` là danh sách phiên bản đã kiểm trên GPU (ví dụ pip freeze đã rà soát); nó bổ sung ràng buộc nhưng không thay danh sách dependency bắt buộc của project/extra. Giai đoạn online resolve/tải wheel, ghi hash và bản khóa resolved; giai đoạn offline chỉ verify/cài wheel local, không sửa/download bundle. Không dùng lock Windows CPU để đóng image Linux GPU.

Nguồn checkpoint nonempty thiếu/sai manifest sẽ bị từ chối, không tự thay checkpoint fine-tune bằng base. `package_solution` viết lại đường dẫn tất cả model về `/app/models/<key>`, RAG index/scratch về `/work`, dữ liệu `/data`, output `/result`. Inventory phải đúng model thực dùng và không vượt 15B.

## 3. Build và rehearsal offline

Sau khi thay placeholder bằng base image đã đối chiếu, tải base trước rồi build trong môi trường Linux phù hợp:

```bash
docker build --network=none -f outputs/packages/release-01/context/docker/Dockerfile.competition --build-arg BASE_IMAGE=OFFICIAL_IMAGE_WITH_DIGEST -t datamate:release-01 outputs/packages/release-01/context
docker run --rm --gpus all --network none --mount type=bind,src=/work/data/split,dst=/data,readonly --mount type=bind,src=/work/results/release-01,dst=/result --mount type=bind,src=/work/scratch,dst=/work datamate:release-01
```

Tạo trước các thư mục mount. Chạy workload cần đánh giá, kiểm submission/type/log và thời gian từ trước load/index/compile đến flush output. Giữ cơ chế timeout trong predict.sh; chạy nhanh trên mẫu nhỏ không chứng minh đủ toàn private test trong 6 giờ.

## Thư viện và weights có sẵn giúp tiết kiệm phần nào?

Trong `Dockerfile.competition`, `RUN python /app/docker/install_runtime.py` **cài thư viện lúc build image** từ wheelhouse; `COPY models /app/models` đưa weights/tokenizer/config đã seal vào image. `predict.sh` không pip install hoặc tải Hugging Face. Khi BTC khởi động container, hai chi phí cài/tải đó đã được loại khỏi inference.

Mỗi process mới vẫn phải import Python/native extension, khởi tạo CUDA, đọc/xác minh weights từ đĩa, nạp weights vào GPU, tạo engine/KV cache; tùy cấu hình có thêm compile/capture. Image lưu dữ liệu trên đĩa, không lưu model đã nằm trong VRAM. Vì vậy Docker **không xóa thời gian startup của vLLM**. Startup có thể thấp hơn Colab/Drive nhờ I/O/cache khác, nhưng phải đo image thật; không coi toàn bộ số load khoảng 653s Hải ghi nhận là thời gian pip/download sẽ biến mất.

`runtime.json` ghi các phase verify/import vLLM/tokenizer/engine; `run_summary.json` và `supervisor.json` ghi thời gian batch/process. Phase tokenizer bao gồm import Transformers/config; phase engine gồm nhiều việc SDK tự làm, chưa tách riêng compile và nạp weights. `/work/vllm` chứa cache vLLM. Cache phù hợp có thể giảm khởi tạo ở lượt sau, nhưng không mặc định có cache trên máy BTC; mount `/work` mới cũng che cache đã đặt cùng đường dẫn trong image. Đánh giá ngân sách 6 giờ bằng lần chạy offline từ trạng thái cache ban đầu đã nêu rõ, tính cả startup/RAG index và dọn output.

Khi thay code/dependency/checkpoint sau rehearsal, phải kiểm lại image mới. Không sửa dữ liệu /data hoặc dùng test để training.

## 4. Xuất image ngày cuối

```bash
docker save -o outputs/packages/datamate-solution.tar datamate:release-01
sha256sum outputs/packages/datamate-solution.tar > outputs/packages/datamate-solution.tar.sha256
```

Giữ kèm README chạy lại, commit/config, solution manifest, license, dependency GPU, kết quả rehearsal và checksum. Cách upload/tên file/giới hạn gói tuân theo guideline được xác minh, không suy từ template này. Nếu cần nén TAR, tính lại checksum trên đúng file gửi.

## Docker dev / CPU smoke

Giữ Dockerfile.dev để kiểm packaging/input/output mà không tải model:

```powershell
docker build -f docker/Dockerfile.dev -t datamate-dev:local .
.\tools\docker-smoke.ps1
```

DemoBackend luôn abstain; smoke này không phải PA hoặc kiểm thử GPU. Kết quả mới đặt dưới outputs/runs/docker-smoke/.
