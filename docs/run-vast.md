# Thuê GPU để kiểm Docker DataMate

Máy BTC theo PDF nhận14/11/2026: **2×RTX5090,32GB/GPU; RAM512GB; CUDA>=12.8; tối đa6giờ kể cả load**. Ưu tiên thuê đúng2×5090. A10080GB có thể kiểm chức năng đầy đủ nhưng không thay bài kiểm Blackwell/32GB mỗi card.

## Chọn máy

1. Tạo tài khoản/nạp số dư trên [Vast.ai](https://cloud.vast.ai/). Giá thay đổi theo host; kiểm tổng GPU/giờ, storage, upload/download và thời hạn máy trước khi Rent. Tôi không tự thuê máy hoặc thanh toán.
2. Tạo SSH key nếu chưa có: PowerShell `ssh-keygen -t ed25519`. Thêm **public key** `.pub` vào Account/SSH Keys **trước khi thuê**; giữ private key trên máy mình.
3. Chọn **Ubuntu22.04 VM**, launch bằng SSH. Theo [hướng dẫn VM chính thức](https://docs.vast.ai/guides/instances/virtual-machines), template này có CUDA/Docker; lọc `vms_enabled=true`. VM hỗ trợ Docker bên trong; instance dạng Docker thông thường không tương đương VM.
4. Chọn On-demand, host verified/reliability tốt, **2×RTX5090**, khuyến nghị RAM>=128GB và disk **>=300GB** để chứa source/models, image layers/build cache, TAR và TAR.GZ cùng lúc. Đây là mức thuê gợi ý; máy BTC là512GB RAM. Kiểm tốc độ download/SSD và thời gian còn lại; không chọn interruptible cho lượt release dài.
5. Copy lệnh SSH Vast cung cấp. Có thể gửi tôi **host,port,user/lệnh SSH** để hỗ trợ tiếp; không gửi private key hoặc API token vào chat. VM ít lựa chọn hơn; nếu không tìm được2×5090, cần kiểm thêm trên máy đúng cấu hình trước khi gọi là đã rehearsal giống BTC.

## Vào máy và kiểm nền tảng

```bash
nvidia-smi
docker version
df -h
free -h
```

Phải thấy2GPU, đủ disk và Docker Linux daemon. `nvidia-smi` hiện CUDA tối đa driver hỗ trợ, không chứng minh image có `nvcc`.

**Điểm chưa chốt:** metadata image `vllm/vllm-openai:v0.33.0` đã kiểm cho thấy CUDA13.0.2; guideline chỉ nêu R570+ và image mẫu CUDA12.8. CUDA13 thường cần R580+ ([NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)); compatibility mode chỉ hỗ trợ một số GPU chuyên dụng/datacenter, không mặc định dùng cho RTX5090 ([vLLM](https://docs.vllm.ai/en/v0.33.0/deployment/docker/#running-on-systems-with-older-cuda-drivers)). Thuê host R580 giúp kiểm CUDA13 nhưng **không chứng minh máy BTC R570 chạy được**. Chỉ chốt khi BTC xác nhận driver đủ mới hoặc đã build/kiểm stack Qwen3.5/vLLM phù hợp CUDA12.8. Không hạ đơn thuần về vLLM0.10.1.1: đó là ví dụ recommended, không có bằng chứng hỗ trợ Qwen3.5 hiện tại.

## Chuyển dữ liệu từ Windows

Đọc [Docker](docker.md) để tạo gói release và danh sách artifact. Dùng `scp -P PORT -r ... USER@HOST:/workspace/...` hoặc upload riêng lên kho artifact rồi tải trước khi chạy offline. Thay PORT/USER/HOST theo lệnh SSH thực tế, không dùng tên ví dụ nguyên xi.

Phải chuyển đủ:

- Source đã chốt và config submission.
- Full model **qwen35-9b-stage2-comparison-merged**, manifest SHA256 `732f292bfb088f8dc90b0fe91e255a60886772292614cf0b1af58219d097b054`; giữ training_report/tokenizer/auxiliary MTP. Không thay bằng Qwen base.
- Hai model RAG đã pin/seal hoặc tải bằng `datamate stage-model` **trước inference**, rồi kiểm hash.
- Training evidence gồm frozen source/dataset của cả hai stage, adapter parent/final, config/log/report, data manifest, bộ thư viện training offline.

Không copy `.venv` Windows sang Linux. Chuẩn bị wheelhouse trong đúng base/Python Linux, build bằng mạng tắt, chạy đúng scoring command trong Docker guide, hai lượt output mới. Sau khi đạt mới export `.tar.gz`, MD5, tải về kiểm lại và upload Drive.

Khi xong, tải đủ image/receipts/logs về nơi lưu bền vững, kiểm checksum rồi **Destroy** instance. Stop có thể vẫn tính phí disk; kiểm trang Billing. Không Destroy trước khi có bản sao artifact đã xác minh.

## Nếu chỉ thuê khoảng1giờ

Chuẩn bị source/ba model/evidence và upload kho artifact **trước** Rent. Sau khi thuê, kiểm GPU/Docker/driver ngay; nếu sai cấu hình thì dừng xử lý thay vì chờ download. Ưu tiên10–20câu TRAIN đại diện ở hai lượt cold offline, kiểm đầy đủ output/log rồi export và tải kết quả. Đừng chạy100câu×2: chỉ inference có thể đã dùng gần hết giờ, chưa kể download/build/startup.

Một giờ **không được bảo đảm đủ**: tải hàng chụcGB, resolver/build và export image có thể chiếm phần lớn thời gian. Không bỏ kiểm offline/full-model chỉ để kịp giờ; nếu chưa qua, giữ artifact/log và trạng thái chưa sẵn sàng. Test Solution trên Portal làm sau khi đã có archive/link/MD5; không cần giữ VM chỉ để chờ BTC xử lý. Nếu BTC chưa xác nhận driver, Test Solution thành công trên máy họ là bằng chứng quan trọng trước khi nộp cuối.

## Laptop RTX4060 8GB

Đã kiểm Docker Desktop Linux thấy GPU4060 qua `--gpus all`. Model nhỏ có thể kiểm import/CUDA/đọc–ghi/offline. Qwen9B FP16 riêng weights đã khoảng18GiB, thêm RAG/KV/workspace;8GB không đủ cho cấu hình nộp. Tiny test qua không chứng minh full model hoặc kernel SM120 của5090 qua.
