# Docker nộp BTC — DataMate

Đối chiếu **toàn bộ PDF chính thức** `zac2026_DataMate_docker_guideline_final.pdf` do Bảo cung cấp14/11/2026. Bản trích10trang và SHA256 lưu trong `outputs/runs/Bảo/docker-release-20261114/`. Đây là nguồn cập nhật thay các giả định của template cũ.

## Cấu hình và trạng thái

| Mục | Đã chốt / giới hạn |
|---|---|
| Máy BTC | 2×RTX5090, **32GB mỗi GPU**, RAM512GB; CUDA>=12.8, build hỗ trợ SM120 |
| Entrypoint | BTC gọi **`bash /code/predict.sh` không đối số**, override entrypoint |
| Input/output | `/data` chỉ đọc; `/result/submission.jsonl`; một record một dòng mỗi câu trong `/result/logs/` |
| Thời gian | <=6giờ toàn private test, gồm import/hash/load/index/JIT/inference |
| Image | **linux/amd64**, một image/tag; docker-save TAR chưa nén <=100GB, tính mọi layer |
| Nộp | **`.tar.gz`**, Google Drive URL, **MD5 của chính file nén**, kích thước TAR chưa nén |
| Mạng | Tắt hoàn toàn lúc inference; weights/tokenizer/config/thư viện/kernel resources phải có sẵn |
| Training | Image chứa source, dữ liệu train/validation, generation code, config/log/report, checkpoint xác minh và manifest |

`configs/submission.json` bật **vLLM + dense_rerank RAG**, FP16. Generator TP1 trên GPU0; hai encoder trên GPU1; một question worker,16K context, chunked prefill2048, memory fraction0,85. Đây là **cấu hình ứng viên phải rehearsal**, chưa được đo trên2×5090. Hai GPU không tạo thành một vùng VRAM64GB tự động. Giữ Torch và lựa chọn `--rag-mode none` cho đối chứng.

Model merged của Bảo: `models/qwen35-9b-stage2-comparison-merged`, manifest SHA256 **`732f292bfb088f8dc90b0fe91e255a60886772292614cf0b1af58219d097b054`**. Đã kiểm16file/19.326.417.873bytes. Bốn shard tải xuống bị thêm hậu tố đã được đối chiếu SHA256 rồi đổi lại tên; không sửa weights/manifest. Hai RAG model đã tải đúng revision và kiểm weight hash. Tổng **10.788.614.897** tham số, dưới15B; xem registry/model-evidence về license/ngày phát hành. Giữ điều kiện trước01/10 mà chủ dự án đã yêu cầu; PDF ghi26/10, không nới điều kiện của đội.

**Chưa có image thi được xác nhận GPU.** Hiện đã chuẩn bị source/assets và kiểm CPU; chủ dự án sẽ thuê GPU sau. Không nộp image `datamate-dev:*` hoặc gọi CPU/tiny test là chứng minh Qwen9B+RAG chạy đúng trên5090.

## Base image và CUDA: phải giải quyết trước khi nộp

PDF **recommended** `vllm/vllm-openai:v0.10.1.1` (Torch2.7.1cu128), cho phép alternative hỗ trợ5090/SM120. Đây không phải bằng chứng vLLM0.10 hỗ trợ Qwen3.5. Adapter hiện nhắm AsyncLLM vLLM0.33; không hạ version chỉ để giống hình mẫu.

Đã inspect tag0.33 chính thức: **CUDA13.0.2**, amd64 digest:

```text
vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3
```

**Đây là base ứng viên, chưa được duyệt cho máy BTC.** Guideline nói R570-or-newer; CUDA13 thường cần R580+ ([NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)). vLLM compatibility mode chỉ hỗ trợ một số GPU chuyên dụng/datacenter, không mặc định áp dụng cho RTX5090 ([vLLM](https://docs.vllm.ai/en/v0.33.0/deployment/docker/#running-on-systems-with-older-cuda-drivers)). Các tag0.33-cu128/cu129 đã kiểm không tồn tại. Cần BTC xác nhận driver đủ mới hoặc một stack CUDA12.8/Qwen3.5 được build và kiểm thật. Chạy trên5090/R580 vẫn không chứng minh chạy trên R570.

Không dùng workaround `nvidia-nccl-cu12==2.27.7` của ví dụ cho stack cu13. Cần kiểm NCCL thực tế nếu đổi TP2; cấu hình TP1/RAG GPU1 hiện tránh chia generator qua hai card. Docker đặt `NCCL_P2P_DISABLE=1`, spawn cho worker vì RAG đã khởi tạo CUDA trước generator. Giữ shm16g; không dùng host IPC.

## Chuẩn bị trước khi thuê GPU

1. Hoàn tất chuyển source/weights/evidence lên kho artifact trước; Git không chứa dữ liệu/weights. [Hướng dẫn Vast](run-vast.md) có lựa chọn VM/SSH/disk.
2. Giữ full merged model cùng artifact_manifest, training_report, tokenizer và MTP auxiliary shard. Hai encoder ở `models/vietnamese-embedding`, `models/vietnamese-reranker`; không dùng `main` thay revision đã pin.
3. Training evidence đã gom từ **đúng ZIP source/dataset và run hoàn tất của cả hai stage** ở `outputs/packages/release-20261114/training-evidence-source/`. Manifest kiểm hash, kích thước, số mẫu và partition. Folder này chứa source/data/report/checkpoints; **còn phải bổ sung wheelhouse Linux** ở bước dưới. `training-input/` là đầu vào trung gian, không cần upload khi đã có evidence sealed.

Stage1 dùng2347train/618validation sequences, stage2 dùng2647/618; giữ24business train/6validation và630validation question IDs. Dữ liệu gốc chỉ TRAIN;80ID của phép so100 từng được train, nên kết quả đó không phải unseen/private score.

Training-only public base được khai báo revision/license/URL/SHA256 trong data_manifest, có thể omit weights theo PDF; mọi model **inference** vẫn bên trong image. Bản huấn luyện nguyên gốc dùng base manifest `f8b32a...`; không thay bằng archive khác hash để tiếp tục parent cũ. `train.sh` tái lập cả hai stage vào output mới, cập nhật hash parent mới và giữ cấu hình/report frozen. Bộ thư viện training độc lập với inference, cài lại offline vào venv khi BTC cần tái lập; không train trong `predict.sh`.

## Chuẩn bị dependency và build trên Linux

Sau khi giải quyết base/driver, đặt `DATAMATE_BASE_IMAGE` thành **digest đã chốt**, không dùng `latest`. Dưới đây chạy tại gốc source trên VM; `EVIDENCE` là đường dẫn folder evidence sau khi chuyển lên. Base image chứa Python>=3.11.

```bash
export DATAMATE_BASE_IMAGE='REPLACE_WITH_VERIFIED_BASE_AT_SHA256_DIGEST'
docker pull "$DATAMATE_BASE_IMAGE"
docker run --rm --entrypoint bash \
  --mount "type=bind,source=$PWD,target=/workspace" -w /workspace \
  "$DATAMATE_BASE_IMAGE" -c \
  'python docker/prepare_release.py --base-image "$1" --evidence outputs/packages/release-20261114/training-evidence-source --output outputs/packages/release-linux-01' \
  prepare "$DATAMATE_BASE_IMAGE"
```

Đây là **bước online trước build**, không phải inference. Helper xác minh model, tải wheel Linux đúng Python, ghi lock hash/freeze/pip-check, tạo bundle và training wheelhouse riêng rồi stage `context/`. Nó không tải thay Qwen merged bằng base. `configs/docker-runtime-constraints.txt` khóa các version inference đã quan sát; resolved wheel manifest khóa mọi dependency của release cụ thể. Lỗi resolver phải được xử lý và ghi nhận, không bỏ `pip check`.

```bash
docker build --platform linux/amd64 --network none --provenance=false \
  --build-arg BASE_IMAGE="$DATAMATE_BASE_IMAGE" \
  -f outputs/packages/release-linux-01/context/docker/Dockerfile.competition \
  -t zaic2026_datamate_loveyou3500:final \
  outputs/packages/release-linux-01/context
```

BuildKit cần có frontend/base đã pull/cache trước khi build offline. `--network none` tắt mạng trong các RUN; bước resolve/pull base/frontend là preparation, không phải bằng chứng build hoàn toàn air-gapped. Installer cài từ wheelhouse có hash; dùng bind mount để không lưu thêm một layer wheel inference. Training wheelhouse được giữ trong `/code/training_code/` theo yêu cầu offline tái lập. Image dùng `ENTRYPOINT []`, `CMD ["bash","/code/predict.sh"]`.

`package_solution.py` mặc định yêu cầu evidence sealed, wheelhouse Linuxamd64 và base digest. `--development` chỉ cho fixture chưa đủ tài nguyên; manifest ghi rõ chưa release. Context có dockerignore riêng để **không loại nhầm training data**. Không build competition trực tiếp từ toàn repo.

## Kiểm đúng lệnh BTC

```bash
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g \
  --entrypoint bash \
  -v /ABSOLUTE/SPLIT:/data:ro -v /ABSOLUTE/NEW_RESULT:/result \
  zaic2026_datamate_loveyou3500:final /code/predict.sh
```

Không mount model/cache/credential/socket từ host; không privileged/host IPC/host service. Runtime có dữ liệu mới ở `/data`, tự tạo scratch dưới `/work`. Code luôn đọc toàn `questions.jsonl` BTC, không nhúng benchmark ID vào image.

Có thể dùng `bash docker/rehearse.sh IMAGE ABSOLUTE_SPLIT NEW_OUTPUT_ROOT`: kiểm nvidia-smi/pip/Ninja/nvcc/compiler rồi chạy **hai container cold riêng** với lệnh trên. Chọn subset TRAIN đại diện để kiểm lỗi trước, tạo subset ngoài dữ liệu gốc; không đưa gold vào inference/RAG. Mẫu10–20câu không chứng nhận6giờ hay chất lượng toàn private test.

```bash
python tools/verify_submission.py --data /ABSOLUTE/SPLIT --result /ABSOLUTE/NEW_OUTPUT_ROOT/run-1
python tools/verify_submission.py --data /ABSOLUTE/SPLIT --result /ABSOLUTE/NEW_OUTPUT_ROOT/run-2
```

Kiểm đủ ID/type/giá trị JSON, log một dòng, answer/elapsed_ms/thinking, run_summary sạch lỗi. So các đáp án theo ID sau khi bỏ `reason` khỏi so khớp giá trị; cần output giống hoặc chênh PA<=0,03 trên TRAIN có gold theo PDF. Chấm nhãn **sau** inference. Lưu wall/startup/peak VRAM từng GPU, OOM/timeout, cấu hình/cache/hardware/driver/freeze và image ID. Ước lượng tham chiếu PDF: startup + trung bình mỗi câu×500; vẫn cần workload đủ đại diện, không lấy vài câu dễ để bảo đảm deadline.

Log BTC giữ raw thinking đầy đủ, có SQL đã chạy/parameters/output khi có; trace chi tiết vẫn còn. Human-readable markdown ở `/result/debug/`; lỗi cấp run ở `/result/runtime-error.json`/`shutdown-error.json`, không lẫn record từng câu. Không sửa metrics cũ hoặc tạo metrics GPU giả để ép pass.

## Export, upload và Test Solution

Chỉ export bản nộp sau các kiểm trên và chốt tương thích driver. Ví dụ:

```bash
python tools/export_solution.py --image zaic2026_datamate_loveyou3500:final \
  --output outputs/packages/zaic2026_datamate_loveyou3500.tar.gz
```

Helper docker-save vào file tạm, gzip, đọc toàn stream/CRC, kiểm một image/tag Linuxamd64, đủ layer và TAR chưa nén<=100.000.000.000bytes; tạo `.md5`, `.sha256`, `.submission.json`. Không dùng gzip ISIZE32-bit để đo image lớn. TAR/file nén tạm đều cần disk; không để ổ đầy giữa export. Archive/hash checks **không** tự chứng nhận GPU hoặc quyền truy cập Drive.

Upload đúng `.tar.gz`, giữ tên sau khi hash, cấp quyền tải phù hợp BTC. Tải từ link trên máy khác, kiểm MD5 lại; điền URL và MD5 của **file nén** vào Portal → DataMate → Test Solution. Mỗi đội chỉ1test/ngày theo thông báo/screenshot người dùng; không tự Submit thay chủ dự án. Nếu lỗi, lưu View full log rồi cập nhật [runtime issues](runtime-issues-2026-11-14.md). Chốt image/test sớm trước deadline, không đợi giờ cuối.

## CPU smoke trên laptop

```powershell
docker build -f docker/Dockerfile.dev -t datamate-dev:release-check .
.\tools\docker-smoke.ps1 -Image datamate-dev:release-check -DataRoot 'E:\Bảo Lập Trình AI\ZALO_AI_2026\train\train'
```

Smoke dùng3câu thật nhưng DemoBackend chỉ abstain; kiểm offline, `/data` RO, rootfs RO, entrypoint đúng và output/log. Không phải image nộp, điểm PA hoặc bằng chứng9B/SM120. RTX4060 8GB không đủ cho full9B FP16 +RAG theo profile này.
