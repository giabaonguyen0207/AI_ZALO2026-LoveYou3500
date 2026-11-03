# DataMate · AI Agent cho Zalo AI Challenge 2026

Ứng dụng Python theo kiến trúc **modular monolith**: đọc dữ liệu doanh nghiệp, cho LLM kiểm tra schema và gọi SQL, lấy đáp án trực tiếp từ kết quả query, rồi xuất JSONL đúng type. CLI, web local và Docker dùng chung mã nguồn. Có notebook Colab để chọn Qwen 3.5 9B hoặc Gemma 4 12B và fine-tune LoRA/QLoRA.

**Mới bắt đầu? Đọc [Hướng dẫn chạy từng bước](docs/huong-dan-chay-tung-buoc.md).** Làm phần A để mở web ngay; phần B cài Qwen trên laptop; phần D hướng dẫn Colab từ lúc chuẩn bị ZIP đến lấy model đã train về. Mỗi bước có lệnh, kết quả mong đợi và cách xử lý lỗi. Các mục bên dưới là tài liệu tham chiếu ngắn.

**Cập nhật inference 03/11/2026:** pipeline v2 đã chạy lại cùng 72 câu development: **56 đúng, 1 sai, 15 bỏ qua; PA 76,39%**, khoảng 18 phút 15 giây riêng batch. Đây là mẫu đã dùng chẩn đoán, chưa đại diện hiệu năng private test. Đọc [phân tích lần 2, rules viết sẵn và lý do thiếu reasoning trong log](docs/inference-review-72-v2-2026-11-03.md), hoặc [những gì đã sửa và cách chạy Colab](docs/pipeline-v2-2026-11-03.md).

**Đã chuẩn bị lượt 3:** 100 câu ngẫu nhiên đều từ toàn bộ 3.000 câu train, seed `20261103`, không cân đối theo type/ngôn ngữ/doanh nghiệp và không lọc câu cũ/holdout. Notebook mặc định chạy `random100-round3-01`, có log từng bước. Xem [danh sách và cách chạy](docs/inference-round3-random100.md). Chưa có kết quả LLM của lượt này.

**Kiểm tra triển khai ban đầu 01/11/2026:** đã kiểm thử phần mềm, đọc được 30 doanh nghiệp train và kiểm tra tokenizer/template. Cập nhật sau khi người dùng gặp lỗi load Qwen 9B trên RTX 4060 8GB: đã thêm **Qwen 3.5 4B · Local 4bit** để thử agent trên laptop. Xem [hướng dẫn chuyển sang 4B](docs/huong-dan-chay-tung-buoc.md). Chưa có benchmark so sánh model hoặc fine-tune. Demo luôn trả `abstain`.

## Bắt đầu trên Windows, ổ E:

Source, môi trường Python, cache, model, notebook và kết quả của dự án đặt trong thư mục này. Script cài đặt đặt pip cache/TEMP ở E:. Docker Desktop quản lý image/volume theo cấu hình engine riêng; các artifact xuất từ Docker được chép về thư mục dự án trên E:.

```powershell
Set-Location -LiteralPath 'E:\Bảo Lập Trình AI\ZALO_AI_2026'
.\tools\setup.ps1
.\tools\start-web.ps1
```

Mở **http://127.0.0.1:7860**. Chọn split, doanh nghiệp và `as_of_date`, rồi nhập câu hỏi. Giao diện có chọn model/chế độ, schema explorer, lịch sử câu hỏi lưu trong trình duyệt, dấu vết SQL và tải JSON. Ở màn hình nhỏ, bấm **Dấu vết** để mở bảng kết quả. Demo kiểm tra luồng phần mềm, không đưa ra đáp án kinh doanh giả.

Máy local hiện có RTX 4060 Laptop 8GB. Dùng bản 4bit có thể là hướng thử, nhưng chưa đo khả năng fit và tốc độ. CPU/BF16 cần đủ RAM và có thể rất chậm. Thiếu checkpoint hoặc lỗi CUDA được báo trong giao diện.

Để thử luồng agent trên laptop này, dùng preset **Qwen 3.5 4B · Local 4bit**. Người dùng đã báo Qwen 9B không tải được với cấu hình phân bổ hiện tại; ưu tiên Colab/GPU lớn hơn khi đánh giá 9B, Gemma 12B hoặc fine-tune.

Đã tải và chạy thử 4B thật: một câu COUNT(*) khớp truy vấn kiểm tra, mất khoảng 186 giây cho 2 lượt model. Local vẫn chậm và bộ nhớ đỉnh cao; xem [kết quả và giới hạn](docs/local-4b-check-2026-11-01.md). Để thử một câu trên Colab trước khi train, dùng [notebook inference](notebooks/DataMate_Colab_Inference.ipynb).

## Chuẩn bị model rồi inference thật

Cài PyTorch CUDA phù hợp driver/GPU trước khi chạy LLM. Lệnh sau cài các dependency còn thiếu vào `.venv` của dự án; `.[llm]` cũng có thể tải PyTorch nếu môi trường chưa có.

```powershell
.\tools\setup.ps1 -LLM
```

**Chỉ bước chuẩn bị sau được tải model online.** Chọn một model chính:

```powershell
# Bản nhỏ để thử trên laptop:
.\.venv\Scripts\python.exe -m datamate stage-model --model qwen35-4b --output models/qwen35-4b

# Các lựa chọn lớn hơn, cần kiểm tra tài nguyên:
.\.venv\Scripts\python.exe -m datamate stage-model --model qwen35-9b --output models/qwen35-9b
# Hoặc:
.\.venv\Scripts\python.exe -m datamate stage-model --model gemma4-12b --output models/gemma4-12b
```

Nếu muốn thử speculative decoding của Gemma:

```powershell
.\.venv\Scripts\python.exe -m datamate stage-model --model gemma4-12b-assistant --output models/gemma4-12b-assistant
```

Lệnh tải đúng full revision đã pin, đếm tensor và seal SHA256 từng file. Inference kiểm tra lại artifact, chỉ dùng file local và tắt chế độ tải tài nguyên. Checkpoint chưa seal không được chạy. Trọng số cần hàng chục GB dung lượng; tokenizer trong `.cache/tokenizers` không phải trọng số và không làm model sẵn sàng inference.

```powershell
# Một câu bằng preset local 4bit:
.\.venv\Scripts\python.exe -m datamate ask --config configs/qwen35-4b-local-4bit.json --business biz_101 --as-of 2026-06-15 --question 'Tổng doanh thu tháng này là bao nhiêu?'

# Batch trên train, output mới để không ghi đè lượt chạy cũ:
.\.venv\Scripts\python.exe -m datamate run --config configs/qwen35-9b.json --result-dir result/baseline-01

# Public chỉ dùng inference; không đưa vào pipeline training:
.\.venv\Scripts\python.exe -m datamate run --config configs/qwen35-9b.json --split-root public_test/public_test --result-dir result/public-01
```

`--resume` chỉ tiếp tục khi cấu hình, tập câu hỏi, hash nguồn và model manifest vẫn khớp. `submission.jsonl` được flush từng câu; `logs/` chứa ID, đáp án cuối, thời gian và trace. Chưa có trọng số thì chương trình báo cần `stage-model`; không tự chuyển sang trả lời giả.

**Debug từng bước:** mỗi JSON log có thêm bản `.md` dễ đọc: đầu ra nguyên văn của LLM, SQL và parameters, kết quả/lỗi công cụ, đáp án cuối. SQL do LLM sinh và SQL do metric tạo được ghi riêng. Đã xuất sẵn 72 log của lượt 2; xem [cách mở log và xuất lại lượt cũ](docs/debug-step-logs.md).

## Chọn model và assistant

| Preset | Model/chế độ | Tổng tham số |
|---|---|---:|
| `configs/qwen35-4b-local-4bit.json` | Qwen 3.5 4B, thử local 4bit/context 16K | 4,660B |
| `configs/qwen35-9b.json` | Qwen 3.5 9B, single | 9,653B |
| `configs/qwen35-9b-local-4bit.json` | Qwen, thử local 4bit/context 8K | 9,653B |
| `configs/gemma4-12b.json` | Gemma 4 12B IT, single | 11,960B |
| `configs/gemma4-12b-speculative.json` | Gemma + assistant MTP ~0,423B | 12,383B |
| `configs/gemma4-12b-self-review.json` | Cùng Gemma gọi lại để kiểm tra | 11,960B |

MTP assistant đề xuất token để target xác minh, nhằm **tăng tốc sinh token**; không phải model kiểm tra nghiệp vụ. Chưa biết có nhanh hơn trên GPU của bạn không. Self-review là thí nghiệm khác và có thể thay đổi độ chính xác. Qwen 9B + Gemma 12B chung một hệ thống inference vượt 15B. Xem [phân tích và revision đã xác minh](docs/model-comparison.md).

## Colab và fine-tuning

Mở [notebooks/DataMate_Colab_LoRA.ipynb](notebooks/DataMate_Colab_LoRA.ipynb). Notebook cho chọn model, LoRA/QLoRA, thu trajectory baseline từ train, resume checkpoint, merge và so base/fine-tuned trên holdout. Sửa đường dẫn Drive ở ô đầu. Chưa được chạy GPU Colab trong phiên triển khai này.

Tạo gói source để đưa lên Colab:

```powershell
.\.venv\Scripts\python.exe tools/package_colab.py
```

Gói `outputs/datamate-colab-source.zip` không chứa dataset, trọng số, `.venv`, cache hay thông tin bí mật. Đưa gói source và gói **train riêng** lên Drive theo notebook.

```powershell
.\.venv\Scripts\python.exe -m datamate prepare-training --split-root train/train --output outputs/training/format --kind format
```

Đã tạo bản thực tế với seed 42: **24 doanh nghiệp / 2.370 câu train**, **6 doanh nghiệp / 630 câu validation**. `format` chỉ học chép/chuẩn hóa kết quả đã cung cấp, không dạy giải câu hỏi hay SQL.

Để train agent, chạy baseline trên `train_ids.json`, lấy log đúng theo nhãn train rồi chuẩn bị `--kind trajectories`. Giữ riêng validation businesses. Quy trình ở [docs/training.md](docs/training.md). Allowlist `configs/training_sources.json` khóa hash gói train đã nhận; public/private hoặc train bị sửa không được dùng âm thầm.

Prompt ban đầu đo bằng tokenizer thật, mỗi doanh nghiệp một câu trên 30 doanh nghiệp: Qwen khoảng **5.867–10.989 token**, Gemma **6.128–11.527 token**, chưa gồm tool results. Vì vậy LoRA mặc định dùng giới hạn 16K, không tự cắt rulebook. Preset local 8K có thể abstain ở schema lớn; muốn tăng context phải kiểm tra VRAM.

## Kiểm thử và đánh giá

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp .cache/pytest -q
.\.venv\Scripts\python.exe -m datamate inspect --split-root train/train --business biz_102 --output outputs/catalog.json
.\.venv\Scripts\python.exe -m datamate models
```

Local evaluator phân biệt type, thứ tự list, dấu tiếng Việt, alias, bảng và PA (+1/−1/0). Không thay thế bộ chấm chính thức. So sánh trên cùng holdout bằng CLI `compare`, xem [docs/model-comparison.md](docs/model-comparison.md).

Kết quả kiểm tra thực tế và giới hạn: [docs/verification-2026-11-01.md](docs/verification-2026-11-01.md).

## Docker

Image phát triển đã build là `datamate-dev:local`; chỉ có runtime nhẹ và demo, chưa có PyTorch/trọng số. Không phải image nộp thi.

```powershell
docker build -f docker/Dockerfile.dev -t datamate-dev:local .
.\tools\docker-smoke.ps1
```

Nếu Docker nhận bind mount đường dẫn Unicode, có thể chạy trực tiếp:

```powershell
$splitPath = (Resolve-Path -LiteralPath train/train).Path
$outputPath = Join-Path (Get-Location) 'result/docker-run'
New-Item -ItemType Directory -Force $outputPath | Out-Null
docker run --rm --network none --read-only --tmpfs /work:rw,size=512m --tmpfs /tmp:rw,size=128m --mount "type=bind,source=$splitPath,target=/data,readonly" --mount "type=bind,source=$outputPath,target=/result" datamate-dev:local --limit 3
```

Docker Desktop trong phiên này không nhận bind mount đường dẫn dự án. Helper chép **đúng subset dự án** qua volume: [tools/docker-smoke.ps1](tools/docker-smoke.ps1). Smoke test tương đương đã chạy 3 câu với `/data` chỉ đọc, rootfs chỉ đọc và `--network none`; kết quả demo là 3 abstain.

Docker guideline chính thức chưa có. `docker/Dockerfile.competition` yêu cầu `BASE_IMAGE` được cung cấp rõ ràng; chưa xác nhận CUDA/PyTorch/RTX 5090 của image này. Trước khi build, chuẩn bị wheelhouse Linux phù hợp và model đã seal, sửa `configs/submission.json` theo model đã chọn. `predict.sh` ghi `/result/submission.jsonl`, `/result/logs/` và có watchdog dưới 6 giờ. Xem [docs/docker.md](docs/docker.md).

## Tài liệu và nguồn

- [Hướng dẫn chạy từng bước cho người mới](docs/huong-dan-chay-tung-buoc.md).
- [Kiến trúc và luồng xử lý](docs/architecture.md).
- [So sánh model, assistant, eligibility](docs/model-comparison.md).
- [Training và chống rò rỉ dữ liệu](docs/training.md).
- [Ngữ cảnh cuộc thi do người dùng cung cấp](docs/competition-context.md), chưa thay thế tài liệu chính thức.
- [Kiểm kê dữ liệu ngày 31/10](docs/colab-assessment-2026-10-31.md) và [nghiên cứu baseline trước triển khai](docs/baselines-and-costs.md) là tài liệu lịch sử; đề xuất cũ không phải kết quả benchmark.

Không hard-code đáp án theo question/business ID, không dùng test để training. Quy tắc kinh doanh lấy từ split hiện tại. Đây là baseline phần mềm cần được benchmark và hoàn thiện trên GPU/model đã chọn trước khi nộp thi.
