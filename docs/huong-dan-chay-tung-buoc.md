# Hướng dẫn chạy DataMate từng bước

Tài liệu này dành cho bạn biết Python nhưng mới làm quen với AI Agent. Bạn không cần đọc hết rồi mới bắt đầu: **hãy làm phần A trước để mở được giao diện**.

Mọi lệnh PowerShell bên dưới chạy trên máy Windows của bạn, tại:

```text
E:\Bảo Lập Trình AI\ZALO_AI_2026
```

Khi thấy một khối lệnh, bạn sao chép nội dung trong khối, dán vào PowerShell rồi nhấn Enter. Không gõ thêm chữ `PS>`.

## Chọn việc bạn muốn làm

| Bạn muốn làm gì? | Đọc phần nào? | Cần tải model chưa? |
|---|---|---|
| Mở web, xem dữ liệu, kiểm tra nút bấm | **A** | Chưa cần |
| Cho Qwen trả lời thật trên laptop | **B**, sau khi mở web thành công | Có |
| Chạy một câu hoặc tạo file kết quả bằng lệnh | **C**, sau khi model chạy được | Có nếu trả lời thật |
| Fine-tune trên Google Colab | **D** | Notebook tự tải model trên Colab |
| Thử Gemma với assistant | **E**, trên máy/GPU đủ bộ nhớ | Có cả target và assistant |
| Đang gặp lỗi | **F** | Tùy lỗi |

**Cập nhật ngày 01/11/2026:** người dùng đã cài phần B và tải Qwen 9B, nhưng báo lỗi phân bổ model sang CPU/disk trên RTX 4060 8GB. Phần B/C dưới đây chuyển ví dụ local sang **Qwen 3.5 4B 4bit**. Giữ các model 9B/Gemma cho Colab/GPU đủ bộ nhớ; không cần cài lại thư viện nếu môi trường LLM đã sẵn sàng.

## A. Mở giao diện web trước

### A1. Mở PowerShell

Nhấn nút Start của Windows, tìm **PowerShell**, rồi mở ứng dụng đó. Nếu dùng Windows Terminal, chọn tab PowerShell.

Bạn không cần mở PowerShell bằng quyền Administrator.

### A2. Chuyển vào đúng thư mục dự án

Chạy:

```powershell
Set-Location -LiteralPath 'E:\Bảo Lập Trình AI\ZALO_AI_2026'
```

Kiểm tra bạn đang ở đúng nơi:

```powershell
Get-Location
```

Kết quả phải có đường dẫn `E:\Bảo Lập Trình AI\ZALO_AI_2026`.

Nếu bạn mở một cửa sổ PowerShell mới ở các lần sau, hãy chạy lại lệnh `Set-Location` này trước các lệnh của dự án.

### A3. Khởi động web

Trên máy hiện tại, phần này đã được cài đặt. Bạn chạy thẳng:

```powershell
.\tools\start-web.ps1
```

Bạn sẽ thấy thông báo có dạng:

```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:7860
```

**Đến đây hãy giữ cửa sổ PowerShell này mở.** Chương trình đang phục vụ giao diện, nên nó tiếp tục chạy và không trả lại dòng nhập lệnh ngay. Đây là trạng thái bình thường.

Nếu báo không có `.venv` hoặc không có module `datamate`, xem phần **F1** để cài lại phần web. Nếu PowerShell chặn file `.ps1`, xem **F2**.

### A4. Mở trình duyệt

Mở Chrome, Edge hoặc trình duyệt bạn đang dùng. Dán địa chỉ sau vào thanh địa chỉ:

[http://127.0.0.1:7860](http://127.0.0.1:7860)

`127.0.0.1` nghĩa là máy của bạn. Giao diện này chỉ mở được khi chương trình ở bước A3 còn chạy.

### A5. Thử một câu ở chế độ demo

Ở thanh lựa chọn phía trên giao diện, đặt:

| Ô trên giao diện | Giá trị để thử |
|---|---|
| **MODEL & CHẾ ĐỘ** | `Demo · kiểm tra luồng` |
| **SPLIT** | `Train` |
| **DOANH NGHIỆP** | `biz_101` |
| **AS OF DATE** | Ngày 15/06/2026 |

`AS OF DATE` là ngày được coi là “hôm nay” trong câu hỏi. Ví dụ, với ngày 15/06/2026, “tháng này” được giải nghĩa theo tháng 6/2026 và rulebook, không theo ngày thực trên máy.

Nhập vào ô chat:

```text
Tổng doanh thu tháng này là bao nhiêu?
```

Bấm nút mũi tên **↑** hoặc nhấn Enter. Shift + Enter dùng để xuống dòng.

Ở chế độ demo, kết quả có dạng:

```json
{
  "type": "abstain",
  "reason": "Chế độ demo chưa tải LLM. Chọn checkpoint local để inference thật."
}
```

Giao diện còn có trường `id` được sinh cho mỗi lần chạy. `abstain` trong bước này là **kết quả đúng của chế độ demo**: phần mềm đang kiểm tra giao diện và luồng xử lý, chưa dùng AI để tính doanh thu.

Để có đáp án thật, tiếp tục phần B. Bạn không cần fine-tune trước khi thử một model gốc.

### A6. Xem schema và dấu vết

Bấm **Khám phá dữ liệu** ở menu bên trái, rồi bấm **Đọc schema ↻**. Bạn sẽ thấy tên bảng và tên cột đọc từ file của doanh nghiệp.

“Schema” là cấu trúc dữ liệu: có những bảng nào, mỗi bảng có những cột nào. Bước này hoạt động ngay cả khi chưa tải LLM.

Quay lại **Thử nghiệm agent** để chat. Sau một lượt chạy, bảng **Dấu vết thực thi** hiển thị các bước xử lý. Khi dùng LLM thật, bạn có thể mở từng bước để xem SQL, kết quả truy vấn hoặc lỗi. Trên màn hình nhỏ, bấm **Dấu vết ◇** ở phía trên để mở bảng này.

### A7. Dừng và mở lại

Để dừng web, quay lại cửa sổ PowerShell đang chạy chương trình và nhấn **Ctrl + C**.

Lần sau muốn dùng lại, bạn chỉ cần:

```powershell
Set-Location -LiteralPath 'E:\Bảo Lập Trình AI\ZALO_AI_2026'
.\tools\start-web.ps1
```

Bạn không cần cài thư viện lại sau mỗi lần mở web. Phần web này cũng chưa cần Docker.

## B. Chạy Qwen 3.5 4B thật trên laptop

Phần này dùng **Qwen 3.5 4B 4bit** để kiểm tra agent trên RTX 4060 8GB. Nếu đã cài PyTorch CUDA và thư viện ở B2–B3 thành công, bạn có thể đi thẳng B4. Bản nhỏ giúp giảm yêu cầu bộ nhớ; kết quả của nó không thay thế benchmark model 9B/Gemma.

“Inference” nghĩa là dùng model đã có để trả lời. “Fine-tune” nghĩa là huấn luyện thêm model. Bạn có thể inference bằng model gốc trước, rồi mới quyết định có cần fine-tune hay không.

### B1. Dừng web rồi chuẩn bị PowerShell

Nếu web đang chạy, nhấn Ctrl + C ở cửa sổ của nó. Việc này tránh giữ một model trong web trong lúc bạn thử thêm một tiến trình khác.

Chạy khối sau tại PowerShell. Các biến này đặt file tải/cài tạm của pip trong dự án trên E:.

```powershell
Set-Location -LiteralPath 'E:\Bảo Lập Trình AI\ZALO_AI_2026'
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache/pip'
$env:TEMP = Join-Path (Get-Location) '.cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
```

Trong các lệnh dưới đây, `.\.venv\Scripts\python.exe` là Python riêng của dự án. Bạn không cần chạy lệnh activate môi trường.

### B2. Cài PyTorch CUDA và thư viện LLM

PyTorch là thư viện chạy phép tính của model. Bản có CUDA cho phép dùng GPU NVIDIA.

Ví dụ cài bản CUDA 12.8 từ kho PyTorch chính thức:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Kho này có wheel Windows/Python 3.13; bước cài và khả năng nhận GPU vẫn cần được kiểm tra bằng B3. Nếu môi trường của bạn thay đổi, đối chiếu [bộ chọn cài đặt PyTorch](https://pytorch.org/get-started/locally/) và [kho CUDA 12.8 chính thức](https://download.pytorch.org/whl/cu128/torch/).

Đợi lệnh trên hoàn tất, sau đó chạy:

```powershell
.\tools\setup.ps1 -LLM
```

Lệnh thứ hai cài Transformers, Accelerate, bitsandbytes và các thư viện liên quan. Đây có thể là các lượt tải lớn. Khi cài xong, PowerShell trả lại dòng nhập lệnh; có thể có thông báo `Successfully installed` hoặc `Requirement already satisfied`.

Nếu script `.ps1` bị chặn, giữ các biến ở B1 và dùng lệnh Python tương đương:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[llm,dev]'
```

### B3. Kiểm tra PyTorch nhận GPU

Chạy nguyên dòng:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Điều cần thấy là:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

Số phiên bản có thể khác. Nếu là `False`, hãy xử lý theo F3 trước khi thử cấu hình 4bit.

### B4. Tải trọng số Qwen

“Trọng số” là các file số đã được huấn luyện tạo thành model. Mã Python và tokenizer trong dự án chưa thay thế các file này.

Chạy:

```powershell
.\.venv\Scripts\python.exe -m datamate stage-model --model qwen35-4b --output models/qwen35-4b
```

Lệnh này cần Internet. Nó tải checkpoint đã khóa revision, kiểm tra số tham số và ghi manifest/hash. Bạn không cần tự tìm revision hoặc tải model từ `main`.

**Lưu ý dung lượng:** pipeline hiện tải trọng số gốc, khoảng 9,3GB cho checkpoint Qwen 4B, rồi mới nén 4bit khi load vào GPU. Chọn cấu hình 4bit không làm lượt tải ban đầu thành một file 4bit nhỏ hơn.

Khi hoàn tất, kiểm tra:

```powershell
Test-Path .\models\qwen35-4b\artifact_manifest.json
```

Kết quả mong đợi là `True`. Nghĩa là bước chuẩn bị file model đã xong, chưa phải xác nhận inference GPU thành công.

Nếu tải bị ngắt, có thể chạy lại **đúng lệnh và đúng thư mục** trên. Bộ tải có thể tận dụng file/cache đã có. Nếu báo thư mục không cùng nguồn, xem F5.

### B5. Mở web với model thật

Chạy lại:

```powershell
.\tools\start-web.ps1
```

Mở hoặc tải lại trang [http://127.0.0.1:7860](http://127.0.0.1:7860), rồi đặt:

| Ô trên giao diện | Giá trị |
|---|---|
| **MODEL & CHẾ ĐỘ** | `Qwen 3.5 4B · Local 4bit` |
| **SPLIT** | `Train` |
| **DOANH NGHIỆP** | `biz_101` |
| **AS OF DATE** | Ngày 15/06/2026 |

Hỏi lại:

```text
Tổng doanh thu tháng này là bao nhiêu?
```

Lượt đầu phải kiểm tra hash và load model nên có thể chờ lâu hơn các lượt sau. Khi xử lý xong, mở **Dấu vết** để xem có truy vấn SQL và kết quả gì. Đáp án có thể là `number`, `string`, `list`… tùy câu hỏi, hoặc `abstain` nếu agent chưa trả lời được.

Nếu gặp `CUDA out of memory` hoặc lỗi context, xem F4. Không mặc định rằng cài thư viện và tải xong file đồng nghĩa laptop 8GB sẽ chạy được mọi doanh nghiệp.

### B6. Những việc chỉ cần làm một lần

Sau khi cài và chạy được:

- Thư viện giữ trong `.venv`; không cần cài lại mỗi lần mở web.
- Trọng số giữ trong `models/qwen35-4b`; không cần tải lại mỗi lần hỏi.
- Các lượt inference dùng file local. Internet cần cho bước cài/tải, không phải dịch vụ trả lời bên ngoài.

Lần dùng tiếp theo, quay về hai lệnh ở A7 rồi chọn preset Qwen 4bit.

## C. Chạy bằng lệnh và tìm file kết quả

### C1. Hỏi một câu từ PowerShell

Sau khi hoàn thành B, **dừng web trước** bằng Ctrl + C, rồi chạy:

```powershell
.\.venv\Scripts\python.exe -m datamate ask --config configs/qwen35-4b-local-4bit.json --business biz_101 --as-of 2026-06-15 --question 'Tổng doanh thu tháng này là bao nhiêu?'
```

Ý nghĩa các phần:

| Tham số | Ý nghĩa |
|---|---|
| `ask` | Hỏi một câu |
| `--config` | File cấu hình chọn model, 4bit và giới hạn chạy |
| `--business` | Chọn thư mục dữ liệu doanh nghiệp |
| `--as-of` | Ngày tham chiếu, định dạng `YYYY-MM-DD` |
| `--question` | Nội dung câu hỏi, đặt trong dấu nháy |

Kết quả JSON được in ngay trong PowerShell. Lệnh `ask` không tự tạo file batch `submission.jsonl`; hãy dùng `run` nếu cần file đó.

### C2. Chạy thử một batch nhỏ

“Batch” là chạy lần lượt nhiều câu hỏi trong file `questions.jsonl`.

```powershell
.\.venv\Scripts\python.exe -m datamate run --config configs/qwen35-4b-local-4bit.json --limit 3 --result-dir result/local-check-01
```

`--limit 3` chọn ba câu đầu của split train. `--result-dir` chỉ định thư mục ghi kết quả. Những câu đầu có thể thuộc doanh nghiệp có schema lớn; preset context 16K vẫn có thể abstain nếu prompt/tool results quá dài. Batch nhỏ này giúp kiểm tra luồng chạy, không đủ để kết luận chất lượng model.

Sau khi hoàn tất, mở File Explorer tới:

```text
E:\Bảo Lập Trình AI\ZALO_AI_2026\result\local-check-01
```

Bạn sẽ thấy:

```text
local-check-01/
├── submission.jsonl     ← đáp án, mỗi dòng một câu
├── logs/                ← dấu vết, lỗi và thời gian từng câu
├── run_summary.json     ← tổng kết lượt chạy
└── run_identity.json    ← thông tin để kiểm tra điều kiện resume
```

Bạn có thể mở `.json`/`.jsonl` bằng VS Code hoặc trình soạn thảo văn bản.

### C3. Chạy tiếp hoặc chạy mới

Nếu bị ngắt và muốn tiếp tục cùng lượt chạy:

```powershell
.\.venv\Scripts\python.exe -m datamate run --config configs/qwen35-4b-local-4bit.json --limit 3 --result-dir result/local-check-01 --resume
```

Giữ nguyên config, nguồn dữ liệu và tập câu hỏi. Nếu đã đủ ba câu thì không có câu mới để chạy; `written` có thể là 0.

Muốn làm một thí nghiệm mới, đổi tên thư mục output, chẳng hạn `result/local-check-02`. Đừng thêm `--resume` vào một lượt chạy mà bạn đã đổi model/config hoặc dữ liệu.

Muốn chạy toàn split thì bỏ `--limit 3`, nhưng chỉ làm sau khi đã kiểm tra model và tài nguyên. Bộ chấm local/harness so sánh nằm trong [tài liệu model](model-comparison.md); chưa cần dùng chúng để mở web.

### C4. Kiểm tra Docker riêng

Khi muốn kiểm tra đóng gói, mở Docker Desktop và chờ engine sẵn sàng, rồi chạy hai lệnh lần lượt:

```powershell
docker build -f docker/Dockerfile.dev -t datamate-dev:local .
```

```powershell
.\tools\docker-smoke.ps1
```

Kết quả mong đợi có dòng `Smoke test OK. Kết quả: ...`. Mở đúng thư mục `result/docker-smoke-...` mà lệnh in ra.

Image phát triển này chạy **demo** nên ba đáp án là `abstain`. Nó kiểm tra container offline, dữ liệu chỉ đọc và đầu ra; chưa chạy LLM. Chạy web ở A không cần Docker, còn image nộp thi cần đối chiếu guideline chính thức trong [docs/docker.md](docker.md).

## D. Fine-tune trên Google Colab

Bạn có thể đi thẳng từ A sang D nếu muốn dùng GPU Colab. Không cần tải model về laptop trước: notebook tự tải checkpoint vào Drive của bạn.

**Nếu chỉ muốn xem agent hỏi–đáp trên Colab:** làm D1–D2 để chuẩn bị hai ZIP, rồi upload **`notebooks/DataMate_Colab_Inference.ipynb`** thay cho notebook LoRA ở D3. Chọn GPU và chạy bốn ô lần lượt. Ô cuối cho sửa `BUSINESS_ID`, `AS_OF_DATE`, `QUESTION` rồi chạy lại từng câu, có trace SQL và file JSON trên Drive. Notebook inference không có bước training/thu hàng nghìn trajectory; mặc định `qwen35-4b`, có thể đổi `MODEL_KEY` sang `qwen35-9b` hoặc `gemma4-12b` khi đủ tài nguyên. Bước tiếp theo bên dưới dành cho fine-tune.

**Cập nhật ô load inference 03/11/2026:** bản notebook đang dùng giữ lựa chọn model của bạn. `MODEL_SOURCE_DIR` là bản model trên Drive; ô load tự chép vào `/content/models` một lần mỗi runtime rồi kiểm tra SHA-256 và nạp từ ổ tạm. Chạy lại ô có thể dùng lại cache, nhưng vẫn kiểm tra toàn bộ hash trước khi nạp. Đã load thành công thì chỉ chạy ô hỏi/batch. Dùng cả notebook mới và source ZIP mới; không cần upload lại train hoặc model. Xem [cách chạy và xử lý cache](pipeline-v2-2026-11-03.md#chạy-lại-colab-bằng-một-phiên-gpu).

### D1. Chuẩn bị hai file ZIP trên ổ E:

Trong PowerShell tại thư mục dự án, tạo lại gói mã nguồn:

```powershell
.\.venv\Scripts\python.exe tools/package_colab.py
```

File tạo ra là:

```text
E:\Bảo Lập Trình AI\ZALO_AI_2026\outputs\datamate-colab-source.zip
```

Tiếp theo, nén **riêng dữ liệu train** bằng Python của dự án:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/package_train.py
```

Chờ dòng **`TRAIN ZIP OK`**. Script đọc dữ liệu ở `train/train`, kiểm tra hash với nguồn train đã đăng ký rồi tạo `outputs/train.zip`. Nếu ZIP hiện có đã đúng, script kiểm tra và dùng lại. ZIP mới chỉ thay bản lỗi sau khi đã kiểm tra đủ file và hash; dữ liệu gốc không bị sửa. Không đưa `public_test` hoặc private test vào ZIP training.

Nếu trước đó dùng `Compress-Archive` và gặp `file ... is being used by another process`, chuyển sang lệnh Python trên. Nếu lệnh Python vẫn báo khóa file, đóng ứng dụng đang mở file đó; với web DataMate, nhấn **Ctrl + C tại cửa sổ đang chạy web**, rồi chạy lại lệnh. Không cần cài thêm thư viện hoặc tạo môi trường Conda.

Gói train phải chứa `answers.jsonl`, `questions.jsonl`, `rulebook.md`, `calendar.csv` và thư mục `data/` của train. Không sửa nội dung các file trước khi nén: pipeline kiểm tra hash nguồn.

### D2. Đưa hai ZIP lên Drive của bạn

Trên Google Drive, tạo thư mục tên **DataMate**, rồi tải hai file ZIP ở D1 vào đó. Cấu trúc mong đợi:

```text
My Drive/
└── DataMate/
    ├── datamate-colab-source.zip
    └── train.zip
```

Không cần giải nén các ZIP bằng tay trên Drive. Notebook sẽ làm việc đó.

Nếu cập nhật ZIP đã có trên Drive, dùng **Manage versions / Quản lý phiên bản → Upload new version / Tải phiên bản mới** trên đúng file. Tránh để nhiều file cùng tên hoặc để Drive đổi tên thành `train (1).zip`: notebook đọc đúng đường dẫn `DataMate/train.zip`.

Ô code đầu tiên phải hiện **`TRAIN OK: 162 file, fingerprint ...`** với gói train hiện tại. Notebook kiểm tra các file trong chính ZIP, đối chiếu hash nguồn rồi giải nén vào thư mục mới dưới `/content/.../uploaded-train/`. Chạy lại ô này không quét các bản giải nén cũ; model và kết quả trên Drive vẫn ở các đường dẫn riêng.

### D3. Mở notebook

Mở [Google Colab](https://colab.research.google.com/), chọn **Upload notebook / Tải lên sổ tay**, rồi chọn file:

```text
E:\Bảo Lập Trình AI\ZALO_AI_2026\notebooks\DataMate_Colab_LoRA.ipynb
```

Colab hỗ trợ nạp notebook có sẵn qua chức năng Upload notebook. Xem [FAQ chính thức](https://research.google.com/colaboratory/faq.html).

### D4. Chọn GPU

Trong Colab, mở **Runtime / Thời gian chạy → Change runtime type / Thay đổi loại thời gian chạy**, chọn GPU được cấp cho tài khoản, rồi kết nối runtime.

Khi chạy ô kiểm tra tài nguyên của notebook, đọc tên GPU và dung lượng VRAM thực tế từ `nvidia-smi`. Không suy ra tên GPU chỉ từ gói Colab bạn mua.

### D5. Sửa các biến ở ô code đầu tiên

Một **ô code** là khối Python có nút tam giác **▶** bên trái. Ở ô code đầu tiên, bạn sẽ thấy các biến dưới đây.

Nếu đặt file đúng như D2, giữ nguyên:

```python
SOURCE_ZIP = Path("/content/drive/MyDrive/DataMate/datamate-colab-source.zip")
TRAIN_ZIP = Path("/content/drive/MyDrive/DataMate/train.zip")
SAVE = Path("/content/drive/MyDrive/DataMate/runs/experiment-01")
```

`SAVE` là nơi lưu model, checkpoint và kết quả trên Drive. Với một thí nghiệm mới độc lập, dùng tên mới như `experiment-02` để tránh trộn kết quả. Khi tiếp tục một lượt bị ngắt, giữ đúng thư mục của lượt đó.

Chọn một model chính:

```python
MODEL_KEY = "qwen35-9b"
```

Hoặc thay đúng dòng đó bằng:

```python
MODEL_KEY = "gemma4-12b"
```

Không dán cả hai dòng chọn model vào cùng ô.

Các biến còn lại:

| Biến | Ý nghĩa và cách dùng ban đầu |
|---|---|
| `METHOD = "lora"` | Học adapter trên base model không nén; là phương án khởi điểm để thử khi có GPU lớn như ~96GB |
| `METHOD = "qlora"` | Học adapter trên base 4bit; có thể thử khi cần giảm VRAM, vẫn phải kiểm tra có fit hay không |
| `MAX_LENGTH = 16384` | Giới hạn độ dài mẫu; giữ 16K ban đầu vì prompt có thể đã dài hơn 8K |
| `MAX_STEPS = 50` | Chạy thử 50 bước cập nhật để đo tài nguyên trước khi chạy dài |
| `RESUME_CHECKPOINT = None` | Giữ nguyên khi bắt đầu mới |

Với GPU ~96GB như bạn từng cung cấp, có thể bắt đầu kiểm tra bằng `lora`. Đây là cấu hình để thử và đo, không phải khẳng định chắc chắn mọi batch/context đều vừa bộ nhớ.

### D6. Chọn bạn đang thử pipeline hay train cách dùng công cụ

Có hai lựa chọn cho `DATA_KIND`:

```python
DATA_KIND = "format"
```

Lựa chọn này dùng kết quả có sẵn để học ghi JSON. Nó giúp bạn làm quen cách chạy training, checkpoint và báo cáo. **Nó không dạy agent tự viết SQL hoặc tự giải câu hỏi.** Khi chọn `format`, ô thu baseline bỏ qua việc chạy agent thu trajectory.

Để fine-tune cách dùng công cụ, chọn:

```python
DATA_KIND = "trajectories"
COLLECT_BASELINE = True
```

“Trajectory” là nhật ký một quá trình giải: model gọi công cụ gì, nhận kết quả gì, rồi trả lời gì. Notebook chạy baseline trên train, đối chiếu đáp án cuối với nhãn train và giữ các trace đúng để tạo dữ liệu fine-tune.

**Ô thu trajectory có thể tốn đáng kể thời gian GPU trước khi bắt đầu LoRA.** `MAX_STEPS = 50` giới hạn số bước training, không giới hạn công việc thu baseline này. Nếu chưa có trace đúng, pipeline sẽ báo lỗi để kiểm tra baseline, không tạo lời giải giả.

### D7. Chạy từng ô từ trên xuống

Bấm **▶** ở từng ô, chờ ô đó kết thúc rồi mới sang ô tiếp theo. Lần đầu nên chạy từng ô để nhìn được lỗi ở đâu, thay vì bấm chạy tất cả.

| Ô / giai đoạn | Bạn cần nhìn thấy gì trước khi đi tiếp? |
|---|---|
| Kết nối Drive, giải nén | Bạn hoàn tất yêu cầu kết nối Drive; ô in được đường dẫn workspace/train/output |
| Cài runtime, kiểm tra GPU | Có `DataMate import OK: .../src/datamate/__init__.py`; in được tên GPU; kiểm tra CUDA không báo lỗi |
| Chuẩn bị checkpoint, tách doanh nghiệp | Tải/verify model xong; in được danh sách train/validation |
| Thu trajectory | Nếu dùng `trajectories`, có log baseline và dataset lọc được mẫu đúng; nếu dùng `format`, bước thu baseline được bỏ qua |
| LoRA / QLoRA | Training chạy và kết thúc; có `adapter/training_report.json` |
| Merge và đánh giá | Có model đã merge và `comparison/comparison.json` |

Nếu một ô báo lỗi đỏ, đọc phần cuối lỗi và xử lý ô đó trước; các ô sau thường cần kết quả của ô trước.

50 bước đầu là một lần thử tài nguyên, chưa phải kết luận model đã đủ tốt. Muốn chạy toàn bộ số epoch đã cấu hình, một thí nghiệm mới có thể đặt `MAX_STEPS = -1`; hãy quyết định sau khi đọc thời gian/VRAM thực đo của lần thử.

### D8. Tìm kết quả và mang về E:

Với `SAVE` mặc định, trên Drive tìm:

```text
DataMate/runs/experiment-01/
├── models/                 ← checkpoint gốc đã tải
├── baseline-train/         ← kết quả/log baseline nếu đã thu
├── format-data/            ← partition và mẫu định dạng
├── trajectory-data/        ← mẫu trace nếu chọn trajectories
├── adapter/                ← adapter, checkpoint, training_report.json
├── merged-model/           ← model sau merge, dùng để inference
├── comparison/             ← so sánh base và fine-tuned
└── pip-freeze.txt           ← phiên bản thư viện của lượt chạy
```

Thư mục chỉ có sau khi giai đoạn tương ứng thực sự chạy thành công.

Để dùng bản fine-tuned trên máy local, tải **toàn bộ** thư mục `merged-model` về:

```text
E:\Bảo Lập Trình AI\ZALO_AI_2026\models\qwen35-9b-datamate
```

Giữ nguyên các file và `artifact_manifest.json`, không chỉ tải một shard safetensors. Với model Gemma, đặt tên thư mục tương ứng để dễ phân biệt.

Với bản Qwen, tạo file mới `configs/qwen35-9b-finetuned-local.json` bằng trình soạn thảo, rồi dán toàn bộ cấu hình sau:

```json
{
  "backend": "transformers",
  "model_key": "qwen35-9b",
  "model_dir": "models/qwen35-9b-datamate",
  "mode": "single",
  "split_root": "train/train",
  "result_dir": "result/qwen35-9b-finetuned-local",
  "quantization": "4bit",
  "max_context_tokens": 8192,
  "max_new_tokens": 768,
  "question_timeout_seconds": 180
}
```

`model_key` vẫn là `qwen35-9b` để truy nguyên checkpoint gốc; `model_dir` trỏ tới trọng số đã fine-tune. Khởi động lại web và chọn preset mới `qwen35-9b-finetuned-local`. Các giới hạn VRAM/context ở B vẫn áp dụng.

Nếu bạn train Gemma, dùng `model_key = gemma4-12b` và cấu hình cùng họ Gemma thay vì dán ví dụ Qwen này. Việc tải model từ Drive về E: chưa chứng minh model đó sẽ fit laptop 8GB.

### D9. Nếu Colab bị ngắt

Các file lưu trên Drive có thể dùng lại. File trong `/content` thuộc máy ảo tạm của Colab; runtime có thể bị thu hồi, nên không dựa vào `/content` làm nơi lưu kết quả lâu dài. Đây là đặc tính được mô tả trong [FAQ Colab](https://research.google.com/colaboratory/faq.html).

Mở lại runtime, chạy lại các ô thiết lập/cài đặt và giữ cùng `SAVE`. Batch baseline sẽ thử tiếp tục từ submission đã ghi nếu thông tin lượt chạy vẫn khớp.

Với training, chỉ điền `RESUME_CHECKPOINT` khi có checkpoint thật, chẳng hạn:

```python
RESUME_CHECKPOINT = str(SAVE / "adapter/checkpoint-50")
```

Đó là ví dụ đường dẫn, không phải file luôn tồn tại. Cấu hình hiện lưu định kỳ mỗi 50 bước; bị ngắt trước checkpoint đầu thì có thể chưa có adapter checkpoint để resume. Chưa có `training_report.json` hoàn chỉnh thì chưa dùng bước merge như một lượt train đã hoàn tất.

Chi tiết kỹ thuật và cách tách holdout: [docs/training.md](training.md).

## E. Chọn Gemma và thử assistant

Bạn chưa cần làm phần này để mở web hoặc thử Qwen. Các preset Gemma hiện dùng base không nén; chúng phù hợp để kiểm tra trên GPU/RAM đủ lớn, chưa phải cấu hình đã xác minh cho laptop 8GB.

Trên một máy có môi trường LLM đã chuẩn bị, tải model chính:

```powershell
.\.venv\Scripts\python.exe -m datamate stage-model --model gemma4-12b --output models/gemma4-12b
```

Muốn bật MTP assistant thì tải thêm:

```powershell
.\.venv\Scripts\python.exe -m datamate stage-model --model gemma4-12b-assistant --output models/gemma4-12b-assistant
```

Trong web có ba lựa chọn khác nhau:

| Lựa chọn | Điều xảy ra |
|---|---|
| `Gemma 4 12B · Single` | Chạy riêng model chính |
| `Gemma 4 12B + MTP Assistant` | Assistant đề xuất token, model chính kiểm tra để thử tăng tốc sinh token |
| `Gemma 4 12B · Self review` | Gọi lại cùng model chính để kiểm tra đáp án; có thể tốn thêm lượt suy luận |

Assistant MTP không phải một AI riêng kiểm tra doanh thu hay sửa SQL. Chưa có số liệu để kết luận dùng assistant nhanh hơn trên phần cứng của bạn. Tổng Gemma + assistant theo metadata đã lưu khoảng 12,383B; ghép Qwen 9B với Gemma 12B sẽ vượt trần 15B.

Notebook hiện cho chọn **model chính** Qwen hoặc Gemma và mặc định baseline `single`; chọn Gemma trong notebook không tự bật assistant. Để làm thí nghiệm assistant, xem preset và harness ở [docs/model-comparison.md](model-comparison.md) sau khi đã chạy model chính ổn định.

## F. Xử lý lỗi thường gặp

### F1. Không có `.venv`, hoặc `No module named datamate`

Trong thư mục dự án, thử:

```powershell
.\tools\setup.ps1
```

Sau khi cài xong, chạy lại A3. Dùng đúng `.\.venv\Scripts\python.exe` khi gọi trực tiếp bằng Python; `python` hệ thống có thể là một môi trường khác.

Nếu script bị chặn, dùng các lệnh tương đương sau. Lệnh `python -m venv` chỉ cần khi môi trường chưa được tạo:

```powershell
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache/pip'
$env:TEMP = Join-Path (Get-Location) '.cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
```

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
```

Dự án khai báo Python từ 3.11 trở lên; môi trường local đã kiểm tra là Python 3.13.7. Nếu không có lệnh `python`, cần cài/chọn Python trước.

### F2. PowerShell báo không được chạy file `.ps1`

Bạn có thể khởi động web trực tiếp bằng Python, không cần đổi chính sách PowerShell:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m datamate serve --workspace . --port 7860
```

Để cài thư viện mà không chạy `setup.ps1`, dùng các lệnh pip tương đương trong F1 hoặc B2.

### F3. `No module named torch` hoặc `CUDA available: False`

Nếu chưa có `torch`, thực hiện B2. Nếu đã có mà CUDA vẫn `False`, kiểm tra lại đúng Python ở B3 và chạy:

```powershell
nvidia-smi
```

`nvidia-smi` nhận GPU không tự chứng minh PyTorch đã được cài bản CUDA phù hợp. Đối chiếu lệnh cài với [trang PyTorch chính thức](https://pytorch.org/get-started/locally/); không tiếp tục thử preset 4bit khi B3 còn `False`.

### F4. `CUDA out of memory` hoặc `Context vượt giới hạn`

Với lỗi hết VRAM hoặc `Some modules are dispatched on the CPU or the disk`, dừng lượt chạy khác và chọn **Qwen 3.5 4B · Local 4bit** để thử trên laptop. Dừng web bằng Ctrl + C rồi mở lại sau khi cập nhật mã/preset. Chỉ giữ một tiến trình inference khi thử. Nếu muốn đánh giá Qwen 9B/Gemma hoặc fine-tune, dùng Colab/GPU lớn hơn.

Với lỗi context, prompt/schema/tool results vượt giới hạn cấu hình. Preset 9B local cũ là 8K; preset 4B local mới là 16K. Một số prompt train đã dài hơn 11K trước tool results. Giảm context thêm không sửa được việc prompt vốn đã quá dài. Tăng context cần thêm bộ nhớ; nếu máy không đủ tài nguyên, chuyển thử model sang Colab/GPU lớn hơn.

Không tự cắt bỏ rulebook để làm câu hỏi chạy bằng mọi giá.

### F5. Chưa chuẩn bị model hoặc manifest không khớp

Nếu báo `Chưa có model đã chuẩn bị`, chạy B4 rồi kiểm tra `artifact_manifest.json`.

Nếu báo thư mục không cùng nguồn hoặc hash khác, kiểm tra bạn dùng đúng thư mục đã stage/merge và có đủ file. Không tự sửa SHA trong manifest. Có thể chọn thư mục tải mới rồi sửa `model_dir` cho đúng; không xóa dataset để xử lý lỗi model.

4B đã được tải và thử COUNT(*) thành công trong lần cập nhật ngày 01/11/2026. Nếu `Test-Path .\models\qwen35-4b\artifact_manifest.json` trả `True`, không cần tải lại; khởi động lại web rồi chọn **Qwen 3.5 4B · Local 4bit**. Một lượt thử đã mất khoảng 186 giây, nên đây là lựa chọn kiểm tra luồng local, chưa phải cấu hình tốc độ đã tối ưu. Chi tiết ở [báo cáo local 4B](local-4b-check-2026-11-01.md).

### F6. Trình duyệt không mở được localhost hoặc cổng đã dùng

Kiểm tra cửa sổ PowerShell ở A3 có còn chạy và có dòng `Uvicorn running` không. Nếu đã đóng cửa sổ đó, mở lại theo A7.

Nếu báo cổng 7860 đã được dùng, chọn cổng khác:

```powershell
.\tools\start-web.ps1 -Port 7861
```

Rồi mở [http://127.0.0.1:7861](http://127.0.0.1:7861). Không cần tắt một tiến trình lạ để lấy lại cổng 7860.

### F7. `submission.jsonl đã có` hoặc output training không rỗng

Chọn output mới cho thí nghiệm mới. Chỉ dùng `--resume` khi tiếp tục cùng nguồn/config/câu hỏi như C3.

Trong Colab, chọn `SAVE` mới cho thí nghiệm mới. Các thư mục output có kiểm tra để tránh âm thầm trộn hoặc ghi đè kết quả cũ.

### F8. `Nguồn chưa được đăng ký là TRAIN` hoặc không có trajectory đúng

Với lỗi nguồn, kiểm tra ZIP là gói train nguyên vẹn, không chứa public/private và không bị sửa các file dữ liệu/rulebook. Không sửa allowlist chỉ để bỏ qua lỗi.

Nếu không có trajectory đúng, mở log baseline xem lỗi truy vấn, context, thiếu dữ liệu hoặc runtime. Chuyển sang `format` chỉ giúp thử cơ chế training, không thay thế việc sửa chất lượng baseline.

### F9. Câu trả lời vẫn là `abstain`

Nếu đang chọn **Demo**, đó là hành vi được thiết kế. Nếu đang chọn LLM thật, mở trace/log để xem model đã làm gì và có lỗi nào. `abstain` có thể do hết ngân sách, context hoặc chưa có kết quả đủ điều kiện.

`unanswerable` có ý nghĩa khác: dữ liệu không đủ để trả lời. Không đổi mọi `abstain` thành `unanswerable` để cố có đáp án.

### F10. Colab báo `Upload only the original TRAIN package`

`Drive already mounted` chỉ báo Drive đã được kết nối; không cần `force_remount=True` để sửa lỗi tìm dữ liệu.

Notebook cũ báo cùng một lỗi khi không tìm thấy split đầy đủ hoặc tìm thấy nhiều split trong thư mục giải nén. Chưa thể kết luận ZIP sai chỉ từ câu thông báo đó. Để dùng bản đã sửa:

1. Chạy lại hai lệnh ở D1. `package_train.py` phải báo `TRAIN ZIP OK`.
2. Thay đúng hai file `datamate-colab-source.zip` và `train.zip` trên Drive theo D2.
3. Upload bản mới của `notebooks/DataMate_Colab_Inference.ipynb` vào Colab, chọn GPU rồi chạy từ ô đầu. Nếu dùng LoRA, mở bản mới của `DataMate_Colab_LoRA.ipynb`.

Bản mới báo cụ thể ZIP thiếu file nào, chứa bao nhiêu split hoặc file nào khác hash. Nó cũng xử lý ZIP có dấu `\` trong đường dẫn từ Windows. Nếu báo **`Gói source cũ`**, thay lại `datamate-colab-source.zip` trên Drive; chỉ thay notebook là chưa đủ. Không bỏ kiểm tra nguồn hoặc xóa trọng số model để xử lý lỗi ZIP.

### F11. Colab báo `No module named 'datamate'`

`datamate` là mã nguồn trong `src/datamate` của dự án. Sau khi giải nén, notebook cài bằng `pip install -e`; phiên Python đang chạy có thể chưa nhận đường dẫn editable vừa được cài trong tiến trình khác. [Python xử lý đường dẫn `.pth` qua module `site` khi khởi động](https://docs.python.org/3/library/site.html).

Để sửa ngay notebook đang mở, tạo một ô code **ngay trước ô load model**, dán và chạy:

```python
import sys, importlib
from pathlib import Path

if "WORK" not in globals():
    raise RuntimeError("Chạy lại ô 1 để giải nén mã nguồn trước.")
source_dir = (Path(WORK) / "src").resolve()
if not (source_dir / "datamate/__init__.py").is_file():
    raise RuntimeError(f"Thiếu mã nguồn ở {source_dir}. Chạy lại ô 1 với đúng gói source.")
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))
importlib.invalidate_caches()

import datamate
from datamate.config import Settings
print("DataMate import OK:", datamate.__file__)
```

Khi thấy `DataMate import OK`, chạy lại ô load model. Nếu lỗi chuyển sang thiếu thư viện khác, quay lại ô cài thư viện (ô 2), chờ cài xong và xử lý lỗi tại đó trước. Không cài một package tên `datamate` từ PyPI để sửa lỗi này.

Bản notebook mới tự thêm đường dẫn source và kiểm tra import ở cuối bước cài. Để dùng lâu dài, mở bản mới của notebook từ `notebooks/` trên máy. Với lỗi này, bạn không cần upload lại `train.zip`.

## Nên làm gì ngay bây giờ?

**Hoàn thành A2–A6 trước:** mở web, chạy demo, xem schema. Khi phần đó hoạt động, chọn B nếu muốn thử laptop, hoặc D nếu muốn dùng GPU Colab.

Các lệnh và tên file trong hướng dẫn đã được đối chiếu với CLI, script, config và notebook hiện có. Phần inference LLM/fine-tune cần tiếp tục kiểm tra trên runtime thật; không có số liệu accuracy hoặc thời gian GPU mới được tuyên bố trong tài liệu này.
