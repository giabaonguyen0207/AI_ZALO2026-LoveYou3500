# Colab Pro — baseline từ bản fine-tune stage 2 hoàn tất

Nguồn: chủ dự án cung cấp ngày **14/11/2026** tại `outputs/runs/Bảo/qwen35-9b-sql-stage2-v1-full-01`. Adapter thực sự nằm trong thư mục con **`adapter/`**. Đây vẫn là LoRA, chưa phải full model đã merge.

Đã đọc report/config và kiểm checksum weights của stage 2 lẫn adapter cha:

| Bản | Step | Epoch yêu cầu / quan sát | Report SHA256 bắt đầu |
|---|---:|---|---|
| Epoch 1 cha | 298 | 1 / 1,0 | `fcc245948098…` |
| Stage 2 hoàn tất | 198 | 0,5 / 0,5003158559 | `9a09f391253a…` |

Hash parent trong stage 2 khớp cả report và hai file adapter epoch 1: chuỗi **1 + 0,5 epoch** được đối chiếu từ artifact đã cung cấp. Trọng số stage 2 là `f346ad51cd37…`, trùng snapshot `checkpoint-198`; khác biệt chính là nay giữ nguyên báo cáo training hoàn tất và đủ bằng chứng nguồn gốc. Không train tiếp, không load chồng adapter epoch 1 lần nữa.

**Baseline của phép so mới = Qwen9B đã fine-tune, chạy Torch, RAG tắt.** Hai bản còn lại là cùng model đó + vLLM hoặc + RAG. Chưa chạy/chấm ba bản trên GPU; training loss trong report không phải PA.

## Ba file đưa lên Colab/Drive

1. Mở notebook `notebooks/DataMate_Colab_Finetuned_Compare.ipynb` trên Colab Pro.
2. Upload source ZIP mới thành `MyDrive/DataMate/datamate-colab-source.zip`.
3. Upload `qwen35-9b-stage2-final-inference.zip` vào `MyDrive/DataMate/`, giữ nguyên tên.

Gói adapter giữ byte nguyên bản `training_report.json`, tokenizer/template, run identity và parent report; không chứa optimizer, RNG, checkpoint con, smoke adapter hoặc log. Notebook pin SHA256 của ZIP, weights và report theo `configs/comparison/finetuned-artifact.json`. Không dùng ZIP snapshot cũ thay ZIP hoàn tất dù weights giống nhau.

Dataset/IDs và base model nhận riêng, không nằm trong source ZIP. Ở ô đầu sửa:

```python
TRAIN_ROOT = Path('/content/drive/MyDrive/DataMate/data/train')
IDS_FILE = Path('/content/drive/MyDrive/DataMate/benchmark/ids.json')
BASE_SOURCE_DIR = Path('/content/drive/MyDrive/DataMateSQLSFT/models/qwen35-9b-c202236')
REFERENCES_FILE = None  # hoặc đường dẫn gold JSONL hợp lệ, chỉ đọc sau inference
```

`IDS_FILE` là JSON array ID cố định, ví dụ `["id-a", "id-b"]`. Dùng benchmark bạn chọn; nên dùng holdout chưa train. Validation phát triển đã được xem trước đó không phải unseen/private test. Notebook kiểm ID tồn tại trong split **trước** khi merge/tải RAG.

## Base phải đúng bản đã train

Base phải có manifest SHA256:

```text
f8b32a3f0b36609c4065a72af44938fa27ba68e4b2d70ef45a8d43b1cc37399e
```

Đường dẫn Drive mặc định lấy theo builder training thực tế; chưa trực tiếp xác minh được thư mục Drive đó từ máy local. Notebook cũng thử cache runtime `/content/datamate-models/qwen35-9b-f8b32a3f0b36609c` nếu còn trên cùng VM. Cả hai chỉ được dùng khi manifest khớp, sau đó kiểm toàn bộ file base. Nếu không khớp, sửa `BASE_SOURCE_DIR`; không tự thay base archive `31cb4f…` hoặc download một bản khác.

Kiểm nhanh sau khi mount Drive:

```python
import hashlib
p = BASE_SOURCE_DIR / 'artifact_manifest.json'
print(hashlib.sha256(p.read_bytes()).hexdigest())
```

## Chạy bốn ô

- **Setup:** mount Drive, giải nén source theo hash, đặt đường dẫn và RUN_NAME mới.
- **Install:** một stack chung Torch/vLLM/RAG/PEFT trước baseline. Nếu pip đổi thư viện đã import, restart kernel rồi chạy lại. Colab Pro GPU/High RAM tùy khả dụng và credit: [Google FAQ](https://research.google.com/colaboratory/faq.html).
- **Prepare:** xác minh ZIP/report/IDs/base, merge một lần bằng CPU child **trên Colab**, giữ tensor MTP và seal inventory. Policy merge Qwen9B cần tối thiểu 32GiB RAM hệ thống còn trống; preflight kiểm RAM trước load. Không merge full model trên laptop. RAG stage online chỉ ở ô này.
- **Compare:** `compare_system.py` chạy **Torch baseline → vLLM → Torch+RAG**, cùng generator merged, source, IDs, dtype và budget; gold chỉ được đọc khi mọi inference xong. `DTYPE='float16'` mặc định cho cả ba; có thể chọn `bfloat16` cho cả ba nếu GPU hỗ trợ, trước khi đo baseline.

Model merged mới lưu ở `MyDrive/DataMateSQLSFT/models/qwen35-9b-stage2-comparison-merged`. Lineage pin **toàn gói inference**, gồm tokenizer/template, ngoài report và base. Bản merge cũ chưa có pin gói sẽ không được tự coi là cache đúng; giữ nguyên bản đó và dùng thư mục output mới. Khởi động lại kernel có thể dùng lại bản merged mới đã kiểm.

Kết quả ở `MyDrive/DataMate/runs/Bảo/stage2-<thời-gian>/comparison/`; các log gồm `prepare-merge.log`, `preparation.json`, ba thư mục arm và `comparisons.json`. Thời gian ô Prepare tách riêng; cài pip/giải nén source cũng tách riêng. Startup model/index/engine vẫn tính trong từng lượt inference. Khi đóng Docker cuối, copy full model merged và thư viện đã chuẩn bị, không merge/cài/tải trong inference BTC.

## Tạo lại gói từ kết quả đã hoàn tất

Chọn output ZIP mới nếu file cũ đã có:

```powershell
python tools/checkpoint_inference.py package-adapter --run 'E:\Bảo Lập Trình AI\ZALO_AI_2026\outputs\runs\Bảo\qwen35-9b-sql-stage2-v1-full-01' --parent-adapter 'E:\Bảo Lập Trình AI\ZALO_AI_2026\outputs\runs\Bảo\qwen35-9b-sql-v1.3-full-01\adapter' --output outputs/packages/qwen35-9b-stage2-final-inference.zip
python tools/create_finetuned_compare_colab.py
python tools/package_colab.py --output outputs/packages/datamate-colab-source.zip
```

ZIP được pin nguyên file: tái đóng ZIP có thể đổi hash do timestamp. Khi chủ ý tạo release mới, cập nhật profile bằng metadata của gói mới rồi sinh notebook/source ZIP mới; không bỏ bước kiểm hash.
