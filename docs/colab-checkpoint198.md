# Colab Pro — checkpoint-198 làm baseline chung

Nguồn ngày14/11/2026: checkpoint do chủ dự án chỉ định tại `outputs/runs/Bảo/checkpoint-198`. Đã kiểm marker và checksum file gốc; đây là **LoRA rank16**, không phải full Qwen9B. Global step198, epoch stage0.5003158559696779, có hash adapter khởi tạo trước đó. Tổng1,5 epoch là thông tin chủ dự án; chưa đủ parent report để xác minh độc lập. Không train thêm và không cộng adapter epoch1 lần nữa.

## File cần upload

1. Notebook `notebooks/DataMate_Colab_Checkpoint198_Compare.ipynb` lên Colab.
2. Gói source mới: đặt trên Drive thành `MyDrive/DataMate/datamate-colab-source.zip`.
3. Gói adapter đã chuẩn bị: `MyDrive/DataMate/checkpoint-198-inference.zip`. Gói có tokenizer/template của chính checkpoint, marker, snapshot report; không có optimizer/RNG/pickle và không ghi đè checkpoint gốc.
4. Dataset đã giải nén và JSON array các ID benchmark; mặc định `MyDrive/DataMate/data/train` và `MyDrive/DataMate/benchmark/ids.json`. Nên chọn holdout chưa dùng fine-tune để đánh giá chất lượng; không bắt buộc Team200.
5. **Đúng base đã train**: sửa `BASE_SOURCE_DIR` ở ô đầu theo nơi lưu trên Drive.

Base phải có `artifact_manifest.json` SHA256:

```text
f8b32a3f0b36609c4065a72af44938fa27ba68e4b2d70ef45a8d43b1cc37399e
```

Base trong archive local có manifest `31cb4f…`, khác hash trên; chưa kết luận các weights khác nhau, nhưng không đủ bằng chứng để tự thay thế base. Chủ dự án đang tìm đúng bản Drive. Có thể kiểm nhanh trên Colab:

```python
import hashlib
from pathlib import Path
p = Path('/content/drive/MyDrive/DataMate/models/qwen35-9b/artifact_manifest.json')
print(hashlib.sha256(p.read_bytes()).hexdigest())
```

## Chạy notebook

- Chọn GPU, đủ RAM hệ thống cho bước merge CPU trên Colab. Policy hiện tại cần ít nhất32GiB **RAM còn trống** cho Qwen9B; preflight kiểm cả cgroup trước khi load. Colab Pro cung cấp GPU/High RAM tùy khả dụng và credit, không bảo đảm một cấu hình cố định: [Google FAQ](https://research.google.com/colaboratory/faq.html).
- Ô1 mount Drive/giải nén source; sửa `TRAIN_ROOT`, `IDS_FILE`, `BASE_SOURCE_DIR` theo dữ liệu thực.
- Ô2 cài **một stack chung** Torch/vLLM/RAG/PEFT trước baseline. Restart kernel nếu pip đổi thư viện đã import.
- Ô3 kiểm checkpoint/base; merge một lần bằng CPU child trên **Colab**, giữ MTP và seal full inventory. Output persistent `models/qwen35-9b-checkpoint-198-merged`; lần sau chỉ dùng lại khi lineage/checksum đúng checkpoint. RAG weights được stage online ở bước chuẩn bị nếu chưa có, không tải trong inference.
- Ô4 chạy `compare_system.py`: baseline Torch → cùng checkpoint+vLLM → cùng checkpoint+Torch+RAG. Tất cả dùng cùng model_dir merged, source, ID, dtype và budget. `REFERENCES_FILE=None` khi không có nhãn; nhãn nếu có chỉ được đọc sau tất cả inference.

Kết quả nằm trong `runs/Bảo/<RUN_NAME>/comparison/`; log merge ở `prepare-merge.log`, thời gian chuẩn bị ở `preparation.json`. Chọn thư mục comparison mới cho phép đo mới. Merge/copy/cài đặt là chuẩn bị riêng; startup model/index/engine của từng arm vẫn được tính trong benchmark. Khi đóng Docker cuối, copy **model merged đã seal**, không chạy merge/pip/download tại inference BTC.

Đây là workflow chuẩn bị; chưa chạy full checkpoint trên Colab/GPU hoặc chứng minh PA/VRAM/tốc độ. Baseline này là **model đã fine-tune, không RAG**, không phải pretrained gốc.

## Tạo lại gói khi cần

Chỉ thực hiện trên bản checkpoint đúng, output ZIP mới:

```powershell
python tools/checkpoint_inference.py package --checkpoint 'E:\Bảo Lập Trình AI\ZALO_AI_2026\outputs\runs\Bảo\checkpoint-198' --reported-total-epochs 1.5 --output outputs/packages/checkpoint-198-inference.zip
python tools/package_colab.py --output outputs/packages/datamate-colab-source.zip
```
