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
- **Install:** tạo venv riêng dưới `/content/datamate-envs/`, cài một stack chung Torch/vLLM/RAG/PEFT trước baseline. Pip dùng `--python` để cài vào đúng venv, không cài stack inference vào kernel Colab. Merge/stage/compare đều dùng `RUNTIME_PYTHON` và environment riêng. Colab Pro GPU/High RAM tùy khả dụng và credit: [Google FAQ](https://research.google.com/colaboratory/faq.html).
- **Prepare:** xác minh ZIP/report/IDs/base, merge một lần bằng CPU child **trên Colab**, giữ tensor MTP và seal inventory. Policy merge Qwen9B cần tối thiểu 32GiB RAM hệ thống còn trống; preflight kiểm RAM trước load. Không merge full model trên laptop. RAG stage online chỉ ở ô này.
- **Compare:** `compare_system.py` chạy **Torch baseline → vLLM → Torch+RAG**, cùng generator merged, source, IDs, dtype và budget; gold chỉ được đọc khi mọi inference xong. `DTYPE='float16'` mặc định cho cả ba; có thể chọn `bfloat16` cho cả ba nếu GPU hỗ trợ, trước khi đo baseline.

Model merged mới lưu ở `MyDrive/DataMateSQLSFT/models/qwen35-9b-stage2-comparison-merged`. Lineage pin **toàn gói inference**, gồm tokenizer/template, ngoài report và base. Bản merge cũ chưa có pin gói sẽ không được tự coi là cache đúng; giữ nguyên bản đó và dùng thư mục output mới. Khởi động lại kernel có thể dùng lại bản merged mới đã kiểm.

Kết quả ở `MyDrive/DataMate/runs/Bảo/stage2-<thời-gian>/comparison/`; các log gồm `prepare-merge.log`, `preparation.json`, ba thư mục arm và `comparisons.json`. Thời gian ô Prepare tách riêng; cài pip/giải nén source cũng tách riêng. Startup model/index/engine vẫn tính trong từng lượt inference. Khi đóng Docker cuối, copy full model merged và thư viện đã chuẩn bị, không merge/cài/tải trong inference BTC.

## Nếu bản notebook cũ lỗi tại pip check

Log Colab ngày14/11 cho thấy các xung đột có sẵn/sau cài stack giữa google-colab, Gradio, RAPIDS, google-adk và runtime mới. Có hai ràng buộc không thể đồng thời thỏa trong cùng environment: google-colab yêu cầu requests2.32.4 trong khi vLLM0.33 yêu cầu requests>=2.33; Gradio yêu cầu Pillow<13 trong khi vLLM yêu cầu Pillow>=13. Nguồn: [requirements vLLM0.33](https://github.com/vllm-project/vllm/blob/v0.33.0/requirements/common.txt).

Bản đã sửa tách môi trường inference khỏi các package kernel. `pip check` vẫn phải đạt trong venv; nếu còn lỗi, thông báo đầy đủ được in trực tiếp và lưu `SAVE/pip-check.txt`. Phiên bản runtime lưu tại `SAVE/runtime-freeze.txt`. Không bỏ kiểm tra dependency và không hạ requests/Pillow của runtime để chạy theo package không dùng trong inference.

Để cập nhật: dùng notebook và source ZIP mới, giữ lại các đường dẫn bạn đã sửa ở ô1, rồi chạy1→4. Không cần upload lại ZIP adapter hoặc train/merge lại nếu đã có model merged đúng gói. Venv và pip cache nằm trên ổ local, dùng lại khi chạy lại cùng cấu hình dependency. Cách bootstrap venv không cần ensurepip: [pip --python](https://pip.pypa.io/en/stable/topics/python-option/).

## Nếu baseline xong nhưng vLLM lỗi image processor

Log chủ dự án cung cấp ngày14/11/2026: trên A10040GB, baseline kết thúc process sau3362,8s; vLLM dừng ở startup sau94,9s, chưa gọi model. Lỗi gốc là thiếu `preprocessor_config.json` trong đường chạy khởi tạo processor ảnh của Qwen, không phải OOM. Đây là bằng chứng chẩn đoán một lượt, chưa xác nhận chất lượng baseline hoặc hiệu năng vLLM.

Adapter chỉ gửi token văn bản. Bản source mới đặt `language_model_only=True` khi tạo engine, tránh khởi tạo xử lý ảnh/video. vLLM0.33 có tùy chọn này: [MultiModalConfig](https://docs.vllm.ai/en/v0.33.0/api/vllm/config/multimodal/#vllm.config.multimodal.MultiModalConfig.language_model_only).

Nếu giữ baseline cũ, **giữ nguyên source WORK cũ**, không chép source mới đè lên nó: phép so kiểm hash toàn bộ source. Upload riêng `tools/continue_system_comparison.py` vào `/content/continue_system_comparison.py`, rồi gọi bằng Python trong venv cũ:

```python
command = [RUNTIME_PYTHON, '-u', '/content/continue_system_comparison.py',
           '--project-root', str(WORK), '--comparison-dir', str(COMPARE_OUTPUT)]
if REFERENCES_FILE is not None:
    command += ['--references', str(REFERENCES_FILE)]
process = subprocess.Popen(command, cwd=WORK, env=RUNTIME_ENV,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding='utf-8', errors='replace', bufsize=1)
try:
    for line in process.stdout:
        print(line, end='', flush=True)
    if process.wait() != 0:
        raise RuntimeError('Recovery failed; see the error above and the new recovery log')
finally:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
```

Tool kiểm baseline đã hoàn tất/đủ ID, source/model manifest và phiên bản thư viện; không giả định hoàn tất đồng nghĩa chất lượng tốt. Nó truyền chế độ text-only qua `extra_engine_args` đã được source cũ hỗ trợ, không sửa weights/manifest/source cũ, không chạy lại baseline. Mỗi lần tạo thư mục `comparison.../recovery-.../` mới, giữ cả log lỗi trước, dùng cache riêng, giữ watchdog và in log trực tiếp. Mặc định chạy vLLM rồi Torch+RAG trên đúng ID/dtype/budget đã ghi trong baseline, sau đó tạo `comparisons.json` so với baseline gốc. Chỉ đọc gold sau toàn bộ lượt yêu cầu. Có thể thêm `--arms rag` để chỉ chạy RAG hoặc `--arms vllm` để chỉ chạy vLLM; các kết quả nằm ở recovery riêng, không tự ghép hai recovery.

Chưa kiểm fix bằng full model trên GPU; phải chạy lại vLLM để xác nhận startup/inference thực tế. Không tải một processor ngẫu nhiên hoặc sửa config/manifest model đã seal để bỏ lỗi.

Log recovery tiếp theo xác nhận đã qua lỗi processor nhưng dừng ở FlashInfer JIT vì không tìm thấy `ninja`. Venv cần có Ninja và thư mục `bin` trong `PATH` của worker; gọi riêng Python venv không tự kích hoạt PATH. Bản notebook mới sửa environment và extra vLLM có Ninja. Với session cũ, sửa `RUNTIME_ENV` và cài riêng Ninja nếu chưa tìm thấy, kiểm CUDA/C++ compiler trước khi chạy lại recovery; giữ runtime/source/baseline cũ. Xem [nhật ký lỗi và yêu cầu Docker](runtime-issues-2026-11-14.md#3-filenotfounderror--ninja-khi-warmup-sampler).

## Tạo lại gói từ kết quả đã hoàn tất

Chọn output ZIP mới nếu file cũ đã có:

```powershell
python tools/checkpoint_inference.py package-adapter --run 'E:\Bảo Lập Trình AI\ZALO_AI_2026\outputs\runs\Bảo\qwen35-9b-sql-stage2-v1-full-01' --parent-adapter 'E:\Bảo Lập Trình AI\ZALO_AI_2026\outputs\runs\Bảo\qwen35-9b-sql-v1.3-full-01\adapter' --output outputs/packages/qwen35-9b-stage2-final-inference.zip
python tools/create_finetuned_compare_colab.py
python tools/package_colab.py --output outputs/packages/datamate-colab-source.zip
```

ZIP được pin nguyên file: tái đóng ZIP có thể đổi hash do timestamp. Khi chủ ý tạo release mới, cập nhật profile bằng metadata của gói mới rồi sinh notebook/source ZIP mới; không bỏ bước kiểm hash.
