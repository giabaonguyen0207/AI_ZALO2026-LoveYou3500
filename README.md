# DataMate · AI Agent cho Zalo AI Challenge 2026

Ứng dụng Python theo kiến trúc **modular monolith**: đọc dữ liệu doanh nghiệp, cho LLM kiểm tra schema và gọi SQL, lấy đáp án trực tiếp từ kết quả query, rồi xuất JSONL đúng type. CLI, web local và Docker dùng chung mã nguồn. Có notebook Colab để chọn Qwen 3.5 9B hoặc Gemma 4 12B và fine-tune LoRA/QLoRA.

**Hiện đang dùng pipeline 6 — rollback 08/11/2026:** theo yêu cầu người dùng đã hoàn tác interface v7 và đồng bộ cả notebook/kit infer **100 lẫn 200 câu**. **225 kiểm thử qua**, chưa đo lại tốc độ GPU. Tải lại đúng gói và notebook, restart kernel; ô load phải hiện pipeline 6. V7 và log được giữ riêng. Xem [mốc khôi phục và cách chạy](docs/rollback-pipeline6-2026-11-08.md).

**Mới bắt đầu? Đọc [Hướng dẫn chạy từng bước](docs/huong-dan-chay-tung-buoc.md).** Làm phần A để mở web ngay; phần B cài Qwen trên laptop; phần D hướng dẫn Colab từ lúc chuẩn bị ZIP đến lấy model đã train về. Mỗi bước có lệnh, kết quả mong đợi và cách xử lý lỗi. Các mục bên dưới là tài liệu tham chiếu ngắn.

**Benchmark chung cho nhóm, 07/11/2026 — v2:** giữ nguyên **200 câu train**, seed **4000930841**, 28 doanh nghiệp. Dùng được với mọi thiết kế agent/model; chỉ nộp **submission đúng schema cuộc thi và metadata sáu trường**, không cần question log. Có ZIP gửi nhóm không nhãn, ZIP chấm riêng và bảng so từng câu/PA/tổng thời gian. **18 kiểm thử qua; chưa inference model trên mẫu này.** Xem [gói v2, hướng dẫn và kiểm chứng](docs/team-benchmark-train200-2026-11-07.md).

**Notebook Colab chạy 200 câu — rollback 08/11/2026:** notebook và kit đã trở về **engine pipeline 6**, dùng nguyên 200 câu/seed. Tải lại cả notebook và kit vì source được khóa hash; tên lượt mới kết thúc `team200-rollback-pipeline6-Nov8th-01`. Giữ output contract, log/review, resume và tự ngắt runtime. Xem [cách chạy và gói hiện tại](docs/team200-colab.md).

**Fine-tune Gemma trên Colab Pro:** đã chuẩn bị bộ seed từ hai lượt 72+100: **136 câu train / 144 quyết định assistant**, 17 câu cần rà soát và 18 câu validation seed giữ ngoài train. Có ZIP dataset, source riêng và notebook QLoRA cho Gemma; đã kiểm chứng bằng công cụ/tokenizer, chưa train GPU. Xem [hướng dẫn và dữ liệu bàn giao](docs/gemma-seed72-100-colab-pro.md).

**Lịch sử interface v7, đã hoàn tác ngày 08/11/2026:** đã sửa final/query_id, projection, tham số typed/khóa khách, cách giữ kỳ khi sửa API và ngân sách review. **241 kiểm thử qua**; replay toàn bộ 31 lỗi projection giữ được kết quả công cụ, 15 candidate đúng bị mất trước đây được chốt lại trong replay CPU. Chưa có PA/tốc độ checkpoint thật sau sửa. ZIP/notebook 100 câu mới dùng pipeline 7, diagnostic 9, giữ mẫu/seed cũ cho Qwen 9B và Gemma 12B. Phần model hiểu sai câu hỏi để đo và fine-tune sau. Xem [thay đổi, kiểm chứng và cách chạy](docs/interface-v7-2026-11-08.md). Các gói SFT và bộ 200 câu đã khóa giữ nguyên; kit Colab 200 câu được đóng gói lại riêng theo hướng dẫn bên trên.

**Kết quả Qwen pipeline 6, nhận ngày 07/11/2026:** cùng 100 câu cho **43 đúng, 20 sai, 37 abstain; PA 23%**, so với pipeline 5 là 49/32/19 và PA 17%. Trung bình 10,68 giây/câu, 311 calls. Dùng đúng source/checkpoint; 31 câu lỗi projection, 15 candidate đúng bị mất. Sau candidate không có lượt nào sửa plan; ba ca sai do kiểu tham số/key lọt guard. Đã chấm/hash audit và probe CPU, chưa sửa runtime tiếp. Xem [phân tích, nguyên nhân từng nhóm và ưu tiên xử lý](docs/inference-review-qwen9b-pipeline6-2026-11-07.md).

**Sửa chất lượng dùng chung, 07/11/2026:** pipeline **6** bổ sung kiểm tra plan, ngày serial/POS, nhóm danh mục, ngưỡng theo đơn/ngày, dòng tiền nợ, thẻ kho và lineage SQL. Preset Gemma 12B/Qwen 9B và Colab dùng cùng model kiểm tra candidate trước final (`self_review`); **213 kiểm thử qua**, chưa có PA/tốc độ GPU mới. ZIP inference/notebook mới giữ mẫu 100 cũ; ô load chặn source/module RAM cũ trước khi nạp trọng số. Xem [thay đổi, kiểm chứng, giới hạn và cách chạy](docs/shared-quality-fix-2026-11-07.md). Bộ Gemma SFT đã khóa giữ nguyên.

**Đủ 100 câu Qwen 9B sau sửa, nhận ngày 06/11/2026:** đã chạy đúng pipeline 5/backend 3: **49 đúng, 32 sai, 19 abstain; PA 17%**, trung bình **9,64 giây/câu**. Nhanh hơn Gemma fast v5 nhưng sai nhiều hơn; 31/79 auto-final sai. Đã chấm lại, kiểm tra hash/log và probe CPU từng nhóm lỗi; chưa sửa runtime tiếp hoặc đo PA mới. Xem [phân tích đầy đủ 32 câu sai, 19 abstain và thứ tự xử lý](docs/inference-review-qwen9b-100-2026-11-06.md).

**Đã sửa giao tiếp Qwen 9B/14B, 06/11/2026:** hợp nhất EOS từ checkpoint/tokenizer, xử lý nhiều marker cuối, làm rõ plan/feedback và dừng retry parse liên tiếp. **187 kiểm thử qua**; tokenizer thật và Qwen rất nhỏ ép token trên CPU xác nhận dừng đúng lượt, chưa benchmark checkpoint đầy đủ trên GPU sau sửa. Dùng ZIP inference và notebook mới, chọn `MODEL_KEY="qwen35-9b"` ở ô đầu, restart runtime và chạy bảy câu trước. Xem [bản sửa, kiểm chứng và cách chạy Colab](docs/qwen-protocol-fix-2026-11-06.md). Bộ Gemma SFT vẫn dùng source ZIP riêng đã khóa.

**Notebook Qwen 9B nhận tiếp, 06/11/2026:** một câu thử vẫn abstain vì ô load dùng **pipeline 4 / Generation tokens None**, prompt khớp bản trước sửa và cả hai raw lỗi parse marker cuối. Batch bảy câu bị guard chặn nên chưa chạy. Source ZIP local có pipeline 5/backend 3; chưa xác định từ notebook là ZIP Drive cũ hay module Python giữ trong RAM. Chưa đủ căn cứ kết luận Qwen kém; xem [bằng chứng và cách kiểm tra đúng phiên bản](docs/inference-review-qwen9b-notebook-2026-11-06.md).

**Cập nhật inference 03/11/2026:** pipeline v2 đã chạy lại cùng 72 câu development: **56 đúng, 1 sai, 15 bỏ qua; PA 76,39%**, khoảng 18 phút 15 giây riêng batch. Đây là mẫu đã dùng chẩn đoán, chưa đại diện hiệu năng private test. Đọc [phân tích lần 2, rules viết sẵn và lý do thiếu reasoning trong log](docs/inference-review-72-v2-2026-11-03.md), hoặc [những gì đã sửa và cách chạy Colab](docs/pipeline-v2-2026-11-03.md).

**Kết quả lượt 3 nhận ngày 04/11/2026.** Chấm lại 100 câu ngẫu nhiên từ toàn bộ train cho **49 đúng, 13 sai, 38 bỏ qua, PA 36%**. Hai lượt 2 và 3 chỉ trùng một câu. Xem [phân tích và kiểm chứng lỗi](docs/inference-review-100-v3-2026-11-04.md) và [cách lấy mẫu](docs/inference-round3-random100.md).

**Kết quả fast v5 nhận ngày 06/11/2026.** Đủ cùng 100 câu lượt 3: **46 đúng, 16 sai, 38 bỏ qua; PA 30%**, trung bình **23,19 giây/câu** so với 18,26 giây trước đó, không gồm load model. Số model calls giảm nhưng token đầu ra tăng; lỗi tập trung ở phạm vi tỷ lệ, khớp tên/nhóm, hợp đồng plan và kiểm tra/chốt đáp án. Đã chấm lại và kiểm chứng nguyên nhân bằng CPU; chưa triển khai các sửa đổi tiếp theo hoặc chạy inference mới. Xem [báo cáo so sánh, SQL/plan lỗi và thứ tự xử lý](docs/inference-review-fast-v5-2026-11-06.md).

**Thử thêm Pinkstackorg, 06/11/2026.** Notebook inference nhận `MODEL_KEY="qwen35-upscaled14b-v2"` cho `Pinkstackorg/Qwen-3.5-upscaled-14B-V2-noft`, pin revision trước cutoff và đếm đủ **14.599.485.040** tham số, gồm vision. Key `qwen35-9b` vẫn là Qwen chính thức. Kiểm tra tích hợp ban đầu dùng cấu trúc/tokenizer offline. Xem [cấu hình và cách chạy cùng fast v5](docs/qwen35-upscaled14b-v2-colab.md) và kết quả thực tế bên dưới. Bộ fine-tune vẫn chỉ Gemma.

**Kết quả Qwen 14B trước sửa, nhận ngày 06/11/2026.** Bundle có **10 câu, tất cả abstain**, trung bình 49,06 giây/câu; riêng bảy câu đầu là 51,76 giây/câu. Cả 35 output lỗi parse trước khi thực thi plan: generation thiếu stop ID `im_end`, parser chỉ bỏ một marker cuối, rồi các lượt sửa lỗi sinh dư/lặp. Gần 87,5% thời gian generation nằm ở retry. Xem [phân tích EOS/parser, độ trễ và hướng xử lý](docs/inference-review-qwen14b-2026-11-06.md); bản sửa triển khai sau phân tích được ghi bên trên.

**Bản giảm độ trễ fast v5, 05/11/2026.** Mặc định code ánh xạ trước; model lập plan rồi code tự tính/chiếu đáp án. Chỉ thử model sửa phần thiếu theo nhu cầu, có giới hạn và cache; có thể tắt hẳn bằng `schema_mode="code_only"`. Notebook chạy đúng bảy câu đầu của mẫu 100 cũ trước, `RUN_LIMIT=None` để resume đủ mẫu. Xem [thay đổi, kiểm tra và hướng dẫn chạy tại thời điểm triển khai](docs/fast-inference-v5-2026-11-05.md); kết quả model thật nhận sau đó ở báo cáo 06/11 bên trên.

**Pipeline v3 trước khi tối ưu, 05/11/2026.** Đã thêm chuẩn bị schema/rulebook bằng cùng model, plan cho từng câu và phép tính biên dịch có kiểm tra. Xem [thiết kế và giới hạn của bản trước](docs/pipeline-v3-2026-11-05.md).

**Bản trước fast v5 chạy dở bảy câu, 05/11/2026.** Cả bảy đều bỏ qua, trung bình 179,14 giây/câu; chuẩn bị schema/rulebook chiếm 83,38% thời gian. Xem [phân tích độ trễ dẫn đến fast v5](docs/inference-review-7-v4-2026-11-05.md).

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
| `configs/qwen35-9b.json` | Qwen 3.5 9B, self_review / pipeline 6 | 9,653B |
| `configs/qwen35-9b-local-4bit.json` | Qwen, thử local 4bit/context 8K | 9,653B |
| `configs/qwen35-upscaled14b-v2.json` | Pinkstackorg Qwen 3.5 upscaled V2, single / fast v5 | 14,599B |
| `configs/gemma4-12b.json` | Gemma 4 12B IT, self_review / pipeline 6 | 11,960B |
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
