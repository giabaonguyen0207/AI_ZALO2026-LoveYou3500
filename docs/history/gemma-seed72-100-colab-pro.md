# Gemma · Bộ fine-tune khởi đầu từ 72+100 câu · Colab Pro

Chuẩn bị ngày 05/11/2026 theo yêu cầu đã duyệt; sau đó người dùng xác nhận **chỉ dùng Gemma**. Gói này dành cho **Gemma 4 12B**, checkpoint đang dùng trong hai lượt inference. Chưa chạy GPU fine-tune, chưa có PA mới của model.

## Kết quả chuẩn bị

Hai bundle `gemma4-12b-stress72-pipeline-v2-01` và `gemma4-12b-stress100-baseline-v3` có 172 log, 171 câu riêng biệt. Nguồn được đối chiếu fingerprint với TRAIN đã đăng ký; public/private test không tham gia.

| Nhóm | Số câu |
|---|---:|
| Được nhận vào train | **136** |
| Còn trong hàng chờ sửa/rà soát | **17** |
| Seed thuộc doanh nghiệp validation, giữ ngoài train | **18** |
| Tổng câu riêng biệt | **171** |

136 câu tạo **144 quyết định assistant**: 136 plan và 8 lượt final `unanswerable` có bằng chứng công cụ. Bao phủ cả tám type: number 51, percent 25, date 16, list 10, table 7, string 10, boolean 9, unanswerable 8.

Trong 71 ứng viên train có lịch sử đúng và không ghi lỗi, nhận 70; một câu vẫn phải giữ lại vì công cụ hiện chưa tái dựng được tồn theo lô. Có thêm **66 câu được sửa/kiểm chứng** từ nhóm 82 câu cần xử lý. Các câu về nhóm hàng/nhóm khách, giới tính, kê đơn, dòng tiền trong kỳ, lý do trả hàng và hai câu tồn theo lô chưa đủ căn cứ thực thi nằm trong `repair_queue.jsonl`.

Đây là **mẫu được biên tập và kiểm chứng bằng công cụ**, không phải trajectory mới do Gemma sinh. Log cũ không lưu prompt ban đầu; không giả lập rằng đã khôi phục được prompt lịch sử. Câu hỏi/rulebook được dùng để chú thích task, sau đó chạy qua agent fast v5 hiện tại; nhãn tham chiếu chỉ được đọc sau khi niêm phong đề xuất và bản thực thi.

Model học action JSON, plan/SQL, không học chép số đáp án. Code tự chiếu final không được thêm thành một lượt model giả. Cấu hình review có ID để ghép dữ liệu huấn luyện; file review không được agent inference import hoặc tra cứu để trả lời.

## Kiểm tra dữ liệu

- Tất cả mẫu được nhận có type/giá trị/thứ tự khớp nhãn TRAIN sau thực thi, không có lỗi tool trong trace mới. Kiểm tra này không chứng minh tuyệt đối mọi diễn giải ngôn ngữ; chưa có người thứ hai kiểm định độc lập toàn bộ.
- Giữ cách tách 24 doanh nghiệp train / 6 doanh nghiệp validation, seed 42. `validation_seed_ids.json` có 18 câu từng được chẩn đoán; `validation_ids.json` giữ toàn bộ 630 câu của sáu doanh nghiệp. Đây là validation phát triển đã có tiếp xúc, không phải tập giữ kín hoàn toàn.
- Chỉ kiểm tokenizer **Gemma** ở release này: 144 sequence dài **8.447–13.271 token**, target dài tối đa **327 token**, nằm trong context 16.384 và ngân sách output 768. Không cắt mẫu hay nhân bản diễn đạt.
- Chỉ tính loss trên assistant cần học. Có hỗ trợ chọn vị trí assistant được huấn luyện nếu bổ sung mẫu sửa lỗi sau này; action sai trong lịch sử có thể làm ngữ cảnh mà không thành target.
- Các query do công cụ tạo được chuyển sang dạng subquery tương đương khi cần cho guard hiện tại; binding tham số được giữ bằng numbered parameters. Hai biểu thức được chạy và so kết quả trên nguồn trước khi đọc nhãn. Truy vấn nợ quá khứ chỉ lấy cột cần thiết và sổ thu đến ngày chốt, không đọc dư nợ snapshot tương lai.
- Bản build không gọi model thật và không train GPU. Có 155 kiểm thử phần mềm đạt ở lượt kiểm tra toàn bộ; 13 kiểm thử seed/Colab được chạy lại sau cập nhật riêng Gemma. Kiểm tra GPU thực tế vẫn cần chạy trên Colab.

## Những file cần dùng

1. `outputs/seed72_100_gemma_v1.zip`: dataset đã niêm phong.
2. `outputs/datamate-colab-gemma-seed-source.zip`: source riêng cho lượt fine-tune này.
3. `notebooks/DataMate_Colab_Seed72_100_QLoRA.ipynb`: notebook mới, chỉ dùng Gemma.

Dataset nguyên bản nằm ở `outputs/training/seed72_100_gemma_v1/`. Các file chính:

| File | Mục đích |
|---|---|
| `train.jsonl` | 136 hội thoại SFT theo fast v5 |
| `validation.jsonl` | Rỗng; không tạo gold action từ holdout |
| `validation_questions.jsonl` | 18 câu validation trong seed |
| `validation_seed_ids.json`, `validation_ids.json` | Đánh giá generation trên 18 / 630 câu |
| `inventory.jsonl` | Nguồn, lịch sử và trạng thái của 171 câu |
| `repair_queue.jsonl` | 17 câu chưa được nhận, kèm nguyên nhân |
| `execution_audit.jsonl` | Đề xuất, trace thực thi và kết quả kiểm chứng |
| `review_annotations.json` | Chú thích task được niêm phong trước chấm nhãn |
| `token_report.jsonl` | Độ dài context/target theo từng quyết định |
| `manifest.json`, `report.json`, `data_report.md` | Hash, nguồn, cách chia và báo cáo |

## Chạy trên Colab Pro

1. Upload hai ZIP vào `/MyDrive/DataMate/`, mở notebook mới trên Colab. Dùng runtime mới để tránh giữ agent inference và các thư viện cũ trong RAM.
2. Giữ model ở `/MyDrive/DataMate/models/gemma4-12b/`, hoặc sửa `MODEL_SOURCE_DIR` cho đúng thư mục đã dùng inference. Chỉ khi chưa có manifest, notebook mới stage checkpoint theo revision đã pin. Trọng số được chép sang ổ tạm để train.
3. Chọn GPU, chạy ô cài thư viện và kiểm dữ liệu. PyTorch/CUDA của Colab được giữ nguyên; notebook dùng Transformer 5.22.0 từng được ghi nhận ở inference trước, cài PEFT/datasets/bitsandbytes và lưu `pip-freeze.txt`. Đây chưa phải stack đã được chứng nhận fine-tune GPU.
4. Ô preflight kiểm hash ZIP, từng file, fingerprint TRAIN và hash source cùng release, rồi hiển thị GPU/VRAM thật. Mặc định profile 16K dừng dưới **22 GiB**. Đây là ngưỡng thử do dự án chọn, không phải cam kết đủ VRAM. Nếu Colab cấp GPU lớn như A100 thì dùng để thử; không đổi xuống 8K vì các mẫu hiện đều dài hơn 8K.
5. Chạy smoke: **một optimizer step trên sequence dài nhất**. Adapter smoke lưu riêng, không dùng làm điểm bắt đầu của pilot. Nếu OOM, dừng và đổi GPU/profile; không cắt rulebook hoặc âm thầm bỏ mẫu. Dùng `RUN_TAG` mới khi đổi cấu hình sau lượt lỗi.
6. Chạy pilot mặc định **5 optimizer steps**, QLoRA NF4, rank 16, alpha 32, attention-only, batch 1, gradient accumulation 8, learning rate 5e-5. Đây là cấu hình thử để đo chi phí và bộ nhớ, chưa phải cấu hình tối ưu. Muốn chạy một epoch sau khi xem pilot, dùng `MAX_STEPS=-1`.
7. Checkpoint và báo cáo ở `/MyDrive/DataMate/runs/<RUN_TAG>/`. Khi Colab ngắt, đặt `RESUME_CHECKPOINT` tới checkpoint trong `adapter/`; giữ model/dataset/cấu hình. `run_identity.json` chặn resume khác dữ liệu hoặc cấu hình học; tăng giới hạn bước vẫn được phép.

Colab Pro cấp tài nguyên theo khả dụng, không cố định loại GPU; High-RAM là RAM hệ thống, không làm tăng VRAM của GPU. [Google Colab FAQ](https://research.google.com/colaboratory/faq.html). QLoRA nén base 4bit và học adapter để giảm nhu cầu bộ nhớ, không bảo đảm mọi context đều fit mọi GPU. [Hướng dẫn PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization).

Notebook mặc định `RUN_EVALUATION=False`. Khi muốn merge/so base và fine-tuned, bật tùy chọn này và dùng `train.zip` gốc đã có trên Drive. Merge cần thêm RAM/VRAM/dung lượng vì xuất model không nén. Pilot so trên 18 câu phát triển; mở rộng sang 630 câu khi đã kiểm runtime. So PA, sai/bỏ qua, type và thời gian cùng cấu hình; training loss không thay thế đánh giá agent.

## Mở rộng sau

Giữ release v1 nguyên trạng. Thêm log TRAIN mới, bổ sung review theo câu hỏi/nguồn rồi tạo output v2 mới:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/prepare_seed_training.py --reviews configs/reviews-v2.json --runs "outputs/diagnosis-2026-11-03/gemma4-12b-stress72-pipeline-v2-01" "outputs/diagnosis-2026-11-03/gemma4-12b-stress100-baseline-v3" "outputs/diagnosis-moi/gemma-run-moi" --output outputs/training/seed72_100_gemma_v2
```

Chưa thêm paraphrase trong v1. Khi thêm, giữ cùng câu gốc và cùng doanh nghiệp trong một partition. Các thư mục preview trước khi chốt Gemma chỉ là artifact làm việc; hai ZIP nêu trên mới là gói bàn giao cho lượt này.
