# Chạy đúng 200 câu benchmark trên Colab

**Rollback 08/11/2026 theo yêu cầu người dùng:** notebook và kit hiện dùng **engine pipeline 6**, thay cho bản interface v7 vừa thử. Tải lại **cả notebook lẫn kit** cùng bản vì notebook khóa SHA256 source bên trong. V7 được giữ ở `outputs/team200-colab/history/27fbb37ba527/`. Mẫu 200 câu, seed, dữ liệu, nhãn và cách chấm giữ nguyên.

Notebook: `notebooks/DataMate_Colab_Team200.ipynb`.
Gói đầu vào: `outputs/team200-colab/datamate-team200-colab-kit.zip`, khoảng
**125,73 MB**. Gói này dùng riêng để bạn chạy agent DataMate và xuất review
có nhãn; hướng dẫn benchmark chung vẫn cho phép mọi thiết kế của đồng đội.

## Chạy

1. Upload notebook `.ipynb` lên Colab.
2. Đưa **datamate-team200-colab-kit.zip** vào `MyDrive/DataMate/`.
   Không cần tải thêm source ZIP, train ZIP hay coordinator ZIP riêng:
   chúng đã được đóng gói/khóa hash bên trong kit.
3. Chọn GPU **G4**, sửa `PARTICIPANT`, `MODEL_KEY` và `BASE` ở ô đầu nếu cần,
   rồi **Run all**. Mặc định `MODEL_KEY="qwen35-9b"`, `MODE="self_review"`,
   quantization none, cùng ngân sách agent pipeline 6 đã khôi phục. Có thể chọn
   `gemma4-12b`, `qwen35-upscaled14b-v2` hoặc `qwen35-4b` trong registry.
4. Giữ `RUN_LIMIT=None` để chạy đủ 200, `AUTO_DISCONNECT=True` để đồng bộ Drive
   rồi ngắt runtime sau khi hai ZIP mới đã vượt kiểm tra.

Preflight phải hiện **pipeline 6/backend 3**. Tên lượt mới mặc định kết thúc
bằng `team200-rollback-pipeline6-Nov8th-01`; không resume vào kết quả v7 hoặc lượt v6 đã hoàn tất. Khi thay
kit/notebook, restart kernel trước khi chạy từ đầu; model đã seal trên Drive
vẫn được dùng lại. Context 16K, output 768, 6 bước, timeout 90 giây và
self_review khớp cấu hình notebook 100 câu ở chat đã đối chiếu.

Model đã seal ở `MyDrive/DataMate/models/<MODEL_KEY>/` được dùng lại, chép vào
ổ tạm `/content/models` trước khi nạp GPU. Chưa có model thì bước chuẩn bị
dùng stage-model tải checkpoint/revision đã pin, trước inference. Dependency
giữ bản PyTorch CUDA do Colab cấp. Notebook không chứa trọng số model.

Mẫu được đọc nguyên từ participant v2: **200 câu, seed 4000930841, 28 doanh
nghiệp**, hash câu hỏi `7ae2079219d0a79ca1dd66952610ceb091e36c0ecf65359b11dac8263c67b782`.
Không bốc lại, không sử dụng diagnostic 100 câu cũ. Dataset inference không
có answers.jsonl; nhãn chỉ được mở ở bước review sau vòng generation.

## Kết quả trên Drive

Thư mục mặc định: `MyDrive/DataMate/runs/team200/<RUN_NAME>/`.

| File | Dùng làm gì |
|---|---|
| `group_submission.zip` | **Gửi nhóm**. Chỉ có `submission.jsonl` và `run_metadata.json` sáu trường theo protocol v2. |
| `diagnostic_bundle.zip` | Giữ để phân tích; có submission/metadata, review, log, manifest/runtime và lịch sử thời gian. Có nhãn chuẩn. |
| `review.csv`, `review.jsonl`, `review.md` | Đáp án dự đoán, nhãn, đúng/sai/abstain, điểm, cờ lỗi và thống kê từng câu. CSV có UTF-8 BOM. |
| `logs/<id>.json`, `logs/<id>.md` | Prompt ban đầu, raw output model, SQL/tool, kết quả hoặc lỗi, đáp án cuối, thời gian và model calls. |
| `summary.json`, `runtime.json` | PA/timing/coverage, phần cứng, dependency và giao thức generation thực tế. |
| `run_manifest.json`, `selected_questions.jsonl` | Code/config/model/hash và đúng thứ tự mẫu đã chạy. |
| `session_history.json`, `report_receipt.json` | Lịch sử phiên, thời gian cộng dồn và biên nhận hai ZIP mới trước khi tự ngắt. |

Submission chỉ có id/type/answer/reason theo schema cuộc thi; không thêm
thời gian hoặc log vào từng dòng. Metadata chỉ có participant, models, gpu,
load_seconds, inference_seconds, compute_units_used. Unit để null khi chưa
đo được; không giả định 10 unit/giờ là số unit thực tế đã dùng.

PA review có mẫu số **200**, thiếu/abstain = 0, kể cả nếu batch dừng sớm.
Log/review đầy đủ là tiện ích riêng của lượt chạy này, không đổi quy định
hai file mà đồng đội cần nộp. Đây là benchmark nội bộ TRAIN, không phải
holdout độc lập hoặc điểm chính thức.

## Resume và tự ngắt

- Mỗi câu được lưu ngay. Cùng kernel/model đang load thì chạy lại ô batch
  cuối. Kernel/runtime mới thì Run all với nguyên RUN_NAME và cấu hình.
- Code, model manifest, mẫu, config hoặc dependency/GPU khác sẽ chặn resume;
  dùng RUN_NAME mới nếu chủ động đổi hệ thống. Không ghép/chọn đáp án giữa
  các phiên bản và không trả lời lại câu đã có log.
- Thời gian load cộng một lần cho mỗi lượt nạp model; chạy lại batch trong
  cùng kernel không cộng load lần nữa. Inference cộng các phiên batch.
- Nếu runtime bị hủy đột ngột khiến không biết đủ thời gian phiên trước,
  inference_seconds giữ null và summary ghi rõ timing chưa đầy đủ.
- Lỗi/ngắt batch vẫn tạo báo cáo partial và hai ZIP khi có thể. Nếu kit đáp
  án lỗi, vẫn lưu submission/log và review chưa chấm, ghi review_error.
- Chỉ tự ngắt khi có biên nhận đúng lượt vừa chạy, hai ZIP khớp SHA256/CRC
  và Drive flush thành công. Nếu chưa xác nhận lưu được kết quả thì giữ
  runtime để kiểm tra. Cài thư viện/load lỗi trước batch không tự ngắt.
- Không dùng lại ZIP cũ để xác nhận một lượt mới bị lỗi. Việc ngắt dùng API
  `google.colab.runtime.unassign()` sau `drive.flush_and_unmount()`; cách gọi
  được đối chiếu với [mã Colab chính thức](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py).

## Kiểm chứng trong workspace

- **225 kiểm thử toàn bộ workspace qua** sau rollback, gồm engine, runner 200 câu, resume/interrupt, nhãn chỉ đọc sau generation, schema/matching và điều kiện đồng bộ Drive trước unassign qua API giả lập.
- Notebook có bốn code cell và preflight pipeline 6/backend 3. Kit/nested ZIP được kiểm tra CRC, hash và nội dung source; tất cả inference source khớp bản pipeline 6 đã sao lưu, không có module agent/parameters của v7.
- Hai ZIP benchmark participant/coordinator giữ nguyên hash. Bộ inference không đưa nhãn vào agent; coordinator chỉ mở sau generation để review.
- Unit test dùng agent/timing giả lập. **Chưa có lượt GPU, PA, thời gian/câu hoặc compute unit mới sau rollback.**

Mã: `tools/infer_team200.py`, `tools/create_team200_colab.py`.
Kiểm chứng đóng gói: `outputs/team200-colab/verification.json` và
`outputs/team200-colab/build_manifest.json`. Notebook 100 câu cũng được đồng bộ về pipeline 6; các gói SFT Gemma frozen giữ nguyên.
