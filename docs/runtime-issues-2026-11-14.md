# Nhật ký lỗi Colab và kiểm tra trước khi nộp Docker

Ngày ghi nhận: **14/11/2026**. Nguồn: traceback/log Colab, `nvidia-smi`, `process_timing.json` do chủ dự án gửi trong cuộc trao đổi; đối chiếu source hiện tại. Đây là lỗi quan sát thực tế và quyết định kỹ thuật của đội, không bổ sung quy định BTC. Các bản sửa mới được kiểm bằng fixture CPU; chưa có kết quả chạy hoàn chỉnh image thi trên GPU.

## 1. `pip check` thất bại trong môi trường Colab dùng chung

Quan sát: runtime có vLLM0.33.0, Torch2.14.0, Transformers5.20.2, NumPy2.4.6, PEFT0.21.2. Các xung đột báo về IPython/Jedi, google-colab/requests, RAPIDS/Numba, Gradio/Pillow, google-adk/OpenTelemetry, numba-cuda/cuda-core. Đây là ảnh chụp môi trường đã báo lỗi, chưa phải dependency lock được xác nhận cho image cuối.

Colab yêu cầu requests2.32.4, trong khi vLLM0.33 yêu cầu requests>=2.33; Gradio yêu cầu Pillow<13, trong khi vLLM yêu cầu Pillow>=13. Không thể thỏa cả hai bộ ràng buộc đó trong một môi trường. Không hạ thư viện của vLLM theo package Colab và không bỏ `pip check`.

Đã sửa: notebook tạo venv riêng; merge/inference/benchmark đều dùng `RUNTIME_PYTHON`, lưu `pip-check.txt` và `runtime-freeze.txt`. Docker cần môi trường runtime tối thiểu, wheelhouse Linux/Python/CUDA phù hợp; không sao chép venv/lock Windows hoặc toàn bộ môi trường Colab vào image. Installer của image đã có `pip check`.

## 2. vLLM tìm `preprocessor_config.json` dù chỉ chạy văn bản

Quan sát: baseline process hoàn tất sau3362,8s; vLLM dừng sau94,9s, chưa có model call. Traceback đi qua `MultiModalBudget` → Qwen image processor và lỗi thiếu `preprocessor_config.json`. A10040GB không có tiến trình chiếm VRAM sau khi worker thoát. Log này không phải lỗi OOM.

Nguyên nhân: config ngoài của Qwen khai báo kiến trúc multimodal, nhưng adapter DataMate chỉ gửi token văn bản. Engine mặc định vẫn khởi tạo processor ảnh/video.

Đã sửa: `models/vllm_backend.py` truyền `language_model_only=True`. Source cũ đã đo baseline có thể dùng `vllm.extra_engine_args.language_model_only=true` thông qua tool recovery, giữ nguyên hash source/weights của baseline. Không lấy processor từ revision khác hoặc viết file giả vào artifact đã seal.

Kiểm chứng GPU mới nhất: lượt recovery đã in **running in text-only mode**, vượt qua lỗi này, nạp weights rồi gặp lỗi Ninja ở mục3. Chưa suy ra engine đã sẵn sàng hoặc có kết quả inference.

Docker: giữ chế độ text-only trong cấu hình cuối, kiểm startup thật với đúng artifact đóng gói. Nếu sau này hỗ trợ ảnh/video, phải thiết kế lại port và đóng gói processor tương ứng; không tự bật lại multimodal chỉ để qua startup.

Nguồn kỹ thuật: [vLLM0.33 MultiModalConfig](https://docs.vllm.ai/en/v0.33.0/api/vllm/config/multimodal/).

## 3. `FileNotFoundError: ... 'ninja'` khi warmup sampler

Quan sát: recovery `recovery-20261114-113219-781222` trên A10040GB đã nạp model, dùng khoảng20GiB VRAM rồi dừng trong FlashInfer JIT: `top_k_top_p_sampling_from_logits` → `build_and_load` → `run_ninja`. Engine chưa trả lời câu nào. Các cảnh báo cast BF16→FP16 và deprecated API không phải exception cuối làm worker thoát.

Nguyên nhân đã xác nhận: subprocess không tìm thấy executable `ninja`. Chưa có output chứng minh package Ninja hoàn toàn chưa cài. Lỗ hổng source đã xác định: gọi `/venv/bin/python` không tự thêm `/venv/bin` vào `PATH`, nên cài package trong venv vẫn có thể không tìm được executable đi kèm.

Đã sửa source:

- Extra `vllm` khai báo `ninja>=1.11,<2` để wheelhouse/image có build tool khi dùng đường JIT.
- Notebook thêm `runtime_dir/bin` vào `RUNTIME_ENV['PATH']`, đặt `VIRTUAL_ENV`; recovery truyền thư mục executable của chính Python đang chạy vào `PATH` của worker. Không resolve symlink Python về hệ thống rồi làm mất đường dẫn venv.
- Lượt Colab cũ có thể sửa `RUNTIME_ENV`, kiểm `shutil.which('ninja', path=...)`, chỉ cài Ninja bằng Python của venv nếu còn thiếu. Không cần đổi Torch/vLLM hoặc chạy lại baseline chỉ vì bổ sung executable build tool. Ghi freeze mới riêng, giữ freeze trước đó.

Docker phải có một đường chạy kernel hoàn chỉnh, đã kiểm offline: JIT cần Ninja, CUDA compiler/toolkit và C++ compiler phù hợp; hoặc có bộ kernel prebuilt/cache đúng phiên bản và GPU mục tiêu. Có package FlashInfer không đồng nghĩa kernel đã sẵn sàng. Không cài một wheel cubin tùy ý để khớp tên; không mặc định cache tạo trên A100 dùng được cho GPU BTC. Không coi dòng `CUDA Version` trong `nvidia-smi` là bằng chứng đã có `nvcc` trong image.

Trước load model, kiểm `ninja --version`, đường dẫn/phiên bản `nvcc` và C++ compiler khi dùng JIT. Sau đó vẫn phải chạy engine thật trong container với mạng tắt và cache ban đầu được ghi rõ. Image cần chuẩn bị tài nguyên kernel trước khi nộp; không tải kernel lúc inference BTC. Không dùng `enforce_eager`/tắt tối ưu tùy tiện để che thiếu môi trường rồi gọi kết quả đó là cấu hình vLLM đã tối ưu.

Nguồn kỹ thuật: [kiểm tra Ninja/nvcc của vLLM0.33](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/utils/flashinfer.py), [cài đặt FlashInfer](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/installation.rst).

## 4. Torch OOM trên L4 khoảng22,5GiB

Quan sát: cần cấp thêm khoảng204–206MiB trong khi GPU chỉ còn29–33MiB. Log có `[35/3000] LỖI XỬ LÝ ... | abstain`; OOM đã ảnh hưởng kết quả câu hỏi. Snapshot VRAM gần đầy một mình không chứng minh OOM, nhưng các dòng allocation failed này đã xác nhận.

Xử lý: chuyển phần đối chứng sang A100, chạy lại tập đã chọn trong thư mục mới; giữ dtype/checkpoint/budget thống nhất giữa các cấu hình. A100 chạy được không chứng minh fit GPU BTC. Image cuối phải đo peak VRAM gồm generator, KV cache, workspace và embedding/reranker khi bật RAG. Nếu đổi quantization/context/budget thì cần baseline tương ứng, không gộp số đo khác điều kiện.

## 5. Baseline hoàn tất nhưng có measurement warnings

Recovery mới báo baseline hoàn tất nhưng `valid_complete_run=false`. Chưa nhận đầy đủ `baseline/benchmark_metrics.json` để xác định cảnh báo cụ thể. Không tự coi baseline sạch lỗi hoặc kết luận phương án khác tốt hơn.

Giữ file gốc và đọc `measurement_warnings`, `oom_calls`, `timeout_calls`, `possible_time_stop_calls`, `generation_error_calls`, `cpu_or_disk_offload_detected`, `error`, cùng log từng câu. Có thể tiếp tục chạy để chẩn đoán, nhưng phép so hợp lệ phải qua các gate trong `comparisons.json`; không sửa flag/log để ép pass. Không có gold thì chưa có bằng chứng chất lượng tốt hơn.

## 6. Thiếu file ID và lượt chạy quá lớn

Đã gặp `FileNotFoundError` khi `IDS_FILE` chưa tồn tại; sau đó lựa chọn đủ3000 câu khiến cả ba lượt tổng cộng9000 lần xử lý câu hỏi. Trước chuẩn bị model, kiểm ID tồn tại, unique và thuộc split. Thử một tập nhỏ cố định trước; mọi cấu hình đối chứng dùng cùng tập đó.

Không sửa file ID trong khi vòng so sánh đang chạy. Recovery giữ bộ ID/checkpoint/source/dtype của baseline, tạo thư mục mới thay vì ghi đè. Docker cuối phải đọc input BTC; không đóng cứng tập benchmark100/3000 ID vào đường chạy submission.

## 7. Ngắt cell sau khi vLLM đã lưu100/100 câu

Log chủ dự án gửi tiếp ngày14/11: recovery `recovery-20261114-114601-004845` đã qua lỗi Ninja, compile/JIT/capture và báo model sẵn sàng sau391,1s. Log có đủ `[100/100] ĐÃ LƯU`; sau đó lúc engine teardown có `KeyboardInterrupt` ở vòng đọc stdout của cell. Các dòng SIGTERM trong phần shutdown tự nhiên của vLLM không tự chứng minh lỗi/OOM. Chỉ bấm Enter thường không dừng cell; traceback xác nhận kernel đã nhận một thao tác ngắt, không xác định chính xác phím/nút người dùng đã bấm.

Kết luận giới hạn: engine đã khởi tạo và thực hiện inference100 câu theo log, xác nhận hai sửa text-only/Ninja đã đi qua trên A100 của lượt này. Chưa nhận file kết quả để kiểm đủ ID/giá trị, chất lượng, hay trạng thái hoàn tất benchmark. Không ghi là image Docker đã kiểm GPU thành công.

`submission.jsonl` được flush/fsync từng câu trước dòng ĐÃ LƯU, nên giữ lại thư mục gốc và kiểm số dòng/ID. Nhưng `run_summary.json` và `benchmark_metrics.json` được ghi sau `agent.close()`; ngắt lúc teardown có thể làm thiếu metadata dù đã có đủ đáp án. Không viết metrics giả hoặc đánh dấu complete để ép gate pass. Nếu thiếu metadata, giữ đáp án cho đánh giá chất lượng, ghi lượt timing bị ngắt; muốn số đo cold/end-to-end hợp lệ thì cần lượt đo hoàn chỉnh mới.

Có thể chạy riêng RAG bằng `continue_system_comparison.py --arms rag`, dùng baseline/source/IDs cũ, giữ vLLM ở recovery trước. Kiểm tiến trình GPU cũ đã thoát trước khi nạp model mới. Báo cáo recovery RAG chỉ ghép baseline–RAG; phép so baseline–vLLM riêng cần đủ metadata của lượt vLLM cũ. Không chạy lại mặc định cả hai arm nếu mục tiêu chỉ là RAG. Khi đã có100/100, không resume một benchmark mới vào thư mục đó để ghi đè số đo bằng lượt không còn việc.

Chủ dự án xác nhận tiếp bằng ô kiểm tra và `nvidia-smi` lúc12:22:12 ngày14/11: `submission.jsonl` đọc được100 bản ghi với100 ID riêng biệt; `benchmark_metrics.json` chưa tồn tại; A100 hiển thị0MiB sử dụng và không còn process. Có thể bắt đầu lượt RAG riêng. Chưa kiểm đối chiếu chính tập ID hoặc chấm gold từ các file thực tế; không ghi lượt vLLM này là benchmark timing hoàn tất.

Đã lưu [thời gian tạm tính từ output100 câu](vllm-100-provisional-timing-2026-11-14.md): startup391,1s; tổng elapsed từng câu1476,8s; tổng hai phần1867,9s, chưa tính khoảng trống/I/O và shutdown. Bản sao log/JSONL/tổng hợp nằm trong outputs, không qua Git. Không dùng số tạm để thay metrics hoàn tất hoặc khẳng định speedup/chất lượng.

## 8. Cửa kiểm tra release Docker

1. PDF BTC đã nhận/đọc14/11: recommended vLLM0.10.1.1/cu128, alternatives phải hỗ trợ5090/SM120. Ghi digest/commit/config/model revision/hash/license. vLLM0.33 ứng viên đã inspect CUDA13.0.2, cần xác nhận driver BTC hoặc stack phù hợp; không coi recommended cũ là hỗ trợ Qwen3.5.
2. Dùng full model merged đúng lineage fine-tune và tokenizer; thêm embedding/reranker nếu bật RAG. Không thay bằng base hoặc adapter chưa merge. Giữ tổng tham số theo giới hạn cuộc thi.
3. Chuẩn bị wheelhouse/runtime đúng môi trường Linux mục tiêu, gồm Ninja khi dùng JIT; `pip check` phải đạt trong image. Kiểm executable bằng chính environment chạy `predict.sh`, không chỉ environment của shell build.
4. Chạy startup + inference mẫu trong **image cuối**, với GPU, `--network none`, input `/data` chỉ đọc, output `/result`, scratch/cache mới theo chính sách release. Kiểm text-only, kernel và đủ câu/type/log; kiểm lỗi cấp phát và tiến trình còn giữ GPU sau thoát.
5. Rehearsal workload đủ đại diện: startup/import, hash/load weights, index RAG, JIT/compile và inference đều nằm trong ngân sách. Mẫu nhỏ thành công không chứng minh hoàn tất toàn private test trong6giờ. Không loại chi phí startup chỉ vì thư viện/weights đã nằm trong image.
6. Chỉ ghi trạng thái đã kiểm chứng sau khi thực sự chạy. Lưu log, peak VRAM, timing, freeze, source/model/config manifest, điều kiện cache và checksum image. Đổi code/dependency/model sau rehearsal phải kiểm image mới.

Lệnh staging/build/rehearsal/export: xem [Docker](docker.md). Recovery Colab: xem [hướng dẫn đối chứng](colab-finetuned-compare.md).

## 9. Đối chiếu kết quả100 câu sau khi nhận đủ folder

Nguồn chủ dự án cung cấp: `E:/Bảo Lập Trình AI/ZALO_AI_2026/outputs/comparison-100-20261114-101046`, gồm baseline, recovery vLLM114601 và recovery RAG122413. Đã đọc đủ300 trace,100 timing mỗi arm, GPU samples và identity;151 file dữ liệu local khớp hash. Gold train đúng hash được dùng để chấm sau inference. [Báo cáo và giới hạn](comparison-100-review-2026-11-14.md).

- Baseline66 đúng/9 sai/25 abstain, PA57%; vLLM67/8/25, PA59%; RAG71/9/20, PA62%.80 ID thực sự có trong train.jsonl,18 trong validation đã được xem trước. Không coi đây là benchmark unseen/private.
- Đã xác định cảnh báo baseline:2 generation có `time_or_other`, RAG1; thời gian gần max_time và không EOS. Không phải lỗi pip/OOM. vLLM có1 call chạm token budget. Comparator strict vẫn fail các gate deadline/valid run; không sửa flags để ép pass.
- Raw timing đủ100 câu cho phép so cùng mốc bắt đầu câu1→kết thúc câu100: Torch3286,68s, vLLM1479,86s, RAG2898,20s. vLLM khoảng2,22 lần nhanh hơn trong phase này; thiếu metrics/shutdown completion vẫn phải được ghi rõ.
- Sai số tiền lớn được kiểm chỉ đọc: SUM trên cột TEXT có dấu phân cách nghìn. SQL cần kiểu số/parse có căn cứ. Lỗi này tồn tại ở logic dữ liệu/SQL dùng chung, không phải GPU tính tiền sai.
- RAG retrieval100/100 ok và đúng scope, nhưng index có26 chunk_id không duy nhất về nội dung;28 context có ID trùng. `sections()` thiếu table/sheet/hash text trong danh tính chunk. Cần sửa và rebuild cache trước khi tin provenance/dedup; chưa chứng minh collision là nguyên nhân của mọi đáp án sai.
- Peak GPU lấy mẫu: baseline31,69GiB, vLLM30,58GiB, RAG34,51GiB trên A10040GB. Phải kiểm lại image theo VRAM/GPU BTC thực tế; tổng tham số dưới15B không bảo đảm fit VRAM.

Phân tích lần này không thay đổi runtime/Agent/RAG, không sửa các output gốc hoặc tạo metrics giả. File phân tích/gold ghép/timing từng câu nằm trong outputs và không qua Git.

## 10. Chuẩn bị submission sau khi nhận Docker PDF và full model

Nguồn: PDF chính thức do chủ dự án chỉ đường dẫn, nhận14/11; inspect base image Docker Registry; full model download từ Drive. Nội dung xác nhận:2×5090/32GB mỗi card, RAM512GB,6giờ gồm load, `/code/predict.sh` không đối số, shm16g, TAR chưa nén<=100GB, nộp TAR.GZ+MD5 của file nén và training assets/offline dependencies bắt buộc.

- Template cũ `/app/predict.sh` đã đổi `/code/predict.sh`, reset ENTRYPOINT và CMD bash; không trông vào ENTRYPOINT vì BTC override.
- Log cũ thiếu answer/elapsed_ms/thinking canonical và còn pretty-print nhiều dòng. Đã bổ sung fields từ kết quả/trace thực, giữ raw thinking, SQL/parameters/output; Markdown chuyển `debug/`, lỗi cấp run ngoài `logs/`. Không biến action.reason thành thinking.
- Trùng chunk_id RAG được sửa bằng metadata/table/sheet và hash nội dung; index source identity thay đổi nên cache cũ không tái dùng. Các đáp án benchmark cũ giữ nguyên; chưa có kết quả GPU mới từ sửa provenance.
- Model download có4shard bị hậu tố `-005/-001/-003/-002`, làm thiếu tên được index/manifest tham chiếu. Đã kiểm size+SHA256 từng shard rồi khôi phục tên canonical; sau đó verify16file,19.326.417.873bytes, manifest732f292b... khớp bản đối chứng. Không reseal để che lỗi hoặc sửa trọng số.
- Hai RAG checkpoint đã tải/seal đúng pinned revision và hash weights. Training data của cả hai stage được đối chiếu toàn bộ file hash với report hoàn tất; source/config/log/parent/final adapter giữ frozen, wheelhouse training Linux còn phải chuẩn bị trước release.
- Inspect vLLM0.33 image chính thức: CUDA13.0.2, SM120 có trong TORCH_CUDA_ARCH_LIST; **không đồng nghĩa tương thích driver R570**. cu128/cu129 tags0.33 đã kiểm không tồn tại. Không mặc định forward compatibility cho GeForce hoặc cài NCCL-cu12 workaround vào cu13. Driver/Blackwell/full9B+RAG cần kiểm trên image thật và Test Solution.

Chủ dự án chọn hoàn tất phần chuẩn bị trước, thuê GPU Vast sau (ngân sách khoảng1giờ). Giữ rõ phần chưa xác minh, không ghi full image/GPU đã đạt dựa trên CPU smoke. Hướng dẫn: [Docker](docker.md), [Vast](run-vast.md).

## 11. Sửa sau review độc lập trước bàn giao

Đã tái hiện bằng CPU rồi sửa bốn vấn đề: thiếu merged path vẫn đi vào helper tải base trước khi evidence gate chặn; thinking đảo outer/repair vì trace append lồng nhau; log bỏ helper/SQL nội bộ; README image trỏ docs không có trong image.

Release preparation nay kiểm đủ model local/hash/merged lineage trước download/install. Agent có execution ledger riêng mỗi câu/worker, ghi actual function/SQL/arguments/result/error tại lúc thực thi. Các model call có mốc thứ tự độc lập; serializer giữ nguyên thinking theo thứ tự gọi. Stage helper sinh README riêng liệt kê model/revision/license/ngày, dữ liệu và lệnh chạy/tái lập trong image. Không sửa đáp án benchmark hoặc số đo cũ.

Kiểm sau sửa:505CPU tests passed,5skipped; Docker CPU fixture3câu offline, scoring entrypoint, dataRO/rootfsRO và gzip export/checksum thật qua. Artifact demo có chữ CPU-DEMO-NOT-SUBMISSION, không được nộp BTC. Các kiểm này không chứng nhận full9B+RAG, compiler/kernel SM120, driver/VRAM/deadline hoặc training wheelhouse Linux.
