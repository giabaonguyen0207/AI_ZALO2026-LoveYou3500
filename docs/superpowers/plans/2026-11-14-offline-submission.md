# Offline DataMate Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Đồng bộ source về ổ E và chuẩn bị image linux/amd64 chứa Qwen9B đã merge + vLLM + dense/rerank RAG, đủ training evidence, đúng guideline BTC, có kiểm chứng offline và MD5.

**Architecture:** Giữ modular monolith Adapter + Strategy đã được duyệt; dùng cùng application.run_batch cho CLI/Docker. Chỉ sửa các vấn đề packaging/log/provenance và loại bỏ đường notebook thử nghiệm đã nghỉ hỗ trợ; không thay logic SQL/nghiệp vụ của benchmark trong đợt release.

**Tech Stack:** Python, Transformers baseline, vLLM0.33, Vietnamese Embedding/Reranker, Linux Docker/BuildKit, NVIDIA CUDA.

**Spec:** Yêu cầu thực hiện trực tiếp của chủ dự án ngày14/11/2026; PDF `E:/Bảo Lập Trình AI/zac2026_DataMate_docker_guideline_final.pdf`, đã trích toàn bộ và lưu hash trong outputs/runs/Bảo/docker-release-20261114/organizer-guideline-source.json.

## Global Constraints

- `/code/predict.sh` chạy không đối số; `/data` read-only; `/result/submission.jsonl` và một log một dòng cho mỗi câu.
- Log bắt buộc: id, answer, elapsed_ms integer >=0, thinking nguyên văn hoặc null; code/inputs/output phải phản ánh thực thi thật.
- Image linux/amd64, không tải/gọi mạng khi inference; toàn bộ inference assets bên trong. Tổng model params <=15B, CUDA>=12.8/Blackwell-compatible. Giữ điều kiện phát hành model chặt hơn01/10 mà chủ dự án đã yêu cầu; PDF ghi26/10, không âm thầm nới điều kiện.
- Training code, dữ liệu train/validation, logs, configs, checkpoint xác minh và manifest phải nằm trong image; training-only base được khai báo link/revision/hash khi đủ điều kiện. Không xóa training source cần tái lập.
- Hai GPU5090, 32GB/GPU, shm16g; <=6h gồm startup; .tar không nén <=100GB; nộp .tar.gz và MD5 của chính file nén, kiểm tải lại.
- Không ghi đè18 thay đổi training local chưa commit ở E; không push hoặc Submit Test thay người dùng.
- Không coi tiny GPU test là chứng minh full9B+RAG/5090 hoặc bảo đảm điểm BTC. Bản model merged đang được người dùng tải; thiếu artifact/GPU thì giữ trạng thái chưa xác minh.

## Review Focus

- Scoring override `--entrypoint bash ... /code/predict.sh` và container root read-only: entrypoint/path/caches phải đúng.
- Nhiều model calls có reasoning: không tóm tắt/cắt reasoning, không mất SQL/parameters/output.
- Models fine-tuned và training data/dependencies: không thay merged bằng base, không bỏ dữ liệu bởi dockerignore hoặc stage helper.
- Export đứt giữa chừng, tag nhiều kiến trúc hoặc vượt100GB: không sinh receipt/MD5 như artifact hợp lệ.
- Dataset/tokenizer/kernel caches: network disabled, chưa có warm cache, Ninja/nvcc/Blackwell phải kiểm trên image thật.

## Task1: Source handoff và thu gọn an toàn

Files: README.md, docs/team-workflow.md, các notebook/tool cũ đã xác nhận không còn dùng; tests của artifact đã nghỉ hỗ trợ; giữ src runtime và training code thực sự dùng.

- [x] Đọc guideline và ghi baseline hiện trạng: CPU suite474 passed,5 skipped,4 failed (notebook đã nghỉ2, UTF-8 test1, thiếu psutil dev1).
- [x] Ghi inventory/reference trước khi retire; lưu hash/bản sao local của9file bỏ khỏi tree trong outputs/runs/Bảo/docker-release-20261114/retired-source. Không xóa archive/dataset/weights/frozen artifacts.
- [x] CPU505passed/5skipped; commit và fast-forward E với hash-before/after:18file dirty được giữ nguyên nội dung,5delete cũ đã trùng thay đổi upstream, còn13WIP training nguyên trạng; không push.

## Task2: Log và cấu hình vLLM+RAG theo guideline

Files: src/datamate/competition_log.py (new), src/datamate/application.py, src/datamate/retrieval/core.py, configs/submission.json, tests/test_competition_log.py (new), tests/test_rag_integration.py.

- [x] RED/GREEN: log canonical, elapsed_ms, raw thinking/call order, SQL/parameters/output; UTF-8, một dòng. Markdown/debug và lỗi cấp run chuyển ngoài logs/.
- [x] Helper serialize record giữ trace cũ; không đổi câu trả lời.
- [x] RED/GREEN: schema khác sheet không cùng chunk_id; metadata/text hash đi vào source identity.
- [x] Config vllm+dense_rerank, TP1 GPU0/RAG GPU1; ghi profile ứng viên chưa kiểm5090, model merged đúng pin.

## Task3: Training evidence, build context và image paths

Files: tools/package_solution.py, tools/training_evidence.py (new), docker/Dockerfile.competition, docker/install_runtime.py, predict.sh, tests/test_runtime_artifacts.py và test staging/log mới.

- [x] RED/GREEN: context /code, toàn inventory, từ chối release thiếu training evidence, Linux wheels/base digest; kiểm dữ liệu không bị dockerignore loại.
- [x] Stage allowlist, hash/path, dockerignore riêng; generic evidence assembler và training-only base declarations; train.sh/README tái lập cả hai stage từ frozen source, private venv offline.
- [ ] Dùng base image hỗ trợ Qwen3.5/vLLM0.33 và Blackwell (PDF cho phép alternative); pin digest sau inspect, không hạ về vLLM0.10 chỉ vì recommended example.
- [x] ENTRYPOINT [], CMD bash /code/predict.sh; P2P disable, spawn; installer offline và BuildKit wheel bind mount. Training wheelhouse bắt buộc ở release, còn cần tạo trên Linux.

## Task4: Verify, export và bàn giao

Files: tools/export_solution.py, tools/verify_submission.py (new), docs/docker.md, docs/run-vast.md (new), docs/competition-context.md, release verification ledger.

- [x] RED/GREEN: gzip export/MD5/SHA256, đọc toàn TAR/CRC, <=100GB, một linux/amd64 image/tag, từ chối hỏng/partial; không ghi GPU verified.
- [x] Full CPU505passed/5skipped + Docker CPU fixture offline/scoring entrypoint, gzip export thật. GPU visibility4060 có; không chạy tiny/full GPU, chủ dự án chọn thuê sau.
- [x] Một fresh-context review,4Important RED→GREEN và full suite sau sửa; minor lịch sử ghi ledger.
- [x] Hướng dẫn Vast Ubuntu VM/SSH/disk/GPU và hai lượt offline; lưu yêu cầu mới của chủ dự án: hoàn tất chuẩn bị trước, thuê khoảng1giờ sau, không tự thuê/tính phí.
- [ ] Chỉ xuất image nộp sau đủ models/training/wheels, rehearsal offline và kiểm giới hạn. Nếu thiếu GPU hoặc merged model, hoàn thành source/context/runner và ghi đúng phần còn thiếu; không phát hành image dev như image thi.

## Ledger

- 14/11: User đã yêu cầu thực hiện trực tiếp, chọn vLLM+RAG và đồng bộ source về E. Không mở lại lựa chọn kiến trúc đã duyệt; dùng native implementation và review cuối.
- Docker Desktop Linux engine hoạt động sau khi dùng quyền truy cập đúng; `--gpus all` nhìn thấy RTX4060 Laptop8188MiB. Chưa có full/tiny GPU inference.
- vLLM0.33 amd64 manifest verified: `sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3`; chưa pull/build/rehearsal.
- PDF mới xác nhận /code (template cũ /app chưa đúng), fields log bắt buộc, training assets bắt buộc,100GB tar và gzip MD5. User chưa có máy Linux GPU; đang tải model merged về local.
- Model merged đã kiểm16file/19.326.417.873bytes, manifest732f292b...;4shard sai tên tải xuống đã kiểm hash rồi khôi phục tên. RAG weights đúng pinned hashes đã tải/seal ở E/models.
- Training evidence546file: frozen source/data khớp report cả hai stage, parent/final adapters, raw TRAIN, configs/log/freeze,10public-base file declarations; offline training wheelhouse chưa tạo.
- CPU toàn repo:502passed,5skipped (41,39s), sau sửa log/chunk/staging/export. Docker CPU smoke3câu thật qua offline/dataRO/rootfsRO/scoring entrypoint; không phải full-model GPU result.
- User steering: làm xong phần chuẩn bị/local trước; GPU/Vast sẽ do chủ dự án thuê sau. Vì vậy chưa build/certify image cuối, không export image dev như image nộp. CUDA13/R580 vs guidelineR570 là gate chưa xác minh, đã ghi trong Docker/Vast docs.
- Final review: một reviewer độc lập, range7d8e6f8..9fe4a52. Không có Critical;4Important đã tái hiện RED và sửa trong một pass. Bị ngắt quota giữa lượt, tiếp tục cùng reviewer sau khi user yêu cầu; không mở lượt review thứ hai.
- Final: fixed helper/SQL receipts — test_execution_audit.py RED(module missing)→GREEN, thêm kiểm thread isolation; ContextVar ledger ghi actual function/arguments/output/error.
- Final: fixed nested reasoning order — test_targeted_patch_keeps_valid_fields_and_is_cached RED repair/outer đảo thứ tự→GREEN outer/repair; call timestamp ở cả ba điểm backend.generate.
- Final: fixed missing-merged download fallback — test_release_preparation_never_downloads_when_merged_input_is_missing RED đã gọi preparation→GREEN fail trước download/install với mọi model local/hash/evidence.
- Final: fixed image README — test_release_includes_sealed_training_data_and_offline_training_wheels RED thiếu revision/readme standalone→GREEN; generate README inventory/license/date/paths/counts/exact commands, không link docs ngoài image.
- Final CPU suite:505passed,5skipped (42,62s). Docker CPU image rebuilt; fixture3câu offline/dataRO/rootfsRO/scoring entrypoint qua; export Docker-save gzip thật qua (CPU-DEMO-NOT-SUBMISSION.tar.gz), không upload và không gọi đây là image nộp.
- Final: Ruling: CUDA13/R580, Blackwell/JIT, cold cache, VRAM và6giờ — để GPU/Test Solution kiểm theo chỉ đạo thuê sau; chưa xác nhận, nếu sai có thể làm image không chạy hoặc quá hạn.
- Final: Ruling: entrypoint/data/output/cache — chấp nhận kiểm CPU cho hợp đồng I/O; cache/kernel GPU vẫn chưa được chứng nhận, nếu sai cần sửa image và chạy lại.
- Final: Ruling: export — kiểm cấu trúc/TAR/gzip/hash đã có CPU thật; không suy ra inference hay link Drive hợp lệ, nếu sai transfer/link có thể khiến BTC không nhận image.
- Final: Ruling: training evidence/hash/lineage/dockerignore — giữ frozen assets và kiểm hash; Linux wheel resolve/install và retraining chưa chạy, nếu sai release bị chặn hoặc BTC không tái lập được.
- Final: Ruling: SQL quality/training overlap — không đổi logic đáp án trong release; báo cáo100câu là development, nếu coi là private accuracy sẽ kết luận sai chất lượng.
- Final: Ruling: retired notebook references trong release builders — giữ vì script sinh lại artifact, không phải runtime entrypoint; dùng sai script có thể mở workflow cũ, README hiện hành là điểm bắt đầu.
- Final: minor (deferred): docs/sql-validation100-*.md và docs/sql100-mtp-export-fix.md còn hướng dẫn notebook đã retire; chưa dọn tiếp các tài liệu lịch sử, dùng docs/docker.md/colab-finetuned-compare.md hiện hành.
- Handoff E: e0970b5 trên codex/team-gpu-layout, mọi byte của18file dirty trước merge được giữ;5notebook vốn đã xóa trùng upstream, còn13file WIP. Training evidence546file/861.481.479bytes copy về E và verify manifest93ccb49c... khớp. Chưa push.
- E suite lần đầu:512passed/6skipped/1failed; failure của bài kiểm empty venv do PYTHONPATH test runner làm lộ editable egg-info vào child. Probe bỏ PYTHONPATH/PYTHONHOME đạt pip check; sửa test để đo đúng môi trường trống, không cài thư viện hoặc sửa training WIP để ép pass.
