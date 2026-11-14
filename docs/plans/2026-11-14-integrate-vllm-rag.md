# Tích hợp vLLM và RAG — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge hai nhánh Hải/Thịnh vào `codex/team-gpu-layout`, sửa lỗi và cung cấp cùng một đường inference cho baseline Torch, vLLM và Torch + RAG.

**Architecture:** Modular monolith, Adapter + Strategy theo hai plan đã duyệt. Backend `transformers|vllm|demo` và retrieval `none|dense|dense_rerank` độc lập; giữ baseline prompt/SQL khi RAG tắt. Một application sở hữu engine/retriever, index chuẩn bị trong startup budget; checkpoint và dependency dùng full inventory.

**Tech Stack:** Python, Transformers/PyTorch, async vLLM, NumPy local index, pytest; inference thật chỉ trên GPU host.

**Spec:** Yêu cầu chủ dự án 14/11/2026; [plan vLLM](plan-integrate-vllm-2026-11-09.md), [plan RAG](plan-integrate-rag-2026-11-09.md), [competition context](../competition-context.md).

## Global Constraints

- Không sửa checkout `rag` có training chưa commit; tích hợp trong worktree riêng trên nhánh người dùng yêu cầu.
- Không tải/chạy full checkpoint trên máy cá nhân; test bằng fixture CPU.
- Tổng tham số inference ≤15B; revision/hash/license theo registry/evidence, giữ lineage checkpoint fine-tune.
- Offline Docker; dataset chỉ đọc, output `/result/submission.jsonl`, log `/result/logs/`, scratch `/work`; ≤6 giờ gồm load/index.
- Giữ raw-schema corpus đúng split/business, không index đáp án/log, SQL tính đầy đủ dòng.
- Các phép so dùng cùng tập/model/prompt nền/ngân sách; không tự nhận đã đo chất lượng GPU.
- Merge local được người dùng yêu cầu; không push GitHub.

## Review Focus

- RAG tắt/rỗng/lỗi phục hồi không thay baseline messages hoặc SQL.
- Encoder/reranker lỗi load hoặc OOM không trở thành một lượt RAG thành công/abstain hàng loạt.
- Tool history tăng: bỏ context RAG trước khi overflow, giữ nguyên câu hỏi/rulebook/history.
- Full inventory phải đi xuyên staging, relocate, wheelhouse và Docker; thiếu/sai checkpoint không tự đổi base.
- Comparator không được xác nhận sai phép thử nếu RAG, nguồn, model hoặc budget thay đổi ngoài arm được cho phép.

## Task 1: Hợp nhất source và ghi baseline

**Files:** lịch sử Git, tài liệu tích hợp này; giữ uncommitted training ở checkout gốc.

**Interfaces:** consume refs `a726f39`, `6be9ea3`, target `codex/team-gpu-layout`; produce merged source và checklist các khác biệt có chủ ý.

- [ ] Chạy suite tại target trước merge, lưu log và các lỗi sẵn có.
- [ ] Merge hai refs bằng merge commit, không reset/force-push; giữ các commit training đã có ở target.
- [ ] Loại các thay đổi heuristic nghiệp vụ không thuộc retrieval; giữ tools benchmark và adapter hữu ích.
- [ ] Kiểm Git ancestry và checkout gốc còn nguyên.

## Task 2: RAG theo contract, context và lifecycle

**Files:** `src/datamate/config.py`, `retrieval/{core,contracts,encoders,context,index}.py`, `agent/{runner,prompts}.py`, `application.py`, `models/registry.py`; tests retrieval/integration.

**Interfaces:** `build_retriever(settings, split)`; `prepare(business_ids, ingestor, remaining_seconds)`; `retrieve(question,business_id,catalog,remaining_seconds)`; `identity()/runtime_info()/close()`; `RetrievalFatalError`.

- [ ] Viết và chạy RED cho partial load, fatal, close race, scope/provenance/cache và actual token budget nhiều lượt.
- [ ] Giữ CLS + normalize 1024D; batching/token limits/dtype/chunking là config; init atomic, query deadline, fatal không bị nuốt.
- [ ] Chuẩn bị model/index ngoài deadline câu nhưng trong wall budget; dùng nguồn raw, manifest/checksum/atomic cache, scope trước top-k.
- [ ] Ghép context bằng tokenizer generator, giảm nguyên chunk trước mỗi generate; none/rỗng giữ baseline.
- [ ] Chạy GREEN các test liên quan, ghi ledger và commit.

## Task 3: Full runtime artifact và lỗi vLLM

**Files:** `models/{artifacts,runtime,vllm_backend}.py`, `batching.py`, `tools/{runtime_artifacts,install_runtime,package_solution,diagnose_train}.py`, `docker/install_runtime.py`, `pyproject.toml`; tests runtime/package.

**Interfaces:** shared inventory/path mapping cho generator/assistant/embedding/reranker; dependency profile backend cộng RAG.

- [ ] RED cho nguồn nonempty chưa seal, lock thiếu dependency, full RAG stage/relocate/package và fatal một worker.
- [ ] Chuyển tools dùng active models; giữ manifest generator cũ; verify đúng pinned RAG weights và file configs offline.
- [ ] Resolve dependency project + extra và lock constraints, fail-fast gói thiếu; notebook chọn backend cùng Settings.
- [ ] Deadline init tính từ đầu, cleanup báo rõ giới hạn SDK; metadata thực tế, không che lỗi/đổi model.
- [ ] GREEN các test runtime/package và commit.

## Task 4: Lựa chọn inference và ba arm đối chứng

**Files:** `cli.py`, `tools/{create_inference_notebook,kaggle_workflow,compare_runtime_benchmarks,benchmark_runtime}.py`, notebook sinh ra, `configs/comparison/*.json`, công cụ compare chung nếu cần; tests CLI/compare.

**Interfaces:** `--backend transformers|vllm`, `--rag-mode none|dense|dense_rerank`; baseline, vllm, rag cùng source và selected IDs.

- [ ] RED cho merge config/env/CLI, preset parity và comparator từ chối thay đổi ngoài arm cho phép.
- [ ] Cung cấp lựa chọn independent trong CLI/notebook; Torch là alias UX cho Transformers, không tạo vòng agent riêng.
- [ ] Giữ source/ID/hash/timing tools Hải, thêm comparison RAG có prompt thay đổi có chủ ý; báo thời gian cold/inference/timeout/điểm nếu có nhãn.
- [ ] Chuẩn bị preset cùng generator/budget; baseline chạy trước, vLLM default một worker cho đối chứng và cho phép tăng worker explicit.
- [ ] GREEN test entrypoints/compare và commit.

## Task 5: Tài liệu, kiểm chứng và merge hoàn tất

**Files:** README, `docs/run-gpu.md`, báo cáo tích hợp và nhật ký ngữ cảnh; tests toàn repo.

- [ ] Cập nhật cách stage đúng model, cài extras và chạy ba arm; phân biệt quan sát Hải với kết quả cần chạy lại.
- [ ] Chạy suite toàn repo + smoke DemoBackend/packaging, ghi rõ các lỗi sẵn có và giới hạn GPU.
- [ ] Fresh independent review toàn diff, sửa Important/Critical bằng RED→GREEN.
- [ ] Commit phần sửa, xác minh target chứa cả hai refs; giữ checkout training gốc và không push.
