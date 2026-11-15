# Checkpoint 198 trên Colab Pro — Implementation Plan

> Superseded về nguồn mặc định sau khi nhận kết quả hoàn tất: [plan stage2](2026-11-14-final-stage2-baseline.md), [workflow hiện hành](../colab-finetuned-compare.md). Snapshot command được giữ cho rollback; notebook chính đã đổi sang bản fine-tuned hoàn tất.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Dùng đúng adapter checkpoint-198 của Bảo làm baseline chung cho Torch, vLLM và Torch+RAG trên Colab Pro; giải phóng nhánh team khỏi worktree C.

**Architecture:** Giữ nguyên Adapter + Strategy/runtime ba arm đã có. Chuẩn bị ZIP inference tối thiểu từ checkpoint Trainer đã kiểm checksum; giữ marker và report rõ đây là snapshot, không giả báo cáo train hoàn tất. Merge một lần trên Colab bằng exporter hiện có (giữ MTP/inventory), rồi truyền cùng sealed model_dir vào compare_system.

**Tech Stack:** Python stdlib, pytest CPU, exporter Transformers/PEFT hiện có trên Colab, notebook JSON sinh bằng builder.

**Spec:** Chủ dự án yêu cầu ngày 14/11/2026; checkpoint thực tế tại outputs/runs/Bảo/checkpoint-198. Đây là thay đường checkpoint trong workflow đã duyệt, không thay kiến trúc nghiệp vụ.

## Global Constraints

- Không sửa hoặc load weights checkpoint gốc, không full-model merge/infer trên laptop.
- Checkpoint là LoRA rank16, global_step198, epoch của stage khoảng0.5003; total1.5 là thông tin chủ dự án, marker xác nhận có initial_adapter_files nhưng chưa có parent report để cộng epoch độc lập.
- Base manifest bắt buộc f8b32a3f0b36609c4065a72af44938fa27ba68e4b2d70ef45a8d43b1cc37399e. Archive local31cb… không được tự thay thế.
- Model key qwen35-9b và revision registry đã pin; giữ tổng15B/offline inference, không dùng gold trong merge/index/generation.
- Colab CPU child High RAM dùng exporter đã có; kiểm RAM trước khi load, RAG/generator inference trong các process GPU mới.
- Không stash/reset/switch checkout E có training chưa commit. Nhánh team được giải phóng; nhánh làm mới codex/colab-checkpoint198 không giữ khóa nhánh team.

## Review Focus

- Adapter/marker/tokenizer sửa hoặc checkpoint chưa hoàn tất phải bị từ chối trước package/merge.
- Base hợp lệ theo registry nhưng khác exact manifest đã train phải bị từ chối trước load.
- Export không bịa số epoch hoặc training completion; optimizer/RNG/pickle không đi vào gói inference.
- Model merged có sẵn chỉ dùng lại khi provenance đúng checkpoint; output khác không bị ghi đè.
- Ba arm phải nhận cùng sealed merged artifact; pip/merge/copy diễn ra trước benchmark, không tải model trong inference.

## Task 1: Gói checkpoint và merge trên Colab

**Files:** tools/checkpoint_inference.py, tests/test_checkpoint_inference.py.
**Interfaces:** package_checkpoint(checkpoint, output, model_key, reported_total_epochs); extract_package(archive,destination); verify_package(root); merge_package(root,base,output).

- [x] RED cho gói tối thiểu, checksum, wrong base và output cũ.
- [x] Implement verify/package/extract; report snapshot; reuse merge_sql_export exporter, CPU RAM guard ở CLI merge.
- [x] GREEN và tạo ZIP thật từ checkpoint người dùng, giữ source nguyên hash.

## Task 2: Notebook Colab và bàn giao

**Files:** tools/create_checkpoint198_colab.py, notebooks/DataMate_Colab_Checkpoint198_Compare.ipynb, tests/test_checkpoint198_notebook.py, README, docs/run-gpu.md, docs/team-workflow.md.
**Interfaces:** bốn ô setup/install/prepare/compare; cùng project compare_system.py; BASE_SOURCE_DIR bắt buộc đúng base đã train; sourceZIP và checkpointZIP riêng.

- [x] RED notebook dùng model merged cho ba arm, CPU child merge và cùng dependency stack.
- [x] Sinh notebook, cập nhật hướng dẫn Git worktree/Colab; tạo source ZIP mới không chứa weights/data.
- [x] Kiểm CPU liên quan + fresh read-only review toàn diff; giữ nhánh team có thể switch tại E.

## Kết quả

20 CPU tests pass,1 skip trong nhóm liên quan; real checkpoint ZIP đã extract/verify mọi checksum. Fresh review không tìm Critical/Important. Đường dẫn Drive của base f8b32… đang chờ chủ dự án tìm lại, nên chưa merge/load/infer full model; notebook giữ guard đúng base.
