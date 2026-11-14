# Baseline từ adapter stage 2 hoàn tất — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Chuyển bộ đối chứng Colab Pro sang kết quả hoàn tất `qwen35-9b-sql-stage2-v1-full-01/adapter`, giữ báo cáo training thật, dùng cùng model merged cho baseline/vLLM/RAG.

**Architecture:** Giữ runtime Adapter + Strategy và compare_system đã duyệt. Mở rộng gói inference hiện có cho completed adapter, xác minh parent/report/weights; dùng cùng exporter MTP-preserving và base guard. Đổi tên notebook/builder checkpoint198 thành workflow fine-tuned chung để tránh hai notebook gần trùng nhau.

**Tech Stack:** Python stdlib, pytest CPU, Transformers/PEFT exporter trên Colab, notebook JSON.

**Spec:** Chủ dự án cung cấp thư mục hoàn tất ngày14/11/2026. Đã đọc report, config và hash file thực tế: stage2 step198, epoch~0.5003159; parent step298,epoch1.0. Report parent khớp link fcc245…; weights stage2 f346ad… trùng checkpoint198.

## Global Constraints

- Không sửa artifact gốc hoặc merge/infer full model trên laptop.
- Giữ byte nguyên bản training_report9a09f391… trong adapter dùng merge; không thay bằng snapshot report.
- Base phải match f8b32a3f…; archive31cb… khác manifest, không tự đổi base. Default Drive lấy từ workflow training, kiểm hash khi chạy.
- Encoder/reranker và tất cả arm dùng generator merged có cùng manifest/dtype/IDs/source/budget; gold chỉ đọc sau generation.
- Tổng model<=15B, pin revision/license, inference offline, nguồn/output/cache không qua Git.
- Giữ nguyên checkout E và training chưa commit; cuối turn detach C, fast-forward team chỉ khi branch không checkout nơi khác, không push.

## Review Focus

- Nhầm `smoke_adapter`, checkpoint con hoặc epoch1 phải bị chặn, chọn đúng `run/adapter`.
- Sửa weights/report/config/parent hash hoặc không có report hoàn tất phải bị từ chối.
- Giữ report/tokenizer/template thật, không đưa optimizer/checkpoint/log/data vào ZIP inference.
- Report hoàn tất nhưng không có parent evidence không được tự tuyên bố tổng epoch đã xác minh.
- Notebook phải pin cả package/report, không chỉ weights (weights giống snapshot nhưng provenance khác).

## Task 1: Completed adapter package

**Files:** tools/checkpoint_inference.py, tests/test_completed_adapter_inference.py.
**Interfaces:** package_completed_adapter(run,output,parent_adapter=None,model_key='qwen35-9b'); verify_package/extract_package/merge_package tương thích cả snapshot cũ và completed adapter.

- [x] RED cho raw report preservation, parent chain, wrong/smoke/config/source/tamper và zip exclusions.
- [x] Thêm CLI package-adapter, checksum mọi file retained trước/sau ghi ZIP; verify full source và chain. Dùng report gốc cho exporter.
- [x] GREEN, đóng gói artifact thật và kiểm extract/hash, ghi profile metadata không chứa weights vào configs/comparison.

## Task 2: Notebook fine-tuned và hướng dẫn

**Files:** rename tools/create_checkpoint198_colab.py -> create_finetuned_compare_colab.py; notebook -> DataMate_Colab_Finetuned_Compare.ipynb; docs -> colab-finetuned-compare.md; test notebook, README/run-gpu/context.
**Interfaces:** source profile configs/comparison/finetuned-artifact.json; setup defaults theo DataMateSQLSFT; common dependency stack; CPU child one-time merge trên Colab; shared compare_system.

- [x] RED notebook/profile pin và PREPARE wiring, giữ baseline model selection cùng tất cả arm.
- [x] Đổi tên/nguồn mặc định; fail-fast base chưa có hoặc sai hash; giữ rollback snapshot command nhưng dùng completed adapter làm mặc định.
- [x] Kiểm nhóm liên quan, fresh review và source ZIP bàn giao, commit/kiểm Git ancestry/worktree.

## Kết quả đã chạy

Nhóm CPU liên quan36 pass1skip; toàn repo463 pass5skip3fail (49,86s). Ba lỗi cũ là notebook training LoRA/Seed72 đã bỏ và thiếu psutil trong probe training. ZIP adapter hoàn tất164113141bytes đã extract/verify toàn bộ hashes; giữ nguyên report9a09… và chuỗi parent1+.5. Chưa chạy full-model merge/inference trên Colab hoặc xác minh vị trí Drive.
