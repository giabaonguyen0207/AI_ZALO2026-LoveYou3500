# Cập nhật main và làm việc nhóm

Repo hiện tại: [AI_ZALO2026-LoveYou3500](https://github.com/giabaonguyen0207/AI_ZALO2026-LoveYou3500). Nhánh chuẩn bị bố cục là `codex/team-gpu-layout`. Phần đưa bố cục lên `main` bên dưới chỉ thực hiện nếu PR đó chưa merge; nếu đã merge, chuyển tới phần cập nhật nhánh của Hải/Thịnh.

## Nếu Git báo branch đang dùng bởi worktree

Git chỉ cho một branch được checkout tại một working tree. Worktree tích hợp ở `C:/Users/Windows/.codex/worktrees/vllm-rag-integration/ZALO_AI_2026` đã được tách khỏi `codex/team-gpu-layout` để giải phóng nhánh. Nếu gặp lại, dùng `git worktree list` để xác định nơi giữ branch; chỉ detach worktree đó khi source đã commit/sạch. Không xóa worktree đang có việc chưa lưu hoặc dùng force checkout để mở hai nơi cùng branch.

Checkout `E:` đang có training/notebook chưa commit trên `rag`. Để lưu tạm tất cả phần đó trước khi chuyển (stash không lấy model/data/output đã gitignore):

```powershell
git status --short
git stash push -u -m "WIP rag/training truoc khi chuyen nhanh"
git switch codex/team-gpu-layout
git log -1 --oneline
```

Giữ stash để xử lý riêng trên nhánh training phù hợp; không tự `stash pop` vào nhánh tích hợp. Agent không chạy stash/switch trên checkout E. Các lệnh PowerShell trên không cần dấu `\` ở cuối.

## Chủ dự án: đưa bố cục lên main

Trong PowerShell tại thư mục dự án, kiểm nhánh và diff rồi commit:

```powershell
git branch --show-current
git status --short
git diff --stat
git add -A
git diff --cached --stat
git commit -m "refactor: prepare shared GPU workspace and Docker packaging"
git push -u origin codex/team-gpu-layout
```

Trước commit, bảng staged chỉ được có source/config/docs/tests/tools/notebook; không có archive/, .venv/, models/, train/, public_test/, outputs/ hoặc bí mật. Các PDF ở docs/references được chuyển từ tài liệu đã có trong repo.

Trên GitHub chọn **Compare & pull request**, **base: main**, **compare: codex/team-gpu-layout**. Đọc Files changed, tạo PR rồi merge khi kiểm tra đạt. [Cách chọn base/compare trong tài liệu GitHub](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request).

Nhánh này tách từ qwen35-9B-pre-finetune/pipeline 6; main local trước đợt dọn còn cũ hơn. PR có thể chứa cả các commit baseline trước đó, cần xem toàn diff. Không force-push hoặc reset main để thay lịch sử.

Nếu GitHub báo conflict: fetch, merge origin/main vào nhánh này, sửa conflict và chạy lại tests trước khi push. Nếu main vừa có thay đổi của đồng đội, giữ và hợp nhất chúng thay vì chép đè source.

Sau khi PR đã merge:

```powershell
git switch main
git pull --ff-only origin main
```

## Hải và Thịnh: cập nhật source

Nếu working copy sạch:

```powershell
git status
git switch main
git pull --ff-only origin main
.\tools\setup.ps1
```

Nếu có code đang làm, commit vào nhánh của mình trước. Với nhánh tính năng đang tồn tại:

```powershell
git fetch origin
git switch ten-nhanh-dang-lam
git merge origin/main
```

Giải quyết conflict, kiểm tra lại import/config và chạy tests trước khi tiếp tục. Với việc mới, tách nhánh từ main: `git switch -c codex/hai-vllm` hoặc `codex/thinh-rag`. [Git pull --ff-only](https://git-scm.com/docs/git-pull) dừng khi nhánh phân kỳ; không xử lý bằng cách xóa code chưa lưu.

Mỗi người dùng working copy/worktree riêng, không cùng chuyển branch trên một checkout.

| Người | Nhánh tính năng | Quyền sửa file và contract |
|---|---|---|
| Hải | `codex/hai-vllm` | [Plan kỹ thuật vLLM](plans/plan-integrate-vllm-2026-11-09.md) |
| Thịnh | `codex/thinh-rag` | [Plan kỹ thuật RAG](plans/plan-integrate-rag-2026-11-09.md) |

Cả hai tách nhánh từ `main` đã cập nhật. Mỗi file dùng chung có một người sửa chính theo bảng trong hai plan; người kia yêu cầu patch qua người phụ trách, không cùng sửa bản riêng rồi chép đè.

Thứ tự merge: PR nền H của Hải (backend contract + config) → PR nền T của Thịnh (NullRetriever + composition) → hai PR tính năng phát triển song song. Sau mỗi PR nền, nhánh còn lại merge `origin/main` để nhận đúng interface. Mỗi người chạy baseline của mình trước khi tích hợp; kiểm bản ghép vLLM + RAG sau khi cả hai tính năng đã merge.

## Những gì pull không mang theo

- Dataset/checkpoint/kết quả chạy phải nhận riêng, xem [artifact](artifacts.md).
- Git pull không xóa cache, model hay .venv cũ trên máy đồng đội. Có thể tạo clone/môi trường mới để tránh lẫn dependency cũ; không chạy git clean -xfd trên repo có dữ liệu/model.
- Notebook inference chính vẫn tên DataMate_Colab_Inference.ipynb; notebook training chuyển vào notebooks/training/.
- Tạo/upload gói source mới cho GPU và restart kernel; source ZIP/Drive/Kaggle không tự cập nhật theo Git.
