# Benchmark Team200 tùy chọn

Hải và Thịnh tự chọn benchmark; Team200 chỉ là bộ có sẵn nếu muốn dùng. Bộ v2 gồm 200 câu TRAIN, seed 4000930841, 28 doanh nghiệp; không phải holdout độc lập.

Hướng dẫn gốc và hash: [bản phát hành benchmark](history/team-benchmark-train200-2026-11-07.md). Gói v2, kit/source cũ và notebook đã phát được giữ nguyên tại archive/2026-11-09/outputs/. Nhãn/coordinator giữ riêng, không đưa cho agent hoặc RAG.

Để tạo kit mới theo source hiện tại, cần folder bản phát hành v2 gốc có participants.zip, coordinator-private.zip và release_verification.json:

```powershell
.\.venv\Scripts\python.exe tools/create_team200_colab.py --benchmark-release archive/2026-11-09/outputs/team-benchmark-train200-2026-11-07-v2 --output outputs/packages/team200-current
```

Tool kiểm hash benchmark, đóng gói source mới và cập nhật notebooks/DataMate_Colab_Team200.ipynb với hash tương ứng. Không sửa các ZIP đã khóa; gửi cả notebook và kit mới cùng phiên bản. Người mới pull phải nhận bản phát hành dữ liệu riêng trước khi build kit.

Upload notebook mới lên Colab; đặt kit vào Drive/MyDrive/DataMate. Chọn model/GPU, tên người và run mới. Checkpoint đã seal được tái dùng. Dùng RUN_LIMIT cho smoke và None để chạy đủ mẫu; không resume vào source/config cũ.

Runner có submission/metadata sáu trường, review/log, resume và tự ngắt runtime sau khi đồng bộ báo cáo. Nhãn chỉ mở sau generation. Chi phí load và inference được cộng qua các phiên khi biết đầy đủ; timing không biết giữ null.

Notebook này chuyên Colab và bộ 200; với Kaggle/GPU thuê hoặc benchmark khác, dùng notebook inference chính và CLI theo [run-gpu](run-gpu.md).
