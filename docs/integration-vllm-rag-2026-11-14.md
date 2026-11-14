# Rà soát và hợp nhất vLLM/RAG — 14/11/2026

Nguồn: yêu cầu chủ dự án ngày 14/11; Git đã fetch; hai plan kỹ thuật 09/11; hai file kết quả do Hải cung cấp trong `outputs/runs/hai` của workspace gốc. Đây là báo cáo triển khai/quan sát nội bộ, không phải xác nhận từ BTC.

## Nhánh và phạm vi

Target `codex/team-gpu-layout` ban đầu `ab2ba3a`; đã merge bằng merge commit cả `origin/codex/hai-vllm` (`a726f39`) và `origin/rag` (`6be9ea3`). Target đã chứa bản vLLM cũ `b534be9`, nên phần mới của Hải chủ yếu bổ sung benchmark/Kaggle/metadata. Không force-reset và chưa push GitHub.

Tích hợp trong worktree riêng vì checkout gốc đang ở `rag` với training/notebook chưa commit. Các sửa đổi đó được giữ nguyên; không mang chúng vào target. Hai plan cũ giữ vai trò đặc tả lịch sử; những tên class/contract đã gom và phần triển khai thực tế mô tả dưới đây.

## Phần giữ lại và điều chỉnh

| Phần | Giữ lại | Sửa/bổ sung so với nhánh nhận |
|---|---|---|
| Hải — vLLM | Adapter async, cùng ChatBackend/Agent loop, shared engine cho workers, abort, prefix-cache tùy chọn, metadata cached token, bộ đo/Kaggle có hash và log | Startup tính từ đầu gồm verify/import/tokenizer; phase timing/dtype/hardware thực tế; báo lỗi cleanup; watchdog process cho entrypoint batch; lỗi fatal sequential kèm question ID |
| Thịnh — RAG | Registry pin đúng encoder/reranker; CLS + chuẩn hóa 1024D; corpus rulebook/README/schema; scope split/business; optional retrieval hook | Atomic load cả hai model, lỗi fatal/OOM không thành RAG thành công; batch/dtype/token limits cấu hình được; cache đĩa có checksum/identity; provenance đường dẫn raw/nested thực; prepare model/index trong startup budget; dùng tokenizer generator để giới hạn context và giảm chunk theo history |
| Composition | Adapter + Strategy, dependency injection, Transformers/Demo giữ lại | `create_agent()` sở hữu/prepare RAG trước generator; question workers chia sẻ thành phần và ingestor; none/rỗng/fallback giữ prompt baseline; lỗi load/fatal dừng batch rõ ràng |
| Offline runtime | Tools artifact/bundle/wheelhouse/Docker của Hải | Full inventory/model paths/extras xuyên prepare/verify/install/package; relocate encoder/reranker; lock không bỏ dependency project; thiếu/sai manifest nguồn không tự đổi checkpoint fine-tune về base |
| Đối chứng | Runtime observer, per-call hashes/tokens, deadline và answer differences | Cùng source cho ba arm; CLI/notebook Torch/vLLM và RAG độc lập; common dtype; source/RAG/artifact/sampling/wall-budget gates; RAG được đổi context nhưng phải giữ hash prompt nền; gold chỉ đọc sau inference |

Không giữ các sửa đổi riêng của nhánh RAG trong `agent/planning.py`, `agent/protocol.py`, `agent/tools.py` về suy metric doanh thu, sửa projection và fuzzy entity. Chúng thay logic đáp án ngoài retrieval và có thể gây suy luận sai hoặc làm đối chứng bị lẫn nguyên nhân. Ba file này trở về target ban đầu; cải thiện nghiệp vụ nếu muốn phải có phép thử riêng.

`VietnameseEncoders` gom hai adapter encoder/reranker; `NullRetriever`/`ContextRetriever` cùng `rag.mode` thực hiện Strategy thay vì tạo ba class trùng phần index/lifecycle. Không dùng encoder/reranker như ChatBackend sinh text. Corpus hiện chưa index cell values hay example đáp án; SQL vẫn tính đủ dòng. Những tên class trong plan là thiết kế giao diện, không phải yêu cầu phải sinh class rỗng chỉ để trùng tên.

## Đọc kết quả Hải đúng phạm vi

Hai log chứa 24 ID, không có gold/reference:

| Phép đo | Transformers | vLLM |
|---|---:|---:|
| Startup/load | 108,98s | 652,74s |
| Sau load/inference pipeline | 1275,10s | 713,13s |
| Tổng cold | 1384,08s | 1365,87s |
| Câu chịu ảnh hưởng deadline | 4 | 5 |

Tỷ số thời gian quan sát khoảng **1,79× sau load**, nhưng chỉ **1,013× tổng cold**; khác 4 đáp án/type, trong đó 3 khác biệt chịu ảnh hưởng deadline. Comparator gốc đã báo `paired_comparison_valid=false`. Không có nhãn nên không kết luận chất lượng giữ nguyên hoặc tăng; cũng không gọi tỷ số này là speedup thuần engine. Lượt mới phải giữ tất cả ID, kể cả timeout/lỗi/abstain.

Docker cài dependency/copy weights lúc build nên không cài/tải khi BTC chạy. Nó vẫn import/nạp VRAM/khởi tạo vLLM; không được trừ toàn bộ startup của log như chi phí cài thư viện. Xem [hướng dẫn Docker](docker.md).

## Chạy lại

Ba preset ở `configs/comparison/{baseline,vllm,rag}.json`; `tools/compare_system.py` chạy baseline trước, sau đó vLLM, Torch+RAG. Cùng source/checkpoint/bộ ID/dtype/ngân sách; mặc định một worker. [Lệnh GPU và cách đọc output](run-gpu.md#so-sánh-tự-động-ba-cấu-hình).

CLI `--backend torch|vllm` và `--rag-mode none|dense|dense_rerank` độc lập, có thể bật cả hai. `configs/submission.json` vẫn là lựa chọn của chủ dự án, chưa tự chuyển sang cấu hình thắng khi chưa có kết quả.

## Kiểm chứng và giới hạn

Kiểm thử fixture CPU, không tải/chạy full checkpoint hoặc build Docker GPU. Baseline trước merge: 352 pass, 4 skip, 4 fail; các lỗi gồm hai tham chiếu notebook training đã bỏ, thiếu psutil trong probe training, và test deadline bị phụ thuộc tốc độ I/O host.

Nhóm kiểm thử tích hợp đã chạy 59 pass/1 skip trước review. Fresh independent review không tìm Critical, phát hiện ba Important: khóa RAG notebook sai, source gate chưa đủ, sampling/wall-budget gate thiếu. Đã có regression tái hiện và sửa; nhóm sau review **33 pass**. Comparator mới yêu cầu cùng toàn bộ source cho cả hai phép so, từ chối custom sampling trong phép thử primary và kiểm wall budget.

Suite toàn repo cuối: **437 pass, 5 skip, 3 fail** (75,77s). Ba lỗi đều tái hiện từ baseline: `test_colab_bootstrap[LoRA]` và `test_seed_notebook...` đọc notebook training đã bị bỏ; `test_cpu_merge_ram_policy...` thiếu package `psutil` trong environment local. Không sửa training ngoài phạm vi hoặc che các lỗi này bằng skip. Test deadline được làm độc lập với I/O startup host, vẫn thực sự khởi chạy/hủy request đang chạy. Team200 trở lại mặc định baseline, cho chọn backend độc lập, sinh lại notebook/kit từ benchmark frozen đã verify hash.

Smoke được kiểm qua pipeline Demo thực sự trong worker process (3 ID, đúng submission/log, không load model); các test đóng gói fixture xác minh đủ encoder/reranker, rewrite config và dependency extra offline. Đã sinh notebook chính và gói Team200 mà không chạy full model. Chưa chạy Docker engine/GPU; kiểm packaging fixture không thay cho rehearsal image thật.

Cần GPU để xác nhận vLLM 0.33/Transformers/CUDA tương thích checkpoint, chất lượng PA, VRAM, cold startup, hiệu quả RAG và throughput. Dải version trong pyproject là ứng viên; chỉ `pip install` thành công không chứng minh stack chạy đúng. RAG thêm khoảng 1,1355B tham số; inventory tính đủ và chặn vượt 15B.

CLI `run`/notebook chính có watchdog process; API in-process và notebook Team200 tương tác giữ giới hạn timeout phối hợp SDK, không tự cưỡng bức dừng thread bị chặn. Dùng entrypoint batch có giám sát để đo chính/nộp. Phần GPU/Docker thật được để lại có chủ ý vì máy hiện tại chỉ dev/test CPU; không đánh dấu đã kiểm chứng.
