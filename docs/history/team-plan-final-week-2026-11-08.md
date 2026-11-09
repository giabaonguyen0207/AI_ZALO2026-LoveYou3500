**DataMate — Đề xuất chia việc cho nhóm ba người trong tuần cuối**

Ngày soạn: **08/11/2026**, múi giờ Asia/Saigon. Kế hoạch dành cho tình huống nhóm chọn hệ thống DataMate hiện tại làm nền. Đây là đề xuất để nhóm chốt; chưa phân công người thật, cấu hình GitHub, tạo issue/PR hoặc thay đổi phần mềm.

Mốc làm việc: chuẩn bị ngày 08/11; bảy ngày chính 09–15/11; ngày 16/11 dự phòng. Hạn **16/11/2026 lúc 23:59 GMT+7** lấy từ [ngữ cảnh đội cung cấp](../competition-context.md:434). Đọc lại portal trong phiên này vẫn không lấy được nội dung văn bản, nên chưa xác nhận độc lập mốc hạn hoặc Docker guideline. Việc đối chiếu hướng dẫn chính thức là đầu việc ngày đầu.

Giả định ba thành viên đều có thể làm việc mỗi ngày, chưa có thông tin về số giờ hoặc số GPU. Lập một hàng đợi GPU chung ngay cả khi dùng nhiều tài khoản/máy; nếu nguồn lực hạn chế, bỏ các thí nghiệm tùy chọn trước, giữ thời gian đóng gói và chạy lại.

**1. Cách chia việc đề xuất**

Chia theo phần của hệ thống có thể phát triển độc lập, kèm người chịu trách nhiệm đầu ra và phạm vi file. Mỗi người có quyền dùng AI để thực hiện ticket của mình; người nhận ticket vẫn chịu trách nhiệm đọc thay đổi, chạy kiểm tra và giải thích được kết quả.

| Người | Trách nhiệm chính | Công việc tuần cuối | Sản phẩm phải bàn giao |
|---|---|---|---|
| **A — Bạn, người nắm hệ thống** | SLM/controller và tích hợp | Chốt giao thức; sửa đường final/projection và ngân sách review; cải thiện prompt có kiểm soát; review tích hợp; chọn cấu hình/model và bản nộp | Contract rõ, controller chạy đúng luồng, PR nhỏ có bằng chứng, bản tích hợp có commit xác định và quyết định chọn ứng viên |
| **B — Thành viên 2** | Dữ liệu và công cụ tính toán | Kiểu tham số, khóa thực thể, mapping schema, nguồn/trạng thái, thời gian/đơn vị; tính đúng theo plan; tạo ca hồi quy và curate ví dụ TRAIN | Engine giữ đúng nghĩa dữ liệu trong các ca đối chứng, phân biệt thiếu dữ liệu/chưa hỗ trợ/không phát sinh; bộ ca được thực thi và báo cáo lỗi còn lại |
| **C — Thành viên 3** | Evaluation, runtime và đóng gói | Thiết lập CI CPU; quản lý benchmark/GPU; đo load/infer; kiểm soát dependency; notebook/source kit; Docker offline và rehearsal | Bảng so sánh tái lập, manifest cho từng run, image chạy từ môi trường sạch, checksum và hướng dẫn tái chạy |

Vai C cần bắt đầu Docker từ ngày đầu. Công việc này bao gồm sửa batch/runtime/đóng gói và tự động hóa phép đo; không chỉ chạy test cuối tuần. Người A nên giữ khoảng một phần ba thời gian cho review, tích hợp và xử lý phụ thuộc; giảm thí nghiệm cá nhân nếu phần này quá tải.

Fine-tune là nhánh nghiên cứu có giới hạn, A phụ trách recipe, B hỗ trợ ví dụ đã kiểm chứng, C xếp lịch GPU và chấm độc lập. Không dành trọn một người cho fine-tune khi baseline và image thi chưa sẵn sàng.

**2. Phân quyền sửa file để giảm xung đột khi vibe code**

| Chủ sở hữu | Phạm vi chính |
|---|---|
| A | `agent/runner.py`, `agent/protocol.py`, `agent/native.py`, `agent/prompts.py`, `agent/verification.py`; `training/*` cho ticket SFT tùy chọn |
| B | `data/*`; `agent/planning.py`, `agent/tools.py`, `agent/ledger.py`, `agent/schema_repair.py`, `agent/preparation.py` |
| C | `models/*`, `application.py`, batch/CLI khi có ticket; `evaluation/*`; Docker, `predict.sh`, CI, script tạo notebook/đóng gói/benchmark và notebook bàn giao |
| A điều phối | `domain.py`, `config.py`, các preset chính, `configs/submission.json`, dependency/lockfile, `tests/conftest.py` và thay đổi giao thức chung |

Đường dẫn trên được hiểu trong `src/datamate/` nếu không có prefix khác. Người sửa một module chịu trách nhiệm test tương ứng. Không chia hai nửa cùng một file lớn cho hai phiên AI khác nhau: `planning.py` có một owner là B, dù A là người chốt giao thức planner.

C có thể đề xuất thay đổi dependency hoặc cấu hình nhưng phải đi qua PR tích hợp do A điều phối. Các checkpoint/manifest và gói frozen được bảo toàn; quyền sở hữu `models/*` không có nghĩa được tự đổi model/revision hoặc chế độ dừng.

Nếu ticket cần sửa file của người khác, ghi rõ trong issue trước khi bắt đầu; hai bên chọn một người viết patch và người còn lại review. Không để AI tự mở rộng phạm vi vì thấy một refactor liên quan.

**3. Việc phải chốt cùng nhau trước khi tách nhánh**

Buổi kickoff khoảng 60–90 phút, đầu ra là một tài liệu ngắn và danh sách issue:

1. **Baseline có thể tái chạy.** Workspace hiện có nhiều thay đổi chưa commit. A rà soát và lưu đúng source pipeline 6 đang dùng vào commit, đối chiếu với source kit và config; C tái chạy từ checkout riêng. Tạo tag baseline sau khi đúng source đã nằm trong commit. Tag trên HEAD cũ không đóng băng được thay đổi còn trong working tree.
2. **Contract giữa các module.** Chốt trường/enum plan hiện được hỗ trợ; shape kết quả công cụ; ý nghĩa `verified`/availability; cách final tham chiếu query; error feedback và ngân sách sửa. Bản đề xuất giao thức mới cần fixture chạy được qua parser → validator → executor → final.
3. **Bộ đo và phần cứng.** Giữ nguyên byte/order/seed bộ 200 câu TRAIN, evaluator và các mẫu hồi quy. Chọn model, dtype, GPU và ngân sách nền; mỗi thử nghiệm chỉ thay yếu tố được nêu hoặc ghi rõ các thay đổi đồng thời.
4. **Hợp đồng bàn giao.** Thống nhất commit nguồn, hash config/dataset/model, cách lưu kết quả và tên run; bản benchmark và bản Docker phải truy được về cùng ứng viên.
5. **Giới hạn tuần cuối.** Giữ kiến trúc hiện tại, ưu tiên lỗi có log; quyết định điều kiện tiếp tục/dừng fine-tune, review hoặc tối ưu runtime.

Ví dụ những chi tiết contract phải có người chịu trách nhiệm: B cung cấp cột/shape/type hợp lệ; A chốt cách chọn kết quả mà không bắt SLM chép số hoặc tên biểu thức SQL; C kiểm tra submission cuối vẫn giữ schema cuộc thi. `verified` của công cụ không chứng minh plan hiểu đúng câu hỏi.

**4. Lịch bảy ngày chính và ngày dự phòng**

| Ngày | A — SLM/controller/tích hợp | B — Dữ liệu/công cụ | C — Evaluation/runtime/Docker | Mốc kết thúc ngày |
|---|---|---|---|---|
| **08/11 — chuẩn bị** | Rà working tree, chuẩn bị baseline/contract draft | Đọc engine và chẩn đoán, chọn ca lỗi | Kiểm máy/GPU/quyền repo; thu Docker guideline | Danh sách phụ thuộc, người nhận việc và baseline cần lưu |
| **09/11 — ngày 1** | Chốt contract, tạo issue và commit/tag baseline đã kiểm tra | Tái hiện lỗi kiểu/khóa; chuẩn bị test phân biệt đúng/sai | Tái chạy baseline; dựng CI CPU; thử Docker với model thật theo guideline | Có baseline tái chạy, bằng chứng source/config và danh sách blocker |
| **10/11 — ngày 2** | Sửa final/projection và lỗi ngân sách review trong PR nhỏ | Sửa tham số typed và kiểm tra khóa ở các nhánh, gồm ledger | Hoàn thiện harness so paired; tiếp tục xử lý image/dependency | Các PR có kết quả kiểm tra; vẫn giữ đường fallback |
| **11/11 — ngày 3** | Tích hợp đợt 1; thử single/review có kiểm soát; chốt API cho training | Chạy hồi quy; sửa mapping/operation có bằng chứng; curate TRAIN | Chạy benchmark bản tích hợp, so backbone khi đủ nguồn lực và cùng điều kiện | Một bản tích hợp đo được; contract ổn định cho ứng viên SFT |
| **12/11 — ngày 4** | Thử SFT pilot nếu đủ điều kiện; hoặc tập trung lỗi planner/prompt | Replay và kiểm chứng corpus mới; đóng các ca engine còn ảnh hưởng lớn | Benchmark ứng viên; đo cold start, CPU, generation; khóa stack đã chạy | Tối đa một ứng viên SFT; dự báo được thời gian train/merge/eval |
| **13/11 — ngày 5** | Chọn backbone/checkpoint, chế độ review và cấu hình nộp | Chốt dữ liệu/engine; giải thích regression và giới hạn | Đo cùng bộ 200, hoàn thiện image và manifest ứng viên | Chọn release candidate; đóng băng tính năng cuối ngày |
| **14/11 — ngày 6** | Review toàn bộ cấu hình/nguồn, chỉ sửa blocker có kiểm chứng | Hỗ trợ tìm lỗi từ lần chạy sạch, chuẩn bị thao tác tái chạy | Checkout/build sạch; chạy image offline với model thật; đo toàn wall time | Image ứng viên, output/log và báo cáo; lỗi mới có thể dẫn tới quay về bản ổn |
| **15/11 — ngày 7** | Kiểm bản nộp, thông tin khai báo; chủ trì bàn giao/nộp theo guideline | Tự chạy lại entrypoint như người nhận mới; làm người backup vận hành | Đóng gói cuối, checksum, kiểm link/quyền tải theo yêu cầu và lưu bản sao | Mục tiêu hoàn tất nộp từ ngày 15; lưu biên nhận và đúng artifact đã nộp |
| **16/11 — dự phòng** | Theo dõi xác nhận và xử lý vấn đề nộp/khả năng tái chạy | Hỗ trợ fallback đã đo | Hỗ trợ tải lại/kiểm checksum | Không mở thêm tính năng/thí nghiệm; giữ nguyên bản đã xác nhận |

Mọi lần sửa sau khi đóng băng tính năng đều tạo commit mới, chạy lại các kiểm tra liên quan và rehearsal khi có thể ảnh hưởng artifact. Không giữ tên release/checksum cũ cho image đã thay đổi.

Đến ngày 09 nếu thiếu guideline/base image hoặc GPU phù hợp, ghi nhận là blocker của đường nộp và xử lý ngay. Không chờ đến ngày 14 mới phát hiện Docker dev hiện có chỉ là smoke/demo.

**5. Backlog có thể chuyển thành GitHub Issues**

Các mã dưới đây là nhãn kế hoạch trong tài liệu, chưa phải issue đã tạo trên GitHub.

| Mã | Owner | Đầu việc | Điều kiện hoàn tất | Ưu tiên |
|---|---|---|---|---|
| G01 | A, C xác nhận | Lưu baseline và contract module | Checkout độc lập có đúng pipeline/config/source; fixture đi hết luồng | Bắt buộc |
| G02 | C | CI CPU và mẫu báo cáo PR/run | Test tự chứa chạy trên checkout sạch; test cần model/dataset thật được tách rõ | Bắt buộc |
| G03 | A, B review | Sửa chốt result reference/projection | Ca trước mất candidate được phục hồi đúng shape/type; không nhận số model tự sửa hoặc phép chiếu mơ hồ | Bắt buộc |
| G04 | B, A review | Typed filters và kiểm khóa trước số 0 | Phân biệt key hợp lệ không phát sinh, key sai, tên thay key; kiểm cả debt/ledger; giữ ID dạng chuỗi | Bắt buộc |
| G05 | B | Sửa nhóm lỗi engine được chọn từ log | Giữ đầy đủ kỳ/nguồn/đơn vị/grain; có ca dương và ca âm, không tạo nhánh theo ID | Theo tác động |
| G06 | A | Thử review/prompt có kiểm soát | Báo PA, wrong, lost-correct, revise thật, calls và thời gian; chọn bằng kết quả | Theo nguồn lực |
| G07 | C | Chấm và so ứng viên trên cùng benchmark | Run có source/config/model/data/runtime identity, output và raw log; scorer giữ nguyên | Bắt buộc |
| G08 | A; B cấp corpus, C cấp GPU | Một ứng viên Gemma SFT | Corpus phù hợp contract mới, TRAIN-only; có base-vs-SFT generation và artifact merge/seal | Tùy chọn |
| G09 | C | Docker inference với model thật | Theo guideline, offline, input chỉ đọc, output/log đúng; đo load+ingest+infer | Bắt buộc, bắt đầu ngày 1 |
| G10 | B thực hiện, C hỗ trợ | Rehearsal từ môi trường sạch | Không dựa vào cache/mount ngầm trên máy tác giả; người B có thể chạy từ hướng dẫn | Bắt buộc |
| G11 | A chủ trì, C đóng gói | Release và nộp | Commit/config/model/image/hash rõ, vật liệu yêu cầu đủ, link tải và biên nhận được kiểm tra | Bắt buộc |

Tập hồi quy có thể tham chiếu câu TRAIN và ID để tìm log; runtime không được dùng question/business ID để quyết định công thức/đáp án. Không thay evaluator để tăng điểm ứng viên.

**6. Quy trình GitHub gọn cho ba người**

Giữ một nhánh tích hợp là nhánh mặc định, ví dụ `main`, và nhánh ngắn cho từng issue. Tên đề xuất: `codex/g03-final-projection`, `codex/g04-typed-keys`, `codex/g09-docker-offline`. Không cần thêm nhiều tầng nhánh trong tuần cuối.

Luồng làm việc:

```text
Issue rõ phạm vi → nhánh riêng → patch + kiểm tra → PR
→ người khác review → CI phù hợp → merge bản đã kiểm tra
→ benchmark bản tích hợp theo mốc → ghi kết quả và chọn/loại ứng viên.
```

- Một issue có một owner và kết quả có thể kiểm tra trong khoảng nửa đến một ngày. Mỗi PR tập trung một thay đổi hành vi; tách refactor/format lớn khỏi sửa lỗi.
- Mở draft PR sớm để lộ phạm vi đang sửa; cập nhật từ nhánh tích hợp hằng ngày. Sau khi giải quyết xung đột, chạy lại kiểm tra trên nội dung mới.
- PR của A do B hoặc C review; PR của B do A review; PR runtime của C do B kiểm khả năng chạy và A xem thay đổi giao thức/cấu hình. Không để tác giả tự là reviewer duy nhất.
- A giữ nhịp tích hợp và chọn release. Chốt người backup và quyền thao tác phù hợp trong buổi kickoff để khi A vắng vẫn xử lý được việc bàn giao đã thống nhất.
- Nếu gói/quyền repository hỗ trợ, đặt nhánh bảo vệ với một approval và các status checks bắt buộc. `CODEOWNERS` dùng để chỉ định reviewer theo module; nó không tự khóa quyền sửa file hoặc tự bảo đảm mọi PR phải được duyệt. Tham khảo [branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) và [CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners).
- Nếu tính năng bảo vệ không có trên gói repo hiện tại, giữ cùng quy ước review và merge trong nhóm; không chuyển repo sang public chỉ để bật tính năng.
- CI có thể dùng GitHub Actions cho Python/CPU tests. Thiết kế tập kiểm tra tự chứa bằng fixture, không cần tải full weights/dataset; GPU benchmark là bước riêng có run identity. Đây là hướng thiết lập, chưa có CI được tạo trong phiên này. [GitHub Python CI](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- Code, config, tài liệu, hash và báo cáo gọn đi qua Git. Dataset/model/ZIP/image/log lớn đi qua nơi lưu artifact mà nhóm đã chọn, ghi đường dẫn và SHA256 vào báo cáo. `.gitignore` hiện đã loại các thư mục dữ liệu, model và outputs; không force-add toàn bộ để chia sẻ nhanh.

**7. Mẫu giao ticket cho AI và mẫu PR**

Mỗi người nên bắt đầu phiên AI bằng phạm vi issue, source commit và contract đã chốt. Mẫu sau là hướng dẫn cho phiên coding; không phải yêu cầu thực thi của tài liệu này.

```text
Mục tiêu: [một hành vi cần sửa, kèm ví dụ lỗi/đầu ra mong đợi].
Nguồn nền: [commit SHA], đọc AGENTS.md và tài liệu contract hiện hành.
Owner: [A/B/C]. Chỉ sửa: [danh sách file hoặc phạm vi rõ].
Giữ: [API/enum/output contract/model revision/scorer liên quan].
Không sửa file ngoài phạm vi hoặc refactor phần khác để tiện triển khai.
Nếu phát hiện phụ thuộc ngoài phạm vi, mô tả và đề xuất patch riêng.
Kiểm tra bắt buộc: [ca có thể phân biệt đúng và sai, gồm regression cần giữ].
Báo: thay đổi, kiểm tra thực sự đã chạy, kết quả, giới hạn và diff cần review.
Giữ quy tắc dự án: luật theo split/as_of_date; inference offline;
không quyết định đáp án theo ID; không dùng public/private để training.
Không ghi PA/timing hoặc kết luận đã hoàn tất nếu chưa thực sự đo.
```

Mỗi PR có tối thiểu:

```text
Vấn đề và hành vi sau thay đổi:
Issue / owner / commit nền:
File và contract bị tác động:
Kiểm tra đã chạy, kết quả và môi trường:
Run ID / link artifact / hash nếu có benchmark:
Regression hoặc giới hạn còn lại:
Cách quay lại cấu hình/bản trước:
```

Người review nên đối chiếu ít nhất một ca lỗi và một ca cần giữ đúng, đọc diff ở ranh giới module và kiểm kết quả thô. Nhận xét tự động của AI có thể hỗ trợ việc này; trách nhiệm nghiệm thu vẫn thuộc người review.

**8. Cách dùng GPU và chọn ứng viên**

C quản lý danh sách run chung: commit, thay đổi cần đo, model/checkpoint, GPU/dtype, dataset/order, config, người yêu cầu, thời gian dự kiến và trạng thái. Tránh ba người cùng khởi chạy benchmark giống nhau hoặc dùng tên run khác nhau cho cùng một ứng viên.

Mỗi thay đổi qua ba mức kiểm tra phù hợp:

1. **CPU/fixture và replay mục tiêu** trước PR; xác nhận cơ chế, không gọi đó là model PA mới.
2. **Model smoke trên tập TRAIN cố định** cho lỗi liên quan; giữ tập và cách chọn trước khi xem kết quả để tránh chọn câu có lợi.
3. **Benchmark tích hợp 200 câu** tại các mốc chọn ứng viên, và rehearsal Docker với model thật trước release. CPU test pass không thay cho hai mức này.

Ưu tiên lịch GPU: baseline tái lập → ứng viên bắt buộc → rehearsal image → thí nghiệm tùy chọn. Khi đủ nguồn lực có thể đo ứng viên song song, nhưng phép so tốc độ phải dùng điều kiện phần cứng tương ứng và ghi khác biệt. Đừng so mỗi người chạy một GPU khác nhau rồi kết luận về tối ưu code.

Mẫu 200 câu là TRAIN development và có thể trùng nguồn dùng curate/SFT. Không gọi PA trên nó là chất lượng holdout sau fine-tune. Ghi riêng phần overlap và dùng evaluation tách theo doanh nghiệp khi so model; các tập đã được nhìn để sửa hệ thống tiếp tục được ghi là validation phát triển.

Các điều kiện tiếp tục SFT: contract đã ổn định; corpus mới replay được; pilot dự báo đủ thời gian train, merge, seal và generation evaluation trước mốc chọn ứng viên 13/11; không làm trễ Docker. Bộ 136 câu/144 quyết định hiện có khóa theo fast v5 cũ, chưa được dùng thẳng như dữ liệu pipeline 6 đã xác nhận. Nếu không đủ điều kiện, dùng base checkpoint đã kiểm chứng và đóng nhánh thí nghiệm.

Chọn bản bằng PA, số sai/regression và tổng thời gian, với cùng bộ đo. Nếu chất lượng chưa phân biệt rõ và không có lỗi nghiêm trọng, ưu tiên bản tái chạy ổn, nhanh và ít phụ thuộc. Không gán nhãn chiến thắng từ một câu CPU đã sửa tay hoặc một lần loss training giảm.

**9. Điều kiện bàn giao cuối tuần**

Nhóm cần kiểm được các đầu ra sau, phân biệt rõ phần đã đo và phần còn suy tính:

- Commit/tag thực sự chứa đúng code, cấu hình, model revision/manifest và environment đã dùng; notebook/source kit khớp source, tránh code cũ còn trong kernel.
- Image chứa model và dependency cần thiết, chạy inference không mạng; `/data` chỉ đọc; `/result/submission.jsonl` và `/result/logs/` đúng yêu cầu.
- Đo wall time từ trước load, gồm xác minh artifact, ingest, model calls và ghi kết quả. Workload rehearsal được mô tả đúng; chạy đủ 200 câu không tự chứng minh đủ ngân sách toàn private test.
- Kiểm đúng type/giá trị/thứ tự trên ca đối chứng; không biến lỗi lookup hoặc nguồn chưa hỗ trợ thành số 0/unanswerable.
- Tổng tham số mọi model inference ≤15B; checkpoint trọng số, revision, thời điểm công khai trước 01/10/2026 và license có bằng chứng. Không dùng đồng thời Qwen 9B và Gemma 12B trong giải pháp inference.
- Không có test data vào training; bản SFT nếu được chọn có source/corpus/report truy nguyên được, cùng source training cần bàn giao theo guideline.
- Có baseline/fallback đã đo, artifact/hash bản cuối, hướng dẫn tái chạy và ít nhất một người ngoài tác giả đã vận hành lại.
- Người nộp và người backup cùng biết vị trí artifact, checksum, cách xác nhận tải và biên nhận. Số lượt nộp và nội dung phải theo hướng dẫn chính thức đã đối chiếu.

**Nhịp phối hợp mỗi ngày:** khoảng 10 phút đầu ngày để chốt owner/phụ thuộc/GPU, khoảng 10–15 phút cuối ngày để xem PR và số đo. Cập nhật gồm việc đã hoàn tất kèm bằng chứng, việc bị chặn, và đầu ra ngày kế tiếp. Tính từ ngày 13, cắt việc tùy chọn trước khi cắt rehearsal hoặc thời gian nộp.

Nguồn hiện trạng: [phân tích pipeline 08/11](system-pipeline-analysis-2026-11-08.md), [rollback pipeline 6](rollback-pipeline6-2026-11-08.md), [Docker và giới hạn kiểm chứng](../docker.md), [benchmark team200](../team200-colab.md). Chưa triển khai các thay đổi trong kế hoạch, chưa chạy thêm benchmark/GPU hoặc tạo nội dung trên GitHub trong phiên này.
