# DataMate Agent Research Roadmap

**Backlog kỹ thuật cho Codex, dùng sau khi người dùng chạy baseline.**

- Ngày lập: **02/11/2026**, múi giờ Asia/Saigon.
- Trạng thái: **đề xuất, chưa triển khai các cải tiến, chưa có benchmark trước/sau**. Checkbox trong tài liệu là công việc tương lai.
- Phạm vi yêu cầu hiện tại: lưu ý tưởng thành tài liệu; không tự sửa agent, đổi model, chạy GPU, fine-tune hoặc thay baseline.
- Kế thừa cuộc trò chuyện [Đọc abstract và intro](chatgpt-conversation://6ae40b1d-274c-83ec-86a9-fe9a4fc130cc). Các nhận định trong chat được đối chiếu lại; chat không thay thế paper, code hay rulebook thực tế.
- Mục tiêu: cải thiện **đáp án đúng theo type và giá trị, Penalized Accuracy (PA), thời gian và khả năng tái lập** của agent offline. Chưa chọn phương án thắng hoặc cam kết mức tăng điểm.

## 1. Chỉ dẫn bắt đầu cho Codex

Khi người dùng giao triển khai một milestone sau baseline:

1. Đọc `AGENTS.md`, `README.md`, `docs/competition-context.md`, tài liệu kiến trúc, chẩn đoán và training hiện tại. Kiểm tra thay đổi mới hơn ngày lập roadmap.
2. **Đọc code thật và các call site trước khi sửa.** Kiểm tra `git status`, test, config, notebook và bundle baseline mà người dùng thực sự đã chạy. Không suy ra runtime Colab từ config local.
3. Xác nhận những gì đã có, những gì còn thiếu và lỗi nào xuất hiện trong trace. Bảng module ở mục 4 là điểm tìm kiếm, không phải giao diện luôn đúng ở phiên bản tương lai.
4. Hoàn thành việc bảo toàn baseline ở P0.0 trước thay đổi hành vi. Không ghi đè notebook có output, model manifest, log, prediction hoặc config baseline. Nếu baseline dở, giữ nguyên và ghi `partial`.
5. Chọn ticket nhỏ nhất giải quyết lỗi có bằng chứng, đọc dữ liệu/README/rulebook liên quan, viết kiểm thử có thể phân biệt đúng và sai. Không viết lại toàn hệ thống chỉ để khớp sơ đồ trong tài liệu.
6. Dùng cờ cấu hình để bật riêng cải tiến; mặc định tương thích đường chạy cũ. Mỗi thử nghiệm có thư mục kết quả mới. Chạy các kiểm tra phù hợp trước khi dùng GPU.
7. Báo thay đổi, bằng chứng, kết quả thực sự đã chạy và phần chưa kiểm chứng. Chỉ giữ cải tiến theo tiêu chí ở mục 9. Nếu không đạt, tắt cờ hoặc quay lại bản baseline đã lưu.

Không coi tên module, hàm, schema JSON hoặc ngưỡng thử nghiệm đề xuất dưới đây là code đã tồn tại. Không tự áp dụng toàn bộ P0–P4 khi chỉ được giao một ticket.

## 2. Nguồn nghiên cứu và ranh giới giữa paper với đề xuất

Quy ước: **[Paper]** là nội dung đã đối chiếu tại nguồn ghi bên dưới; **[Quan sát]** là trạng thái workspace; **[Đề xuất]** là thiết kế DataMate phải thử nghiệm. Kết quả trên benchmark của paper không phải kết quả DataMate. Không dùng thứ hạng hội nghị/tạp chí để suy ra hiệu quả của thiết kế.

### 2.1 ATR — EMNLP 2026

**[Paper]** A Young Kim và cộng sự, *Every Answer Has Its Own Path: Agentic Routing for Retrieval-Augmented Table Question Answering*. Nơi công bố được xác nhận từ [ACL Anthology, EMNLP 2026 main](https://aclanthology.org/2026.emnlp-main.1589/). Đã đọc Method §3.1–3.7, Appendix A–B, Table 3 và Algorithm 1 trong PDF local; đã xem trang Algorithm 1.

- Bài toán là QA kết hợp bảng và văn bản liên kết, với bảng nguồn đã được gắn với câu hỏi; không phải tìm mọi bảng trong một kho doanh nghiệp bất kỳ.
- Năm view dùng chung: text chunks, table chunks có mapping về bảng/schema, schema index, cell index và relational database. Cell/row retrieval hỗ trợ đọc hàng và nối giá trị thật với câu hỏi.
- Decomposer sinh từng `SUBQUERY` dựa vào lịch sử. Sáu trường: `sub_query`, `expected_operator`, `required_modalities`, `need_global_table_view`, `entity_mentions`, `uncertainty`; có sentinel `TERMINATE`.
- Router chọn `TEXT`, `RETRIEVE`, `SQL`, `HYBRID`. **`RETRIEVE` của ATR đọc ô/hàng bảng**, không đồng nghĩa truy xuất README. `HYBRID` dùng ứng viên `(column, value)`, LLM linker và SQL được ràng buộc bởi cột/giá trị đã grounding.
- Verifier được prompt để trả `reject/insufficient/support`, mã hóa `0/0.5/1`. Failed-route history được đưa vào router để đổi primitive khi chưa được hỗ trợ. Dừng sớm còn phụ thuộc residual uncertainty của decomposer. Các số này không phải xác suất đúng đã hiệu chỉnh.
- Appendix mô tả DistilBERT 66M học hard route labels từ LLM teacher trên HybridQA-train. Retrieval dùng BGE-M3 và BGE-reranker-v2-M3; SQL service gốc là MySQL. Đây không phải stack bắt buộc cho DataMate.
- Cấu hình gốc có tối đa 5 vòng decomposition và 3 escalation bổ sung cho mỗi sub-query. SQL scalar có thể bỏ qua synthesis; final synthesis cho nhiều hop vẫn có trong paper.
- Linker có các fallback nới ràng buộc: unconstrained, LIKE fuzzy và reroute. Vì vậy không diễn giải grounding là bảo đảm mọi query đúng nghiệp vụ, đúng join hoặc đúng metric.

**[Đề xuất]** Mượn hợp đồng plan, value grounding, verification có bằng chứng và lịch sử thất bại. Bắt đầu bằng SQLite local, lexical/fuzzy retrieval, controller Python và backbone baseline. Không sao chép API online, embedding/reranker, classifier, uncertainty threshold hay ngân sách vòng lặp của ATR nếu chưa đo.

Các khác biệt chủ động: không tự bỏ filter khi linker thất bại; cho phép sửa rồi chạy lại route SQL khi có query/bằng chứng mới; không cấm SQL cho cả câu chỉ vì một lỗi cú pháp. Kết quả số cuối lấy từ executor và serializer, không nhờ final synthesis viết lại số.

### 2.2 Squirrel — ICML 2026

**[Paper]** Jing Ye và cộng sự, *Beyond Text-to-SQL: Can LLMs Really Debug Enterprise ETL SQL?*. Nơi công bố được xác nhận từ [PMLR 306, ICML 2026](https://proceedings.mlr.press/v306/ye26d.html). Method được đọc từ bản local [arXiv:2601.18119v2](https://arxiv.org/abs/2601.18119v2), §3.1–3.5, Appendix D.1.2–D.1.3 và D.3. Đây là bản cụ thể được đối chiếu, không khẳng định byte-identical với PDF proceedings.

- Đóng góp chính là benchmark sửa SQL ETL doanh nghiệp và pipeline tạo bài lỗi. Taxonomy ba tầng được xây từ **268 SQL lỗi thực tế**; profile AST/cấu trúc giúp chọn loại lỗi phù hợp rồi chèn thay đổi nhỏ.
- Có bước lọc bằng nhiều model và expert review, không chỉ tự sinh lỗi rồi mặc định nhãn đúng. Syntax task có thông báo lỗi; semantic task có thể đúng cú pháp nhưng sai yêu cầu.
- Các metric gồm Exact Match, Graph Match của biểu diễn đã chuẩn hóa bằng Calcite, và Modify Better theo khoảng cách AST tới reference. Chúng không phải metric chấm đáp án DataMate.
- **Điểm cần tránh hiểu sai:** dù có chỗ dùng từ “execution-based”, Appendix D.1.3 nói rõ SQL **không chạy trên live engine**; TQS thực hiện phân tích tĩnh và xác minh schema/cột. Static validity không chứng minh đúng kết quả trên dữ liệu.
- Paper cũng thử Vanilla SFT, Dynamic-Masked SFT, Diff-SFT và vòng main agent/code-generation agent/TQS. Không được quy toàn bộ paper thành một taxonomy, cũng không gọi controller DataMate đề xuất dưới đây là thuật toán nguyên bản của Squirrel.

**[Đề xuất]** Dùng minimal-change mutation để xây regression suite và tập sửa SQL từ **train được phép**. Mở rộng taxonomy sang ngày, locale số, metric, trạng thái, đơn vị và join fan-out của DataMate. Chạy query thật trong SQLite sandbox, kiểm tra typed answer và witness data; AST chỉ hỗ trợ. Controller một backbone, error classes và ngân sách repair là thiết kế riêng, không phải port TQS hoặc tái lập benchmark Squirrel.

### 2.3 NVIDIA SLM-first — arXiv:2506.02153

**[Paper]** Peter Belcak và cộng sự, *Small Language Models are the Future of Agentic AI*. Đã đối chiếu [arXiv v3, ngày 22/09/2026, §6](https://arxiv.org/html/2506.02153v3#S6). Đây là **position paper**, không phải thử nghiệm chứng minh SLM/router cụ thể thắng trên DataMate.

§6 phác thảo sáu bước chuyển đổi: thu trace, lọc dữ liệu, clustering tác vụ, chọn SLM, fine-tune chuyên biệt rồi lặp cải thiện. Paper thảo luận distillation và LoRA/QLoRA. Đề xuất này là định hướng chuyển đổi hệ thống, không đưa sẵn router hay recipe training DataMate.

**[Đề xuất]** Đo các lời gọi hiện có, dùng code cho thao tác xác định, chỉ chuyên biệt hóa tác vụ lặp lại nếu dữ liệu và ablation cho thấy cần. Ban đầu giữ backbone của lượt baseline thực tế. Không mặc định “nhỏ hơn sẽ đúng hơn/nhanh hơn”; mọi model phụ vẫn tính vào 15B và phải đủ điều kiện checkpoint.

### 2.4 Scientific Reports 2026 — s41598-026-75169-w

**[Paper, mức xác minh giới hạn]** Emrah Inan, Damla Oguz, *Selecting and generating web search based features using small language models for tabular data classification*. [Trang nhà xuất bản](https://www.nature.com/articles/s41598-026-75169-w) xác nhận metadata, abstract, DOI và ngày công bố **11/10/2026**. Trang đang hiển thị bản accepted/early access. Truy cập bài/PDF trực tiếp gặp chuyển hướng xác thực; chưa đọc được toàn bộ Method trong phiên này.

Phạm vi xác nhận được: SLM hỗ trợ chọn/sinh feature cho **tabular classification**, có hướng web search/RAG. Không quy cho paper một thuật toán business QA, metric grounding, SQL repair hay mức cải thiện DataMate chưa được đọc/đo.

**[Đề xuất, hoãn]** Nếu sau này cần derived feature/semantic enrichment, chỉ thử công thức có định nghĩa và nguồn dữ liệu offline hợp lệ, đối chiếu đầy đủ Method trước khi viết ticket cụ thể. Không mang web search vào inference; không bịa cột, tỷ giá, giá vốn hoặc kiến thức bên ngoài để lấp dữ liệu thiếu. Hướng này không nằm trên đường bắt buộc P0–P3.

### 2.5 Dấu vết nguồn và những điều đã cập nhật

| Nguồn | Bản đã dùng ngày 02/11/2026 | Giới hạn |
|---|---|---|
| ATR PDF có sẵn trong workspace | `PaperForReferences/2026.emnlp-main.1589.pdf`; SHA256 `c009712a2082139f9b548909811b15ccd46866393477edced571cea83ef7f120` | Đọc Method và phụ lục liên quan; chưa audit toàn bộ repo tác giả |
| Squirrel PDF có sẵn trong workspace | `PaperForReferences/2601.18119v2.pdf`; SHA256 `8b16756d43a8ee1b8d40b59c3a8882bb02277ba9b4275b82c94b0e303b61484d` | Phân biệt bản arXiv v2 với metadata proceedings |
| NVIDIA | HTML arXiv v3, §6 | Không phải benchmark DataMate |
| Scientific Reports | Metadata/abstract nhà xuất bản; Method chưa lấy được | Không dùng để biện minh implementation cụ thể |
| Quy định cuộc thi | `docs/competition-context.md`, bản tổng hợp người dùng cung cấp 30/10 | Chưa thay thế tài liệu chính thức; phải đối chiếu guideline mới khi nhận |
| Rulebook thực tế | `train/train/rulebook.md`, đọc trực tiếp | Chỉ xác nhận luật **train**, không xác nhận luật public/private |
| Code và chẩn đoán | Các file ở mục 4 và `docs/inference-diagnosis-2026-11-02.md` | Snapshot hiện tại, không phải số đo cho mọi phiên bản |

Các điểm sửa cách hiểu từ trao đổi trước:

- Nguyên nhân case doanh thu bị trả `unanswerable` không còn chỉ là phỏng đoán: tài liệu chẩn đoán local đã xác nhận lỗi so ngày raw, nhãn trạng thái và locale số; chưa có kết quả batch mới để kết luận tần suất toàn hệ thống.
- “Verifier ba mức”, `RuleContext`, error taxonomy DataMate và ba route đề xuất bên dưới có nguồn cảm hứng khác nhau; toàn bộ hệ thống kết hợp **không phải một phương pháp đã được paper chứng minh nguyên vẹn**.
- Train rulebook quy định lãi gộp lấy **giá vốn ghi trên dòng bán**, không tự tính lại từ phiếu nhập; cùng tài liệu còn có định nghĩa giá vốn bình quân theo lô cho ngữ cảnh khác. Cần grounding **metric đang hỏi**, không lấy một câu trong bản tóm tắt làm công thức cho mọi chỉ số/split.
- Ingest, cache, SQL chỉ đọc và projection trực tiếp typed answer **đã có**. Đề xuất là mở rộng và kiểm chứng, không tuyên bố xây mới các phần này.

## 3. Ràng buộc không được phá

Các điều kiện cuộc thi dưới đây kế thừa bản tổng hợp và AGENTS; phải cập nhật chênh lệch nếu nhận tài liệu chính thức mới.

| Ràng buộc | Yêu cầu triển khai |
|---|---|
| Rulebook theo split | Luôn đọc `rulebook.md` của split đang chạy. Mọi rule/metric/chuẩn hóa có provenance và hash; không fallback âm thầm sang train |
| Thời gian | Neo theo `Question.as_of_date`; dùng calendar của split. Không dùng ngày máy hoặc cắt mọi loại dữ liệu sau as-of theo một luật chung |
| Ba định dạng | Đọc CSV, toàn bộ sheet XLSX liên quan và SQLite. Khám phá schema/khóa thật; giữ raw và nguồn gốc |
| Đáp án | Typed JSONL; đúng type và giá trị; `list` có thứ tự; quy tắc bảng/chuỗi/số theo tài liệu áp dụng |
| Thiếu thông tin | `unanswerable` phải có bằng chứng dữ liệu thiếu. Query lỗi, chưa grounding được hoặc hết ngân sách dẫn tới repair/`abstain`, không tự suy thành thiếu dữ liệu |
| Mục tiêu điểm | Đúng +1, sai −1, abstain 0; `unanswerable` sai vẫn −1. Local evaluator chỉ là proxy khi chưa có bộ chấm chính thức |
| Model | Tổng **tất cả model dùng trong inference** ≤15B, kể cả classifier, embedding, reranker, assistant, toàn bộ MoE; không chỉ số được load đồng thời. Quantization không giảm số tham số tính |
| Eligibility | Checkpoint trọng số công khai **trước 01/10/2026**, giấy phép phù hợp; lưu bằng chứng ngày, full revision/hash. Ngày ra tên model hoặc `main` hiện tại không đủ |
| Offline | Docker không mạng, không API, không download tài nguyên, không con người xử lý câu hỏi. Đóng gói weights/tokenizer/index dependency/wheels trước |
| Dữ liệu và output | `/data` chỉ đọc; cache/scratch ở vùng ghi khác; `/result/submission.jsonl`, `/result/logs/`; log mỗi câu tối thiểu ID, final answer, elapsed time |
| Không hard-code | ID dùng định tuyến dữ liệu, chia tập, ghép nhãn, output/log. Không dùng question ID/business ID để chọn công thức, trả đáp án hoặc sửa riêng case |
| Không rò rỉ test | Không huấn luyện, distill, fine-tune hay thu nhãn/đáp án public/private. Index/profile tạo từ input lúc inference không được tái sử dụng thành training data |
| Thời gian toàn lượt | Private test ≤6 giờ wall time, **gồm load, xác minh artifact, ingest/index, inference và ghi kết quả**; watchdog không chứng minh đã trả lời hết |

Quy tắc train thực tế dùng **câu hỏi nêu rõ → luật ngành liên quan → luật chung**; README/schema/giá trị giải thích dữ liệu. Khi câu hỏi không nêu khác, không để README tự ghi đè rulebook. Khi split mới khác hoặc quy tắc xung đột/chưa parse được, giữ trích đoạn nguồn và trạng thái chưa xác minh; không áp mặc định train.

## 4. Điểm tích hợp trong code hiện tại

Đã đọc code chính ngày 02/11/2026. Kiểm kê trực tiếp `train/train/data` thấy 107 file XLSX, 20 CSV và 1 SQLite; đây là số file, không phải số bảng hay chứng nhận chất lượng dữ liệu. Đã đọc rulebook train và header calendar, chưa profile toàn bộ cột/quan hệ trong phiên lập tài liệu. Codex phải kiểm tra lại trước thay đổi, kể cả nơi không có trong bảng.

| File có thật | Hiện có | Mở rộng đề xuất |
|---|---|---|
| `src/datamate/data/split.py` | `DatasetSplit`, load/hash rulebook, calendar bắt buộc, định tuyến business | Rule context có nguồn; liên kết cache theo đúng split |
| `src/datamate/data/ingest.py` | CSV/XLSX/SQLite → scratch SQLite, giữ raw, provenance `_dm_*`, view `__all`, cache fingerprint | Column profile, normalized views có căn cứ; không thay raw |
| `src/datamate/data/transforms.py` | `num`, `num_en`, `num_vi`, `iso_date`, `iso_date_year`, `round_half_up`, `norm` | Profile nhận biết định dạng, cảnh báo mơ hồ; không thêm parser song song vô cớ |
| `src/datamate/data/sql.py` | SQLite `mode=ro`, authorizer, query-only, timeout, giới hạn kết quả | Error code có cấu trúc, kiểm tra identifier/grounded parameters; giữ sandbox |
| `src/datamate/agent/prompts.py` | Rulebook/README/catalog vào prompt; hướng dẫn action JSON | Prompt plan/grounding ngắn, giao thức rõ; không bỏ rulebook để vừa context |
| `src/datamate/agent/runner.py` | `schema/sql/final`, feedback lỗi dạng text, `answer_from_query`, self-review cùng backend | Plan, evidence, verifier và targeted repair; giữ đường baseline |
| `src/datamate/domain.py` | `Question`, `Answer`, `RunResult`, kiểm tra type/shape và JSON hữu hạn | Contract nội bộ mới, giữ schema submission |
| `src/datamate/application.py` | Batch, identity/hash, resume, flush/fsync, deadline tính trước load model | Thêm experiment/evidence identity và telemetry; giữ tiến độ khi ngắt |
| `src/datamate/models/{registry,artifacts,backends}.py` | Inventory, pin/seal/verify model, inference offline | Kiểm tra tổng ngân sách khi thêm model; không tự thay backbone |
| `src/datamate/evaluation/{scoring,experiments}.py` | Local scoring và compare trên cùng ID | Báo cáo theo lỗi/business, paired comparison và ablation |
| `src/datamate/training/dataset.py` | Train allowlist, business holdout, lọc trajectory theo final answer | Kiểm chứng ngữ nghĩa trace/mutation trước training |
| `tools/diagnose_train.py`, notebook inference | Harness mẫu chẩn đoán; runtime/trace/report | Tận dụng bundle người dùng chạy; không sửa ngược baseline đã lưu |
| `predict.sh`, `docker/Dockerfile.competition` | Watchdog, offline build từ base image cung cấp rõ | Đóng gói mọi component mới và kiểm thử cold start đúng guideline |

Các test hiện có để đọc trước: `tests/test_agent_application.py`, `tests/test_ingestion_sql.py`, `tests/test_contracts_scoring.py`, `tests/test_train_diagnostic.py`, `tests/test_model_loading.py`, `tests/test_training_api_artifacts.py`.

**[Quan sát cần giữ trong báo cáo]** Ingest hiện có UDF rõ ràng nhưng SQL `SUM` không tự trở thành phép tính Decimal chính xác hoàn toàn. Runner hiện chấp nhận `unanswerable` sau ít nhất một query thành công, chưa chứng minh query đã kiểm tra thiếu dữ liệu đúng nghĩa. `last_query` có thể bị thay bởi query chẩn đoán; P2 cần gắn final với evidence đúng query. Các điểm này là mục tiêu kiểm chứng, không phải chứng nhận đã sửa.

## 5. Luồng và hợp đồng đề xuất

```text
Question + as_of_date + split hiện tại
  → RuleContext + raw catalog + DataProfile
  → route đơn giản hoặc QueryPlan có cấu trúc
  → metric/schema/value grounding + giải nghĩa thời gian
  → SQL local có ràng buộc → ExecutionEvidence
  → kiểm tra type, đơn vị, rule, phạm vi và nguồn
  → supported: typed answer từ đúng evidence
  → rejected/insufficient: repair có mục tiêu trong ngân sách
  → thiếu dữ liệu có bằng chứng: unanswerable
  → không đủ chắc chắn/lỗi/hết ngân sách: abstain
```

Đây là thiết kế DataMate để thử, không phải sơ đồ nguyên bản của một paper.

### Hợp đồng nội bộ tối thiểu

Tên và nơi đặt là đề xuất; ưu tiên tận dụng dataclass/giao diện hiện tại, không thêm framework nếu không cần.

| Contract | Trường/ý nghĩa tối thiểu |
|---|---|
| `RuleContext` | split/rulebook/calendar hash; rule refs gồm trích đoạn/heading; metric definitions, precedence, rounding, time/ranking rules; `verified/unknown/conflict` |
| `DataProfile` | nguồn file/sheet/cột; raw/native type; parse candidates và tỉ lệ lỗi; date range, null, distinct/status values; grain/khóa/join đã kiểm tra; giới hạn sample và coverage |
| `QueryPlan` | intent, expected answer type, metric refs, entities, filters, grain, đơn vị, time window từ as-of, ordering/tie-break, dependencies, unresolved items |
| `GroundedPlan` | bảng/cột thật, biểu thức/chuyển đổi có căn cứ; canonical value/code và raw value; parameters; evidence refs cho lựa chọn |
| `ExecutionEvidence` | `query_ref`, SQL + parameters, plan/rule/profile hashes, nguồn dữ liệu, kết quả đầy đủ/truncated, phạm vi thời gian, đơn vị, elapsed; liên kết về đáp án |
| `ValidationVerdict` | `SUPPORTED/INSUFFICIENT/REJECTED`, error class, checks đã chạy, evidence refs, hành động sửa được phép; không coi verdict SLM là bằng chứng độc lập |
| `FailureRecord` | route, stage, error class, query fingerprint + parameters, evidence version, thay đổi đã thử; cấm lặp không tiến bộ |
| `Budget` | deadline tổng/câu/SQL, tokens, model calls, tool calls, repair count; mọi branch dùng cùng ngân sách |

Không cần mọi câu đều gọi model để lập plan. Câu đơn giản có thể dùng plan tối thiểu từ route hiện có; câu phức tạp mới decomposition. Không tự chọn ngưỡng “bán chậm/tồn nhiều” nếu câu hỏi/rulebook chưa định nghĩa. Quy tắc không ánh xạ được phải giữ nguyên văn trong context, không bị loại chỉ vì parser chưa hỗ trợ.

`query_ref` và metadata mới thuộc log/nội bộ, không thêm field tùy ý vào submission. Tiền/tỷ lệ/phép tổng hợp do SQL hoặc hàm tính xác định đã kiểm chứng thực hiện; model chọn cách tính nhưng không sửa số sau executor.

## 6. Backlog P0–P4

Thứ tự phụ thuộc: **baseline snapshot → P0 → P1 → P2 → P3**. P4 bắt đầu phần mutation/test khi đã có query train được xác minh; training chờ pipeline ổn định. Không có lịch ngày cứng hay ngân sách GPU được phê duyệt trong tài liệu này.

### P0 — Bảo toàn baseline, dữ liệu, rulebook và typed output

**Mục tiêu:** có chuẩn so sánh tái lập; xử lý đúng nền dữ liệu trước khi thêm reasoning. Nguồn: yêu cầu DataMate và lỗi workspace, không phải thuật toán gốc ATR.

- [ ] **P0.0 — Freeze baseline thực tế.** Lưu source commit và diff/untracked cần thiết; config hiệu lực, prompt/template/tokenizer hash, model revision/manifest, quantization/seed/library/hardware, split/data/rulebook/calendar hash, ID câu hỏi, prediction/log/timing. Lưu cả notebook có output và SHA256 của bundle. Không chỉ ghi tên “Gemma 12B”.
- [ ] **P0.1 — Đọc bundle baseline trước khi chọn lỗi.** Phân loại lỗi theo trace và trả kết luận có mức chắc chắn. Mẫu stress 72 thuộc development, 3 câu/business trên 24 business; không đại diện phân bố toàn train. Harness đang dùng model đã load nên thời gian mẫu không phải cold start Docker.
- [ ] **P0.2 — Profile dữ liệu có giới hạn.** Thống kê cột ngày/số, locale, trạng thái thật, null/parse failure, distinct, khóa và cardinality. Sample chỉ gợi ý; kiểm tra toàn cột/quan hệ liên quan trước quyết định ảnh hưởng đáp án. Cache theo source và version transform; đo chi phí build.
- [ ] **P0.3 — Normalized views có provenance.** Giữ raw bất biến. Xử lý thiếu năm theo sheet/README; số theo locale có căn cứ. Forward-fill trong đúng nhóm/khóa, dedup theo quy tắc split và grain, không toàn bảng. Không cộng cả `__all` và sheet thành phần; không tự xem công thức XLSX không có cache là 0.
- [ ] **P0.4 — RuleContext và bộ giải thời gian.** Đọc split hiện tại, giữ toàn văn và refs. Giải nghĩa kỳ theo as-of, tuần/tháng/so sánh kỳ/Tết theo calendar. Không dùng registry luật keyed theo business ID. Rules mới/chưa hiểu không tự kế thừa train.
- [ ] **P0.5 — Giao thức và serializer.** Đo lỗi một JSON action; kiểm tra template/stop tokens thật trước sửa. Nếu backend hỗ trợ JSON/schema constraints thì thử riêng. Không chọn tùy ý một trong nhiều action. Giữ `Answer`, direct query projection, finite numbers và JSONL flush/resume.

**Đầu ra review:** snapshot/manifest B0; báo cáo lỗi baseline; profile mẫu có giới hạn rõ; mô tả rule/transform provenance; config riêng cho từng thay đổi. Tên thư mục mới gợi ý: `outputs/research/p0-*` và `result/research/p0-*`.

**Nghiệm thu:** T00–T04, T08–T12 ở mục 7, cùng test hiện có bị tác động. Khi tắt cờ, đường baseline vẫn dùng prompt/config cũ. Không gọi P0 hoàn tất nếu chỉ có smoke bằng backend giả mà chưa đánh giá hành vi thật cần đo. Chỉ đưa sửa format/normalization vào bản kế tiếp sau ablation tương ứng.

### P1 — Structured planner, metric grounding và schema/value grounding

**Mục tiêu:** query phản ánh đúng chỉ số, cấp dữ liệu và giá trị thật. Nguồn: ATR-inspired decomposition/grounding; business metric/time layer là đề xuất DataMate.

- [ ] **P1.1 — Plan có contract và validation.** Dùng `QueryPlan`/`GroundedPlan` tối thiểu. Mỗi metric có công thức, grain, bảng/cột, đơn vị, status/date filters, aggregation, rounding, output type/order và rule refs. Không chỉ đặt nhãn `revenue` rồi để SQL tự diễn giải lại.
- [ ] **P1.2 — Metric grounding.** Phân biệt doanh thu với thu tiền, số đơn với số dòng/số lượng, tồn snapshot với phát sinh kỳ, giá vốn/lãi gộp với tiền nhập, tỷ trọng với tăng trưởng. Ánh xạ schema theo README + giá trị + quan hệ quan sát; kiểm tra join fan-out và tổng lặp cấp đơn.
- [ ] **P1.3 — Ground schema trước SQL.** Đề xuất bảng/cột từ catalog thật, xác minh tồn tại và ý nghĩa. Không loại mất bảng liên quan chỉ vì top-k hẹp. Nếu shortlist thiếu, có một bước mở rộng catalog có giới hạn và ghi log.
- [ ] **P1.4 — Ground values.** Thử exact code → normalized text giữ dấu → fuzzy candidates. Giữ raw canonical value để truyền qua SQL parameters. Không gộp hai mã cùng tên; fuzzy chỉ sinh ứng viên, không tự quyết chúng là một thực thể.
- [ ] **P1.5 — Thử embedding riêng khi có lỗi lexical còn lại.** Candidate top-k/coverage/latency được đo; model local đủ điều kiện, khai báo tham số và index hash. Không mặc định dùng cả BGE và reranker như ATR.
- [ ] **P1.6 — Decompose có chọn lọc.** Câu so sánh kỳ/tỷ lệ/nhiều metric dùng subplan có dependency và unit/grain; cấp query sau nhận kết quả có nguồn. Số cuối vẫn tính bằng executor. Hạn chế số subplan, không tự nhân ngân sách 10 bước cho từng subplan.

**Điểm tích hợp:** mở rộng agent/data/domain hiện tại; module mới *có thể* là `agent/planning.py`, `data/profiling.py`, `data/grounding.py`, `data/rules.py`, chỉ tạo nếu ranh giới thật sự cần. Tái dùng parser/UDF hiện có.

**Đầu ra review:** schema contract; vài plan/evidence train được kiểm chứng; báo cáo coverage của candidate retrieval và các case không giải quyết được; cờ bật riêng planner/metric/schema/value/decomposition.

**Nghiệm thu:** T02–T07, T09, T11–T13. Candidate không có thật bị chặn trước execution; ambiguity không biến thành câu trả lời chắc chắn. Đo riêng giá trị của planner và từng grounding component; không tuyên bố tất cả cải thiện từ ATR.

### P2 — Error-aware controller, evidence verifier và targeted repair

**Mục tiêu:** sửa đúng tầng lỗi, tránh vòng lặp và false-unanswerable. Nguồn: ATR failed-route history/verifier + Squirrel error construction; controller/taxonomy dưới đây là đề xuất mới.

- [ ] **P2.1 — Phản hồi lỗi có cấu trúc.** Phân biệt `ACTION_JSON`, `SQL_SYNTAX`, `SCHEMA`, `PARSE_DATE`, `PARSE_NUMBER`, `VALUE_LINK`, `METRIC_RULE`, `JOIN_GRAIN`, `EMPTY_RESULT`, `TRUNCATED_RESULT`, `OUTPUT_TYPE`, `TIMEOUT`, `MISSING_DATA`. Giữ message gốc và evidence; không suy nguyên nhân chỉ từ một regex/stacktrace mơ hồ.
- [ ] **P2.2 — Chính sách sửa theo lỗi.** Syntax/action → sửa cú pháp/giao thức; unknown column → mở schema; parse → kiểm tra profile/arity/locale; value → xem ứng viên thật; metric/join → sửa plan và rule refs. Không thử lại toàn bộ prompt như cùng một cách cho mọi lỗi.
- [ ] **P2.3 — Verifier xác định trước, SLM tùy chọn sau.** Kiểm tra query_ref, type/shape, truncated, đơn vị, thời gian, rule application, grain và join multiplicity. Dùng kiểm tra độc lập nhỏ khi cần. SLM review chỉ chạy cho trường hợp chưa giải quyết; cùng model tự đồng ý lần hai không chứng minh đúng.
- [ ] **P2.4 — Guard cho kết quả rỗng/NULL/0.** Đếm raw/parsed rows và phạm vi dữ liệu; kiểm tra date filters, value candidates và null propagation. Kỳ trong coverage nhưng không phát sinh có thể có đáp án 0; kỳ hoàn toàn ngoài coverage hoặc thiếu dữ liệu bắt buộc chỉ thành unanswerable theo rulebook hiện tại. Profile từ sample không đủ chứng minh cả dữ liệu không có trường/thực thể.
- [ ] **P2.5 — Final gắn đúng evidence.** Query diagnostic không được vô tình thay query trả đáp án. Candidate final tham chiếu query/evidence đã được kiểm tra; thay plan/filter/source làm evidence cũ không còn hợp lệ. Từ chối final numeric tự viết lại số hoặc query bị cắt.
- [ ] **P2.6 — Chống lặp và ngân sách.** Lưu fingerprint query gồm parameters và evidence version. Chặn cùng giả định/query lỗi khi không có thông tin mới; cho phép cùng route SQL sau sửa có mục tiêu. Thử ngân sách khởi điểm 1 lần sinh + tối đa 1–2 repair, so với ngân sách cũ trong cùng trần thời gian/calls; đây là hyperparameter đề xuất.
- [ ] **P2.7 — Kết thúc trung thực.** `SUPPORTED` → serialize evidence; thiếu dữ liệu được chứng minh → unanswerable; còn mơ hồ/hết trần → abstain. Log lý do, không ghi “missing data” cho lỗi engine, linker hoặc giới hạn context.

**Đầu ra review:** state transitions/controller, failure/verdict schema, trace trước/sau trên lỗi thật, repair regression suite và báo cáo chi phí từng nhánh. Có thể tách `agent/controller.py`, `agent/verification.py` nếu phù hợp code đã đọc.

**Nghiệm thu:** T05–T10, T12–T13. Query chạy thành công chưa đủ để final/unanswerable. Sửa lỗi không được âm thầm bỏ filter bắt buộc, đổi metric, bỏ dấu âm hay thay kỳ để có kết quả. Repair phải vượt baseline retry trong so sánh ngân sách công bằng; nếu không, giữ cơ chế đơn giản hơn.

### P3 — Lightweight routing, selective execution và early stopping

**Mục tiêu:** giảm model calls/token/thời gian với chất lượng được kiểm tra. Nguồn: ATR lightweight router/stop và NVIDIA specialization; route/gate DataMate là đề xuất.

- [ ] **P3.1 — Heuristic router trước.** Thử `DIRECT_SQL` cho câu rõ, `GROUND_AND_SQL` khi cần ánh xạ metric/entity/values, `DOC_LOOKUP` để lấy định nghĩa trong README/rulebook offline. `DOC_LOOKUP` là tên tránh nhầm với `RETRIEVE` đọc ô/hàng của ATR; thường là bước hỗ trợ rồi quay về query.
- [ ] **P3.2 — Không dùng router để đoán đáp án.** Quyết định dựa vào question, schema/profile, rule ambiguity và lịch sử lỗi; không theo ID. Heuristic chưa chắc → full grounded path. Route đơn giản vẫn phải qua rule/time/output checks.
- [ ] **P3.3 — Dừng khi evidence đủ.** Scalar đã kiểm tra không cần một model call viết lại kết quả. Bỏ self-review có chọn lọc, đo riêng với always-review. Không dùng “model tự tin” làm điều kiện duy nhất; confidence nếu dùng phải hiệu chỉnh trên development và khóa trước holdout.
- [ ] **P3.4 — Chi phí khởi tạo và cache.** Đo cold/warm; tái dùng DB/profile/index theo business và source hash. Bất kỳ cache plan/kết quả phụ thuộc thời gian phải có as_of_date, question, rule/evidence identity phù hợp. Không cache đáp án theo ID hoặc mang đáp án test sang lần khác làm dữ liệu học.
- [ ] **P3.5 — Classifier/router nhỏ chỉ khi có cơ sở.** Tạo route labels từ trace train được review; đánh giá cả quyết định và PA cuối, không chỉ agreement với teacher. So heuristic, LLM-router và classifier trên cùng budget. Kiểm tra eligibility/tham số, tokenizer và vận hành offline trước thêm model.
- [ ] **P3.6 — Chốt runtime Docker.** Chạy end-to-end với weights thật, `--network none`, `/data` chỉ đọc, output/log đúng path; tính từ cold start. Kiểm tra base image theo guideline chính thức đã nhận, không suy ra từ hardware được mô tả.

**Giới hạn `DOC_LOOKUP`:** baseline hiện xuất đáp án có giá trị từ SQL. Không mở một đường final tự do từ văn bản trong cùng thay đổi routing. Nếu thực sự cần trả lời trực tiếp từ tài liệu, tạo ticket riêng cho source/span provenance, output contract và regression; không dùng `SELECT` một literal do model bịa để vượt guard.

**Đầu ra review:** routing reasons, per-route metrics, early-stop ablation, runtime/VRAM/RAM/index build report, Docker manifest. Không bắt buộc huấn luyện router nếu heuristic đã đạt mục tiêu.

**Nghiệm thu:** T00, T02–T03, T07–T13; efficiency gate ở mục 9. Chưa có dữ liệu private thì rehearsal chỉ chứng minh tập đã chạy; ghi rõ cơ sở ước lượng 6 giờ và khoảng dự phòng, không chứng nhận private hoàn tất.

### P4 — Synthetic SQL error suite và SFT tùy chọn

**Mục tiêu:** đo năng lực sửa lỗi và chỉ training khi phần dữ liệu/tool đã ổn. Nguồn: Squirrel minimal-change bugs/SFT; NVIDIA trace curation. Đây không phải cam kết fine-tune sẽ tăng PA.

- [ ] **P4.1 — Bộ reference query train được xác minh.** Train có answers nhưng không có gold SQL. Query khớp final answer chỉ là ứng viên: review metric/rule/join/time/unit, chạy witness fixtures và lưu provenance. Không suy gold SQL ngược từ một số đúng rồi mặc định ground truth.
- [ ] **P4.2 — Mutation từng lỗi.** Tạo cặp query đúng/sai bằng một thay đổi tối thiểu: filter trạng thái, date boundary/as-of, locale, `COUNT(*)`/`DISTINCT`, sum cấp đơn, join duplicate, unit conversion, numerator/denominator, order/tie-break, value mismatch. Ghi operator mutation, lỗi kỳ vọng, vị trí diff, nguồn và witness.
- [ ] **P4.3 — Kiểm chứng lỗi.** Syntax mutant phải phát lỗi dự kiến. Semantic mutant phải làm khác typed answer trên dữ liệu/witness hoặc có vi phạm rule cụ thể được kiểm chứng. Mutant không có tác động/chưa phân biệt được phải loại hoặc cách ly; không báo repair success trên các bài giả.
- [ ] **P4.4 — Tách tập trước khi sinh biến thể.** Reference query, mọi mutant, paraphrase và trajectory của cùng business chỉ thuộc một partition. Dữ liệu synthetic tạo từ business development; giữ holdout và mutation families dành cho đánh giá ngoài phân bố riêng. Chống near-duplicate leakage.
- [ ] **P4.5 — Dùng làm test trước khi SFT.** Báo repair accuracy cuối, fix regression, loại lỗi, calls/time, PA trên câu tự nhiên. AST/SQL EM/Modify Better là số phụ; nếu chỉ dùng sqlglot/diff đơn giản thì không gọi là tái lập Graph Match Calcite của Squirrel.
- [ ] **P4.6 — SFT có điều kiện.** Chọn một tác vụ còn lỗi lặp lại: action format, planner, value linking hoặc SQL repair. Format warm-up hiện có không dạy metric/SQL. Chỉ dùng train allowlist, synthetic được phép và trace đã kiểm chứng; không thu trace/nhãn test để training. Teacher offline dùng backbone hiện có trước khi cân nhắc model khác.
- [ ] **P4.7 — So base/SFT công bằng.** Giữ pipeline, prompt, budget, split, precision và decoding; đổi riêng trọng số. Nếu thử DM-SFT/Diff-SFT, ghi đúng cơ chế và kiểm tra áp patch không đổi đoạn ngoài phạm vi. Không thêm một model repair riêng nếu fine-tune backbone hiện có đã đủ.
- [ ] **P4.8 — Artifact và rollback.** Lưu base revision/eligibility, adapter/merged hash, source license/provenance, training config/version/seed, data hash và eval report. Kiểm tra inventory cuối, Docker offline và thời gian load; giữ B0 và bản pipeline tốt nhất chưa SFT.

**Đầu ra review:** error suite có witness; dataset card/chống leakage; script đánh giá; nếu thực sự train thì có training/eval/manifest. Không có GPU run thì ghi “chưa train”, không đánh dấu ticket training hoàn tất.

**Nghiệm thu:** T01–T13 phù hợp phạm vi; hiệu quả trên business chưa thấy và câu hỏi tự nhiên, không chỉ loss thấp hoặc SQL rất giống reference. Không đạt gate thì giữ pipeline không SFT. Hướng Scientific Reports chỉ mở ticket sau khi đọc được Method và chứng minh liên quan trực tiếp.

## 7. Acceptance tests bắt buộc theo rủi ro thay đổi

Các case dưới đây là **đặc tả test tương lai**, không phải kết quả đã chạy. Dùng fixture tổng hợp nhỏ có oracle tính độc lập và case train được phép. Không lấy nhãn public/private làm oracle. Unit test/backend giả xác minh controller, không đại diện chất lượng LLM thật.

| ID | Case phải phân biệt | Kết quả cần đạt |
|---|---|---|
| T00 | Tắt toàn bộ cờ mới; replay scripted backend/cùng fixture; config/rule/model hash đổi khi resume | Đường baseline giữ contract/prediction; không ghi đè artifact; resume khác identity bị từ chối. Replay LLM chỉ so trong môi trường đã khóa, không hứa bit-identical xuyên GPU |
| T01 | CSV UTF-8/BOM/delimiter; XLSX nhiều sheet/header/ô formula thiếu cache; SQLite; tên/mã có số 0 đầu; duplicate/group blanks | Ba loader giữ raw/native type/provenance; source hash trước/sau không đổi; transformation có căn cứ; không double-count sheet, ffill toàn bảng, dedup bảng không theo đơn |
| T02 | Đổi ngày hệ thống nhưng giữ as-of; tuần/tháng/vắt năm/năm nhuận; so tháng trước bị thiếu ngày; ngày thiếu năm; Tết | Kết quả chỉ theo as_of_date + rule/calendar hiện tại. Biên ngày bao gồm/loại trừ đúng rule; không lấy năm máy để parse. Dữ liệu tương lai xử lý theo rule từng metric, kể cả ngoại lệ trạng thái trả hàng của train |
| T03 | Hai split fixture cùng schema nhưng khác rulebook/calendar; sửa rule trong cùng đường dẫn | Plan/query/answer đổi đúng luật và cache invalidated. Không dùng rule train cũ; parser không hiểu phải báo unknown và giữ source, không tự điền default |
| T04 | Doanh thu/thu tiền/công nợ; đơn đang giao/hủy/trả một phần; VAT/voucher/phí; lãi gộp/giá vốn; quy đổi đơn vị | Oracle theo rule fixture. Kiểm tra grain, phân bổ cấp dòng/đơn và join 1-n; số/rounding đúng. Fixture train có lãi gộp dùng cost trên dòng, fixture rule khác phải cho công thức khác khi được quy định |
| T05 | Entity có viết tắt/không dấu; hai mã cùng tên; nhiều tên gần nhau; literal không tồn tại; mã exact | Candidate tồn tại trong đúng bảng/cột; raw value/code được giữ; không gộp nhầm thực thể hoặc tự bỏ filter. No-match/ambiguity không tự thành missing data |
| T06 | Sai JSON, nhiều action, sai key/type, `iso_date` sai arity, cột bịa, parameters có quote | Feedback đúng class; chỉ một action hợp lệ; identifier được kiểm tra và value qua parameters. Không chọn ngẫu nhiên action, không nới sandbox để chạy được |
| T07 | Syntax repair, schema reground, locale/date repair, semantic replan; lặp query/giả định; query chẩn đoán sau query đáp án | Mỗi hành động có lý do/evidence mới; không lặp quá budget. Same SQL route vẫn sửa được. Final tham chiếu đúng query_ref; stale/truncated evidence bị chặn |
| T08 | Trong kỳ không giao dịch; kỳ hoàn toàn ngoài coverage; query rỗng do sai parse/value; cột thiếu; SUM NULL | Phân biệt 0, unanswerable và abstain đúng rule fixture. Không đánh đồng NULL/0/empty hoặc query lỗi. Unanswerable có missing-evidence refs, false-unanswerable được chấm −1 |
| T09 | Đủ 9 type; bool khác 0/1; percent khác fraction; tháng/ngày ISO; NaN/Inf; list đảo thứ tự; table đảo hàng/cột | JSONL parse được, đúng shape/type, finite JSON numbers, phần trăm theo đơn vị %. List theo câu hỏi/rule và tie-break; table theo matching spec; scalar không qua LLM sửa số |
| T10 | Timeout từng SQL/model/câu/batch; truncated rows; context không đủ; kill rồi resume | Dừng trong budget, giữ dòng đã flush/log hợp lệ, không trùng ID; partial ghi đúng. Không âm thầm cắt rulebook, coi mẫu bị cắt là toàn bộ hay kéo dài deadline bằng repair |
| T11 | `--network none`; `/data` read-only; weights thật local; cache trống; thiếu resource; query ghi/attach/readfile/load_extension | Chạy không mạng với artifact đã đóng gói; không ghi dataset; mutation/file access bị chặn; thiếu artifact báo lỗi rõ, không tải hoặc chuyển sang model/demo khác. Đo cold start toàn lượt |
| T12 | Business holdout, allowlist fingerprint; đưa test vào training; đổi ID fixture nhưng giữ dữ liệu tương ứng; log final/time | Holdout không vào training/teacher/threshold tuning; nguồn test bị từ chối dù đổi tên. ID chỉ định tuyến/ghép; không quyết định công thức/đáp án. Log có tối thiểu ID/final/elapsed |
| T13 | Thêm classifier/embedding/reranker/assistant; checkpoint không có bằng chứng cutoff; mutation vô tác dụng | Inventory chặn >15B kể cả load tuần tự/MoE; reject model chưa đủ eligibility/license evidence. Mutant không có witness bị cách ly; evaluator không nhận AST match là đủ đúng business answer |

Ví dụ test thời gian có mốc cụ thể phải ghi đó là **rule fixture** hoặc **train rule**, không đưa Monday/MTD/công thức lãi gộp thành chân lý cho mọi split. Nếu tài liệu chấm chính thức khác local evaluator, lưu phiên bản cũ và báo chênh lệch; không sửa scorer để làm điểm candidate tăng.

## 8. Chia tập và kế hoạch ablation

**Ablation** là thay/bỏ một thành phần để đo đóng góp của nó. Cùng câu hỏi, data/rulebook, checkpoint, prompt nền, GPU, decoding và trần ngân sách; chỉ khác yếu tố được nêu. Khi dependency bắt buộc khiến không thể tháo độc lập, ghi rõ và dùng biến thể tối thiểu tương đương, không diễn giải như một thí nghiệm độc lập hoàn hảo.

### 8.1 Tập phát triển và holdout

- Giữ cách chia theo business có trong `docs/training.md`/manifest hiện hành. Snapshot hiện tại: 24 business development (2.370 câu), 6 business holdout (630 câu), seed 42. Phải kiểm tra lại file thật và xác nhận holdout chưa từng dùng tuning/training trước khi gọi nó là untouched.
- Mẫu stress 72 là tập chẩn đoán development đã chọn; đủ để tìm lỗi, không đủ để suy ra điểm thi. Không tự biến holdout thành tập huấn luyện bằng cách lấy trace hoặc nhãn route của nó.
- Dùng mẫu development cố định để sàng lọc; mở rộng development khi cần đo nhóm lỗi chưa phủ. Sau khi chốt thiết kế/ngưỡng trên development mới đánh giá shortlist trên holdout. Nếu đã xem holdout nhiều lần để sửa model/prompt, ghi rõ và dùng business-grouped cross-validation trong phần train cho vòng sau; không tiếp tục gọi holdout là độc lập.
- Chia theo ngành nếu metadata thực tế cho phép; báo các ngành/schema chưa thấy riêng. Chia ngẫu nhiên theo question không chứng minh tổng quát hóa sang doanh nghiệp mới.
- Public/private chỉ dùng luồng inference đúng quy định; không xây gold/teacher trace/train corpus hoặc sửa lời giải theo đáp án test. Không chọn nhiều cấu hình bằng cách thu nhãn test.

### 8.2 Ma trận thử nghiệm

| Run | Khác biệt được đo | So sánh chính |
|---|---|---|
| B0 | Baseline người dùng thực sự chạy, đóng băng đầy đủ | Mốc không ghi đè; thiếu mẫu/kết quả thì báo rõ |
| I0 | Chỉ thêm telemetry, không đổi context/decoding/control | I0 vs B0 để kiểm tra phép đo không đổi hành vi |
| D1 | Column profiling/normalized view | D1 vs I0; không trộn thêm router/model mới |
| R1 | RuleContext/time resolver | R1 vs cùng bản nền; tách ảnh hưởng normalization |
| J1 | Protocol/JSON constraints nếu backend hỗ trợ | Parse-error và PA so cùng prompt/budget đã ghi |
| M1 | Planner + metric grounding | So bản P0 đã chọn, không thêm value retrieval cùng lúc |
| S1 / V1 | Schema grounding / lexical value grounding | Thêm riêng rồi kết hợp; đo candidate coverage và PA |
| E1 | Error-aware repair + evidence checks | So retry baseline trên cùng budget; thêm số recovery và regression |
| H1 | Heuristic routing | So always grounded path; LLM-router là đối chứng tùy nhu cầu |
| C1 | Classifier router tùy chọn | So H1 với chi phí load/infer và tổng tham số đầy đủ |
| Q1 | Early stop/selective review | So không dừng sớm hoặc always-review trong trần giống nhau |
| V2 | Fuzzy vs lexical, rồi embedding tùy chọn | Đo lợi ích từng tầng, không tự thêm reranker |
| F1 | Cấu hình kết hợp tốt nhất và leave-one-out | Lần lượt bỏ metric/value/repair/routing để đo đóng góp có tương tác |
| FT1 | SFT một tác vụ trên pipeline F1 giữ nguyên | So base vs fine-tuned; không đồng thời đổi model size, tools và budget |
| FT2 | Thêm synthetic error data vào cùng recipe SFT | So chỉ trace đã xác minh vs trace + mutation; giữ tập eval tách biệt |

Đây là run IDs đề xuất, chưa được chạy. Không cần thử toàn ma trận nếu lỗi/budget không biện minh. Khi so repair hoặc decomposition, báo cả trần calls lẫn calls thực dùng: tăng PA nhờ tăng compute phải được ghi riêng. Khi so retrieval, tính cả thời gian xây/load index và recall của ứng viên, không chỉ thời gian model generate.

## 9. Metric, cổng quyết định và bảo toàn kết quả

### 9.1 Báo cáo tối thiểu mỗi run

- `N_expected`, `N_completed`, trạng thái complete/partial; missing counts như abstain trên **toàn tập đã khóa**, không chỉ chấm câu hoàn tất. `PA = (correct - wrong) / N_expected`.
- Correct/wrong/abstain, accuracy trên toàn tập (`correct/N`), coverage (`(correct+wrong)/N`) và accuracy trên câu đã trả lời (`correct/(correct+wrong)`). Không chỉ báo accuracy trên các câu agent chọn trả lời.
- Theo type/business/nhóm câu: type mismatch, false-unanswerable trên gold answerable, khả năng nhận unanswerable đúng, sai ngày/locale/metric/value/join/ranking; nhóm có thể chồng lấp, không cộng chúng thành tổng.
- Tỉ lệ JSON/action lỗi, SQL syntax/schema/runtime errors, empty/truncated, timeout/context errors; repair recovery và **repair regression** (sửa làm mất đáp án đúng), routing distribution/early stops.
- Model/tool calls, input/output tokens nếu backend đo được; phần không đo được ghi unavailable, không đoán. Thời gian p50/p95/câu, tổng wall time, cold load/hash/ingest/index time, peak VRAM/RAM và dung lượng artifact/index.
- Paired per-question delta so B0 và bản được giữ gần nhất; liệt kê câu được cứu/câu bị hỏng và nguyên nhân review trên train. Ước lượng độ bất định theo business (ví dụ grouped bootstrap với seed cố định), báo rõ chỉ có 6 business holdout nên khoảng ước lượng có thể rộng.

Một câu trong mẫu 72 tương đương `1/72` điểm PA khi wrong→abstain và `2/72` khi wrong→correct. Không gọi chênh lệch nhỏ trên mẫu này là bằng chứng chắc chắn cho private test.

### 9.2 Cổng giữ hoặc bỏ thay đổi

Các ngưỡng sau là **đề xuất kỹ thuật cần khóa trước khi chạy**, không phải quy định cuộc thi hoặc số liệu từ paper. Nếu thay ngưỡng, ghi lý do trước khi xem kết quả candidate.

1. **Correctness/compliance:** mọi acceptance test bắt buộc trong phạm vi phải pass; không vi phạm offline, nguồn dữ liệu, eligibility/tham số, typed output hay baseline preservation. Không bù vi phạm bằng PA cao.
2. **Tính năng tăng chất lượng:** cần cải thiện PA development và không có regression bắt buộc chưa giải thích. Đánh giá finalist trên holdout: PA không thấp hơn B0 và bản pipeline đang giữ; báo uncertainty và nhóm lỗi. Nếu bằng chứng không đủ, để trạng thái experimental/off, không tuyên bố thành công.
3. **Tính năng tăng tốc P3:** mục tiêu thử là giảm ít nhất **15% wall time end-to-end** trên cùng workload, PA holdout không giảm quá **0,01 tuyệt đối** so bản nền đã chọn; p95 và timeout rate phải được báo. Nếu PA giảm trong khoảng cho phép vẫn ghi là trade-off, không nói “giữ nguyên độ chính xác”. Xem cả paired losses và CI trước promote; một mẫu nhỏ không chứng minh non-inferiority.
4. **Abstention:** chọn threshold bằng PA và coverage/error curve trên development. Không dùng self-reported confidence chưa hiệu chỉnh như xác suất đúng. Luôn giữ riêng missing-data verdict và quyết định không trả lời.
5. **Runtime cuối:** rehearsal cold start với weights thật phải xong workload đã công bố dưới 6 giờ, có dự phòng và log đầy đủ. Với private input chưa có, ghi rõ N/kích thước/ước lượng và điều chưa biết; watchdog hoặc demo 3 abstain không đủ nghiệm thu.
6. **SFT:** vượt bản base cùng pipeline trên business chưa thấy/câu tự nhiên và không làm format/runtime/regression tệ ngoài gate; training loss hoặc thành tích mutation suite riêng không đủ. Không đạt thì dùng bản không SFT.

### 9.3 Artifact của một thí nghiệm

Gợi ý cấu trúc tương lai; dùng lại tên file harness hiện có nếu đã đủ, tránh duy trì hai hệ thống manifest khác nhau:

```text
outputs/research/<run-name>/
  manifest.json              # code/config/prompt/model/data/rule/calendar hashes
  experiment_plan.md         # giả thuyết, cờ thay đổi, tập câu, budget, gate đã khóa
  selected_questions.jsonl   # tập cố định, không nhãn trong inference prompt
  runtime.json               # thiết bị, versions, seed, precision, template
  metrics.json
  comparison.json
  review.csv                 # chấm sau inference, chỉ train/holdout được phép
  report.md                  # actual results, partial status, limitations, decision

result/research/<run-name>/
  submission.jsonl
  logs/                      # final + timing + SQL/plan/evidence/failures
  run_identity.json
```

Không ghi đè run cũ. Nếu giới hạn thời gian làm partial, lưu numerator/denominator và số câu thiếu. Cùng một run chỉ resume khi identity phù hợp; thay code/config/model/data phải tạo run mới. P0 B0 và pipeline tốt nhất luôn có bản hash kiểm tra được để rollback. Hash các file weights một lần đúng vòng đời đã thiết kế; tối ưu startup không được bỏ xác minh tính toàn vẹn.

## 10. Handoff cho một phiên triển khai tương lai

Có thể dùng yêu cầu sau khi đã có bundle baseline, thay phần trong ngoặc bằng artifact thật:

```text
Đọc AGENTS.md và DATAMATE_AGENT_RESEARCH_ROADMAP.md.
Baseline: [đường dẫn bundle/log/manifest thực tế và trạng thái complete/partial].
Triển khai ticket: [ví dụ P0.2 + P0.3 cho lỗi parse ngày/locale đã xác minh].

Trước khi sửa, đọc code/call site/test/config/notebook và rulebook/dữ liệu liên quan.
Tóm tắt phần đã có và lỗi có bằng chứng; giữ nguyên B0, model checkpoint và tập câu.
Chỉ thay trong phạm vi ticket, có cờ tắt và output directory mới.
Chạy acceptance tests phù hợp, rồi ablation được giao trên cùng budget.
Không dùng public/private cho training/thu nhãn, không hard-code theo ID.
Báo PA/type/value errors, false-unanswerable, coverage, latency/calls,
artifact hashes, kết quả thực chạy và phần chưa kiểm chứng; giữ hoặc rollback theo gate.
```

Checklist kết thúc ticket:

- [ ] Đã đọc code và nguồn dữ liệu thật; ghi đúng những gì quan sát được.
- [ ] Phân biệt nguồn paper, adaptation và giả thuyết; không gắn kết quả chưa chạy vào paper/agent.
- [ ] Baseline và output cũ còn nguyên; tắt cờ quay được về đường cũ.
- [ ] Kiểm tra regression có oracle độc lập, không chỉ mirror implementation.
- [ ] Tập/cấu hình/budget/threshold đã khóa, so sánh cùng điều kiện; không leak holdout/test.
- [ ] Rulebook/calendar của split và as_of_date đi xuyên suốt tới query/verdict.
- [ ] Typed answer gắn evidence đúng; unanswerable không thay cho lỗi hoặc hết ngân sách.
- [ ] Model inventory/eligibility/offline/Docker đã kiểm tra đúng mức thay đổi.
- [ ] Report ghi complete/partial, PASS/FAIL thực tế, rủi ro còn lại và quyết định promote/off/rollback.

**Trạng thái bàn giao tài liệu:** backlog đã viết; mọi thay đổi agent, ablation GPU, full Docker rehearsal và SFT mô tả ở đây vẫn là công việc tương lai, chỉ bắt đầu theo yêu cầu triển khai tiếp theo của người dùng.
