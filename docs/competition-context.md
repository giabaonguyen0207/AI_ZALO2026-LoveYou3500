# Zalo AI Challenge 2026 — DataMate: Competition Context

## Nguồn và trạng thái hiện tại

Cập nhật tiếp15/11: chủ dự án xác nhận giữ interface/library versions nhưng cho phép thay pipeline và yêu cầu tích hợp correction. Đã thêm public checkpoint yammdd correction (pinned revision/hash, conservative≤400M), hybrid retrieval, profile optimized worker2/prefix/bounded recovery; config cũ làm baseline. Tài liệu hiện tại ở `zlai/pipeline_tichhop_2026-11-15.md`. Chưa đo GPU/PA; không coi test CPU là chứng nhận model runtime. Yêu cầu này thay cách diễn giải trước đó rằng phải giữ mọi cấu hình inference.

Cập nhật từ chủ dự án **15/11/2026**: public dùng Colab A100; solution Docker dùng 2×RTX5090, tải weights trước trên Vast.ai. Đội đã thử Docker qua cổng BTC và chưa gặp lỗi môi trường theo chủ dự án. Giữ interface nộp bài, cấu hình inference mặc định và dependency versions; có thể xem xét đổi model riêng sau, hiện chưa đổi. Chủ dự án chưa giữ image/digest/URL weights cũ và yêu cầu cả hướng dẫn đóng từ file hiện tại. Tool chuẩn bị build có tùy chọn khóa toàn bộ runtime freeze; Dockerfile competition và predict.sh giữ nguyên. Hướng dẫn ở `zlai/huogndan_vast_docker.md`. Đây là thông tin người dùng xác nhận, chưa kèm log/image digest mới trong workspace. Các cảnh báo tương thích bên dưới là lịch sử rà soát, không phải bằng chứng image đã thử hiện đang lỗi.

Nguồn ban đầu là bản tổng hợp người dùng cung cấp ngày30/10/2026. **Cập nhật14/11:** đã đọc đủ10trang PDF chính thức `E:/Bảo Lập Trình AI/zac2026_DataMate_docker_guideline_final.pdf` và inspect các trang interface/hardware/log/build. Hash/bản trích nằm trong outputs/runs/Bảo/docker-release-20261114; [Docker guide dự án](docker.md) ghi các yêu cầu đã xác minh và phần còn chưa rõ.

Quyết định dự án ngày 09/11/2026: code dùng chung modular monolith, hướng tích hợp Adapter + Strategy; Hải phụ trách vLLM, Thịnh phụ trách RAG. Mỗi người tự chọn benchmark, chạy baseline trước rồi so bản tích hợp trên cùng tập đã chọn. Không bắt buộc bộ 200 câu chung. Xem [kiến trúc](architecture.md) và [hai plan](plans/).

Chạy full-checkpoint inference trên Colab/Kaggle/GPU thuê; máy local phát triển/test CPU, chủ dự án cho phép tiny GPU smoke trên4060 khi phù hợp. Qwen4B/local presets và web đã bỏ khỏi runtime. Sau hợp nhất ngày14/11, adapter vLLM/RAG đã ghép và có [đối chiếu100 câu phát triển trên A100](comparison-100-review-2026-11-14.md); kết quả có training overlap, deadline và thiếu metrics vLLM sau ngắt shutdown, chưa chứng nhận private-test/Docker. Pipeline nghiệp vụ giữ phiên bản6. PDF recommended vLLM0.10.1.1/cu128 và cho phép alternative hỗ trợ5090/SM120; stack0.33 ứng viên dùng CUDA13, driver máy BTC chưa xác nhận đủ mới.

Lịch sử nguồn, các lần benchmark và rollback trước khi dọn được giữ tại [nhật ký cũ](history/competition-provenance-through-2026-11-09.md). Báo cáo số đo cũ không thay baseline cho thay đổi mới. Xem [báo cáo dọn bố cục](team-layout-2026-11-09.md).

## 1. General information

**Competition:** Zalo AI Challenge 2026.

**Track:** DataMate — Talking To Your Data Through Agentic AI.

**Final solution submission deadline:** November 16, 2026, 23:59 (GMT+7).

The participant is working as part of a team, is familiar with Python and programming fundamentals, and has no previous experience building AI Agents.

## 2. Problem definition

Build an AI Agent capable of answering natural-language Vietnamese questions about structured business data from Vietnamese small businesses.

The agent must understand questions and business terminology, retrieve relevant information, perform calculations, and return answers in the required structured format.

Challenges include:

- Vietnamese informal expressions, abbreviations, and colloquial questions.
- Unfamiliar database schemas and business-specific terminology.
- Reasoning over multiple tables and files.
- Applying business rules correctly.
- Numerical calculations and aggregations.
- Temporal expressions relative to a specified date.
- Recognizing questions that cannot be answered from available data.
- Exact structured outputs.

### 2.1 Input

```json
{
  "id": "q_06de33fa21cc",
  "business_id": "biz_118",
  "question": "sp bán dc nhiều nhất tuần rồi là gì",
  "as_of_date": "2026-03-04"
}
```

- `id`: Unique question identifier.
- `business_id`: Identifier of the business whose data should be used.
- `question`: Vietnamese natural-language question.
- `as_of_date`: Reference date for interpreting time expressions.

Each business supplies structured files and a `README.md` describing the business and potentially its data. Shared resources include a business rulebook and a solar–lunar calendar.

### 2.2 Expected output

```json
{
  "id": "q_06de33fa21cc",
  "type": "string",
  "answer": "Xi măng Hà Tiên PCB40 bao 50kg"
}
```

Supported answer types:

- `number`
- `percent`
- `string`
- `date`
- `boolean`
- `list`
- `table`
- `unanswerable`
- `abstain`

`unanswerable` means the necessary information is not available in the data. `abstain` means the agent chooses not to provide an answer. They have different scoring implications.

## 3. Evaluation

### 3.1 Per-question scoring

| Outcome | Score |
|---|---:|
| Correct | +1 |
| Wrong | -1 |
| Abstain | 0 |

An answer is correct only when its **type matches the reference type** and its **value satisfies the matching rule**. There is no partial credit.

An answer with type `abstain`, or a missing answer, counts as abstention.

### 3.2 Penalized Accuracy

```text
PA = (1 / N) × sum(s_i)
s_i = +1 for correct, 0 for abstain, -1 for wrong
```

Every question has equal weight. An `unanswerable` response to an answerable question is **wrong (-1)**.

### 3.3 Answer matching rules

**Number**

- Numerical values such as VND amounts, quantities, and counts.
- Exact matching.
- Reference values are integers or rounded to two decimal places.

**Percent**

- Percentage values expressed in percent units.
- Exact matching after rounding to two decimal places.
- The sign must be correct.

**String**

The prediction and each accepted reference form are normalized as follows:

1. Unicode NFC.
2. Lowercase.
3. Replace characters that are not letters, digits, or whitespace with spaces.
4. Trim leading/trailing whitespace and collapse runs of whitespace to one space.

Vietnamese diacritics are **not removed**. References may accept multiple forms, including alternative spellings and item codes.

**Date**

- ISO `yyyy-mm-dd` for dates.
- ISO `yyyy-mm` for month-specific questions.
- Exact matching.

**Boolean**

- JSON `true` or `false`.
- Exact matching.

**List**

- Ordered list of names or numbers.
- Order comes from the question or the rulebook default.
- Always compared position by position.
- Elements use the corresponding string or number matching rule.

**Table**

```json
{
  "columns": ["khach_hang", "so_tien_no"],
  "rows": [
    ["Chị Lan", 3200000],
    ["Anh Tú", 1500000]
  ]
}
```

- Columns are matched by normalized names.
- If column names do not match, columns are compared by position.
- Rows are compared as a set.
- Cell values use their respective string or number matching rules.

**Unanswerable**

Correct if and only if the reference is unanswerable.

**Abstain**

Always receives 0, regardless of the reference.

### 3.4 Ranking questions

- Use the ranking criterion in the question, or the rulebook when unspecified.
- Default ordering is descending by the criterion.
- Ties are ordered by ascending code.
- Questions whose reference ranking has a tie at the k-th position are excluded from test sets.
- If fewer than k candidates exist, the reference contains all candidates.

### 3.5 Final ranking tie-breakers

1. Total inference time in the organizers' re-run.
2. Earlier submission.

## 4. Dataset specification

### 4.1 Business types

Examples include grocery stores, coffee shops, bubble-tea shops, restaurants, pharmacies, building-material stores, motorbike repair shops, mother-and-baby stores, and other small businesses.

**The private test includes business types absent from training and public test data.**

### 4.2 Dataset characteristics

Each business contains approximately 12–30 months of activity and from a few hundred to about 300,000 rows. Seasonal effects include Tết, Mid-Autumn Festival, public holidays, paydays, and weekends.

Data may be supplied in one or more `.csv`, `.xlsx`, or `.sqlite` files. Each business also has a `README.md` whose quality varies: a complete data dictionary, informal notes, or almost no useful documentation.

### 4.3 Package structure

```text
<split>/
├── data/
│   ├── <business_id>/
│   │   ├── *.csv / *.xlsx / *.sqlite
│   │   └── README.md
│   └── ...
├── rulebook.md
├── calendar.csv
├── questions.jsonl
└── answers.jsonl
```

`answers.jsonl` is provided for training only.

### 4.4 Splits

| Split | Businesses | Questions | Reference answers |
|---|---:|---:|---|
| Training | 30 | ~3,000 | Available |
| Public Test | 10–20 | ~1,000 | Hidden |
| Private Test | 10–20 | ~500 | Hidden |

### 4.5 Training answers

Answers are joined to questions by `id`.

```json
{
  "id": "q_06de33fa21cc",
  "type": "string",
  "answer": "Xi măng Hà Tiên PCB40 bao 50kg",
  "alias": ["SP0001"]
}
```

`alias` provides accepted alternative name forms. Submissions do not need this field. Unanswerable reference questions have `answer: null`.

## 5. Business rules

Each split has its own `rulebook.md`, which defines default interpretations when a question does not specify them.

**Rulebooks can differ between training, public, and private splits. Use the rulebook for the current split.**

The following are examples from the supplied summary, **not immutable rules for every split**:

### Revenue

Revenue is the sum of line amounts after line-level discounts:

- Do not deduct order-level vouchers.
- Exclude VAT and delivery fees.
- Exclude cancelled orders.
- Exclude orders with any returned item.

### Best-selling products

“Bán chạy” defaults to ranking by **quantity**, not revenue.

### Calendar and time

- Weeks run Monday through Sunday.
- “Tháng này” means the month containing `as_of_date`, up to and including that date.
- The Tết window runs from lunar 23 tháng Chạp to mùng 10.

### Cost

Default cost uses weighted-average purchase cost by lot.

## 6. Technical requirements and competition rules

### 6.1 Model eligibility

Model weights must have been publicly released **before October 1, 2026**. The model license must permit the intended use. Fine-tuning eligible models is allowed.

### 6.2 Parameter limit

The total parameter count across **all models used at inference** must not exceed **15 billion**.

This includes language models, embedding models, rerankers, classifiers, and any other inference models, regardless of whether they are loaded simultaneously.

All mixture-of-experts parameters count. Quantization is allowed but does not reduce the parameter count used for this limit. Declare all models in the solution README.

### 6.3 Offline inference

Inference must run entirely offline inside the submitted Docker image. Commercial LLM APIs, external services, resource downloads during inference, and any network access are prohibited. Organizers run images with networking disabled.

### 6.4 No human in the loop

All answers must be generated automatically, with no manual intervention. Hard-coded answers and logic keyed on question identifiers or business identifiers are prohibited.

### 6.5 Code execution

Agents may generate and execute code in the container; this is expected but not mandatory. The mounted dataset is read-only. Write temporary files elsewhere.

### 6.6 Training data

Allowed:

- Provided training data.
- Self-generated synthetic data.
- Publicly available datasets.
- Publicly available eligible models.

Prohibited:

- Training on public test data.
- Training on private test data.
- Collecting answers to test questions by any means.

### 6.7 Reference hardware

- 2 × NVIDIA RTX 5090.
- 32 GB VRAM per GPU; 64 GB total VRAM.
- 512 GB system RAM.

The official PDF received14/11 confirms CUDA>=12.8 and matching PyTorch/vLLM with SM120 support, two RTX5090/32GB each and512GB RAM. Its recommended sample base is vllm/vllm-openai:v0.10.1.1 (Torch2.7.1cu128); alternatives supporting the hardware are allowed. The exact scoring driver is still unknown; the guide states R570-or-newer.

### 6.8 Runtime

The entire private test must finish within **6 hours of wall-clock time, including model loading**.

If a re-run exceeds the limit or terminates early, answers already written to `/result/submission.jsonl` are scored. Remaining unanswered questions count as abstain.

### 6.9 Reproducibility

The final score is determined by the organizers' Docker re-run.

```text
Reproducible when abs(PA(uploaded) - PA(re-run)) <= 0.03
```

For differences exceeding 0.03, organizers examine logs, may re-run again, and may request more information. Unexplained discrepancies may lead to disqualification.

## 7. Submission format

### 7.1 JSONL results

File: `submission.jsonl`. Each line must contain one valid JSON object.

```jsonl
{"id":"q_01234","type":"string","answer":"Sting dâu 330ml"}
{"id":"q_01235","type":"number","answer":48250000}
{"id":"q_01236","type":"percent","answer":-12.45}
{"id":"q_01237","type":"date","answer":"2026-08-15"}
{"id":"q_01238","type":"boolean","answer":true}
{"id":"q_01239","type":"list","answer":["Cà phê sữa","Bạc xỉu","Trà đào"]}
{"id":"q_01240","type":"table","answer":{"columns":["khach_hang","so_tien_no"],"rows":[["Chị Lan",3200000],["Anh Tú",1500000]]}}
{"id":"q_01241","type":"unanswerable","reason":"Dữ liệu không có giá vốn nên không tính được lợi nhuận"}
{"id":"q_01242","type":"abstain"}
```

`reason` is optional. Missing questions count as abstain.

### 7.2 Docker solution

The image must contain training source code, inference source code, required data and model weights, dependencies, `predict.sh`, and a README.

The entry point must process the dataset and write predictions.

| Purpose | Required path |
|---|---|
| Input | `/data` |
| Output | `/result/submission.jsonl` |
| Logs | `/result/logs/` |

Per-question logs must include at least question ID, final answer, and elapsed time. Extra logs may include generated code, execution results, errors, and other information explaining the answer. Organizers use logs to investigate reproducibility.

## 8. Competition rounds and dates

All times below use GMT+7.

### 8.1 Round 1 — Public Test

**October 26 – November 16, 2026.**

Teams develop solutions and run inference on public test data, then upload `submission.jsonl` to the portal.

- At most 5 submissions per day; the quota resets daily.
- Scores appear on the public leaderboard.
- Public scores are for development reference only.

### 8.2 Final solution submission

**November 16, 2026, 23:59.**

Submit the complete solution as a Docker package through the portal's Solution Submission form. Required materials include a public Google Drive download link, a checksum, and complete solution materials.

Each team may submit a final solution package up to 2 times. **Only the last submission is evaluated.** The solution is frozen before the private test phase.

### 8.3 Round 2 — Private Test

**November 20 – November 23, 2026.**

- Private data is released at **09:00 on November 20**.
- Teams run their already-submitted frozen solutions.
- Upload `submission.jsonl` by **November 23, 2026, 23:59**.
- The private leaderboard is displayed chronologically by submission time during this phase.
- Uploaded results are provisional.
- The official final ranking is based on organizers' re-runs of submitted Docker images.
- Organizers inspect source code and check prohibited behavior.

### 8.4 Results

- **December 2, 2026:** Top 2 teams from each track announced.
- **December 12, 2026:** Final rankings announced.

## 9. Available materials and unknowns

The user states that the following were reviewed when preparing the supplied summary:

- Challenge Description/Overview.
- Evaluation rules and scoring details, including screenshots.
- Data documentation: splits, structure, business rules, submission format.
- Competition Rules.
- Submission Guidance.

At the time this file was created, the actual downloaded training/public datasets and the separate Docker guideline had **not** been supplied in the conversation or workspace.

Consequently, the following are **not established**:

- Actual schemas, table and column names, keys, and data relationships.
- Actual data distributions, data quality, and README contents.
- Actual contents of split-specific `rulebook.md` and `calendar.csv`.
- The official Docker base image and full execution interface beyond the paths and `predict.sh` described above.
- Model selection or eligibility of any particular model.
- Architecture, framework, training strategy, deployment implementation, or project schedule.
- Accuracy, inference time, GPU memory consumption, or any benchmark performance.

Do not assume these details are known. Verify against the actual files and official materials when they become available.

## 10. Source and purpose

**Source of this saved file:** the user's consolidated competition context, supplied October 30, 2026.

**Official portal cited by the user:** <https://challenge.zalo.ai/portal/datamate/>.

Purpose: retain enough factual context for future work on the competition agent, without prematurely selecting a technical solution. Keep competition requirements, observations from real data, and the team's future design decisions distinct.

## 11. Internal engineering update — November 14, 2026

Source: project owner's request, fetched team branches and local Hải benchmark logs; see [integration report](integration-vllm-rag-2026-11-14.md). Team decisions: Adapter + Strategy, independent generator backend/retrieval selection, three-arm current-source evaluation. Observed Hải timings are diagnostic 24-question evidence without labels, with deadline/answer differences; they do not establish quality or full private-test runtime. This integration has CPU fixture verification, not a full-checkpoint GPU/Docker result. No additional official organizer guideline or base-image requirement was independently established by this update.

### Completed fine-tune artifact received November 14

Source: owner supplied `outputs/runs/Bảo/qwen35-9b-sql-stage2-v1-full-01`. Its `adapter/training_report.json` (SHA256 `9a09f391…`) and saved adapter hashes were verified against the actual files. Stage2 step198/0.5 requested epoch links exactly to parent step298/1 requested epoch, with parent report SHA256 `fcc245948098…`. Both use the same pinned Qwen base manifest `f8b32a3f…`. Final weights match earlier checkpoint198 (`f346ad51…`); the completed original report is now used instead of a snapshot report. These artifacts support the reported training chain, not a claim of improved PA. Current engineering choice: Colab Pro; merge once on Colab and compare Torch/vLLM/Torch+RAG on the same selected set. The actual Drive base location and the three-arm GPU runtime remain to be checked on Colab; no official competition rule changed. See [workflow](colab-finetuned-compare.md).
