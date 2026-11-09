# Plan kỹ thuật tích hợp RAG — Thịnh

Cập nhật **09/11/2026** theo bố cục repo hiện tại và yêu cầu làm trên hai nhánh riêng. **Người thực hiện: Thịnh; nhánh: `codex/thinh-rag`.** Phối hợp với [Hải — vLLM](plan-integrate-vllm-2026-11-09.md).

Đây là đặc tả để code tiếp, **chưa phải các file/chức năng đã triển khai hoặc kết quả GPU đã đo**. RAG triển khai/đánh giá được trên Transformers trước khi vLLM xong. Full-checkpoint inference chạy trên Colab/Kaggle/GPU thuê; local dùng edit/test CPU.

## 1. Mục tiêu, kiến trúc và model

Dùng **Adapter + Strategy**:

- **Adapter:** `VietnameseEmbeddingAdapter` bọc encoder, `VietnameseRerankerAdapter` bọc model chấm cặp. Hai adapter này không phải `ChatBackend` sinh text.
- **Strategy:** `rag.mode = none | dense | dense_rerank` chọn `NullRetriever | DenseRetriever | DenseRerankRetriever` qua factory.
- **Dependency injection:** `create_agent()` lắp backend/retriever rồi truyền vào Agent. RAG chỉ dùng `ChatBackend.count_tokens()` để đóng context, không import TransformersBackend/VLLMBackend cụ thể.

| Key cần thêm vào registry | Repository / revision đã khóa | Tham số |
|---|---|---:|
| `vietnamese-embedding` | `AITeamVN/Vietnamese_Embedding` / `dea33aa1ab339f38d66ae0a40e6c40e0a9249568` | 567.754.752 |
| `vietnamese-reranker` | `AITeamVN/Vietnamese_Reranker` / `f536976248403314225d7fdfdbc87f0e9516a54e` | 567.755.777 |

Hai model tổng **1.135.510.529 tham số**. Metadata/lịch sử/header đã kiểm; **chưa tải/kiểm hash toàn bộ weights local hoặc chạy inference thật**. Giữ revision trên, tải đúng artifact rồi verify/seal trước khi dùng; không lấy `main` thay thế. Nguồn và số hash ở [báo cáo model](../rag-model-review-2026-11-09.md) và [evidence JSON](../model-evidence-rag-2026-11-09.json).

## Phối hợp hai nhánh và quyền sửa file

**Hai người code trên hai nhánh khác nhau, trong working copy/worktree riêng:**

| Người | Nhánh | Preset / kết quả riêng |
|---|---|---|
| Hải | `codex/hai-vllm` | `configs/experiments/hai/` / `outputs/runs/hai/` |
| Thịnh | `codex/thinh-rag` | `configs/experiments/thinh/` / `outputs/runs/thinh/` |

Cả hai tách từ `main` đã có bố cục mới, PR vào `main`. Không dùng chung checkout rồi thay nhau chuyển branch; không chép cả folder của nhánh này đè lên nhánh kia. Nếu đã có nhánh đang làm, merge `origin/main` vào nhánh đó, không tạo lại và bỏ code cũ. Lệnh cụ thể ở [team workflow](../team-workflow.md).

**Một file có một người phụ trách chính**, kể cả file dùng chung. Bảng dưới áp dụng giống nhau cho cả hai plan; đường dẫn rút gọn bắt đầu từ gốc repo.

| Người sửa | File / nhóm file | Ranh giới |
|---|---|---|
| Hải | `src/datamate/config.py`, `src/datamate/cli.py` | Khai báo cả cấu hình vLLM và RAG đã thống nhất; CLI chọn backend/preset |
| Hải | `src/datamate/models/contracts.py`, `chat_format.py`, `runtime.py`, `vllm_backend.py`, `backends.py` | Contract, adapter sinh text, token counter, factory và thông tin engine |
| Hải | `src/datamate/agent/preparation.py`, `schema_repair.py` | Chỉ cập nhật contract/backend identity và truyền lỗi engine; giữ logic nghiệp vụ |
| Hải | `pyproject.toml`, `requirements-colab.txt`, dependency lock GPU mới | Thịnh đưa yêu cầu package; Hải kiểm tương thích và cập nhật một nơi |
| Hải | `tools/diagnose_train.py`, `colab_runtime_check.py`, `package_colab.py` | Diagnostic dùng interface, source bundle, nhận diện runtime |
| Hải | `tools/create_inference_notebook.py`, `create_team200_colab.py` và hai notebook inference sinh ra | Thêm lựa chọn preset/dependency; Thịnh chọn file config, không sửa trực tiếp notebook |
| Hải | `docker/Dockerfile.competition` | Cài offline các extras cần cho cấu hình đã đóng gói |
| Thịnh | Toàn bộ `src/datamate/retrieval/` | Contract retriever, encoder, corpus/index, strategy, context |
| Thịnh | `src/datamate/application.py`, `agent/runner.py`, `agent/prompts.py` | Lắp thành phần, lifecycle, retrieval hook, prompt, log/resume |
| Thịnh | `src/datamate/models/registry.py`, `artifacts.py` | Inventory toàn hệ thống, checkpoint embedding/reranker và seal/verify |
| Thịnh | `tools/infer_team200.py`, `tools/package_solution.py` | Metadata đủ model; build context đủ checkpoint RAG |
| Mỗi người | Test mới, preset và báo cáo của phần mình | Tên test/tệp riêng nêu trong phần kỹ thuật bên dưới |
| Chủ dự án | `configs/submission.json`, README và hướng dẫn chung khi chốt bản ghép | Chọn cấu hình cuối sau kiểm chứng, không tự bật mặc định từ một PR tính năng |

Các tên `chat_format.py`… trong hàng `models/` đều thuộc `src/datamate/models/`; các tên rút gọn trong hàng `tools/` đều thuộc `tools/`. Hải không sửa `application.py/runner.py/prompts.py/registry.py/artifacts.py/package_solution.py`; Thịnh không sửa `config.py/backends.py/pyproject.toml/Dockerfile.competition`. Nếu cần thay đổi file của người kia, gửi yêu cầu interface/patch nhỏ để người phụ trách thực hiện trên nhánh của họ.

### Thứ tự PR để hai nhánh không lệch interface

1. **Cả hai chạy baseline trước**, trên bộ do từng người chọn. Lưu source commit, preset và tập câu; không dùng kết quả lịch sử thay cho baseline mới.
2. **PR nền H — Hải, từ `codex/hai-vllm`:** thêm `ChatBackend` chuẩn, token counter/runtime identity, cấu hình lồng `vllm/rag` và tương thích Transformers/demo. Thịnh review phần sẽ sử dụng. Chưa bật vLLM/RAG.
3. Merge PR nền H vào `main`; Thịnh merge `origin/main` vào `codex/thinh-rag`. **PR nền T — Thịnh:** contract retriever + `NullRetriever`, injection vào Agent, inventory mở rộng, log/resume và lifecycle dùng interface của Hải. `rag.mode=none` giữ prompt/hành vi baseline.
4. Merge PR nền T; Hải merge `origin/main` vào `codex/hai-vllm`. Kiểm tra hai PR nền trên preset baseline; nếu có thay đổi đáp án phải giải thích/sửa trước khi đo tính năng.
5. Hai người tiếp tục **song song trên chính hai nhánh riêng**: Hải hoàn thiện vLLM; Thịnh hoàn thiện dense/rerank. Có thể viết module riêng từ sớm, nhưng PR tích hợp dựa trên hai PR nền đã merge.
6. Mỗi PR tính năng kèm kết quả trước/sau. Sau khi merge, cùng chọn một bộ để kiểm bản `vllm + RAG` với baseline tương ứng; không suy ra hiệu quả bản ghép từ hai benchmark khác nhau. Không bắt buộc chạy đủ ma trận backend × RAG.

Mỗi người commit công việc đang làm trước khi `git fetch origin` rồi `git merge origin/main` trên nhánh của mình. Với squash merge, đồng bộ nhánh và xử lý lịch sử theo hướng dẫn chủ dự án; không force-push `main` hoặc dùng reset để bỏ thay đổi của đồng đội.

## 2. Benchmark: tự chọn, baseline trước

Thịnh tự chọn benchmark, số câu và công cụ chạy; **không bắt buộc Team200 hoặc cùng bộ với Hải**.

1. Chạy/chấm `rag.mode=none` trước khi tích hợp; lưu tập câu, cách chấm, preset, commit và môi trường.
2. Chạy RAG trên **đúng tập đó**, cùng generator/revision/backend/prompt nền và ngân sách tương đương. Có thể đo `dense` rồi `dense_rerank` để thấy lợi ích reranker.
3. Báo chất lượng đáp án và tổng thời gian gồm load/build index; retrieval score cao chưa chứng minh đáp án tốt hơn.

**Đạt:** chất lượng tốt hơn baseline, thời gian/tài nguyên trong giới hạn hệ thống. Chỉ giữ reranker nếu có lợi ích đo được; không chọn lại câu có lợi sau khi xem điểm. Báo cáo ở `docs/reports/thinh-rag-<run>.md`, output lớn ở `outputs/runs/thinh/<run>/`.

## 3. Contract và package cần tạo

### 3.1. `src/datamate/retrieval/contracts.py` — PR nền T

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    split_fingerprint: str
    business_id: str
    as_of_date: str

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    dense_score: float | None = None
    rerank_score: float | None = None

@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    diagnostics: dict

class ContextRetriever(Protocol):
    def prepare(self, business_ids: list[str],
                *, remaining_seconds: float) -> None: ...
    def retrieve(self, request: RetrievalRequest,
                 *, remaining_seconds: float) -> RetrievalResult: ...
    def identity(self) -> dict: ...
    def runtime_info(self) -> dict: ...
    def close(self) -> None: ...
```

Quy ước:

- `split_fingerprint` là hash nguồn/config của split, không chỉ tên folder; `business_id` dùng định tuyến, `as_of_date` giữ ngữ cảnh thời gian. Không dùng question ID làm luật trả lời.
- `metadata` tối thiểu có `split_fingerprint`, scope `global | business`, business_id nếu có, source_kind, đường dẫn nguồn tương đối, source_sha256, mục/đoạn; schema có bảng/cột gốc. Không tạo trích dẫn không có nguồn.
- `diagnostics` có mode, status `ok | disabled | empty | timeout | fallback`, thời gian từng phase, candidate/selected IDs, scores, index identity và lý do lỗi/fallback. Dữ liệu phải JSON-serializable.
- `prepare()` load encoder và chuẩn bị index/cache cho doanh nghiệp được chọn, tái sử dụng khi gọi lại. Thời gian này tính vào wall time, không giấu trong bước online trước benchmark.
- `identity()` ổn định theo model/corpus/code/config; `runtime_info()` ghi phiên bản, dtype/device, chi phí prepare và manifest index thực tế. `close()` giải phóng encoder, không đóng backend generator.
- `NullRetriever` không load model, tạo index, đọc thêm catalog hoặc làm thay đổi prompt; prepare/close là no-op. Retrieve trả rỗng với status disabled.
- Định nghĩa `RetrievalFatalError(DataMateError)` tại contract này cho lỗi encoder/index không phục hồi. Timeout/lỗi truy xuất có thể phục hồi được strategy đổi thành result rỗng có diagnostics; lỗi fatal phải truyền qua runner để dừng batch.

### 3.2. File mới và chức năng cụ thể

| File mới trong `src/datamate/retrieval/` | Class/hàm chính | Trách nhiệm |
|---|---|---|
| `__init__.py` | Public contract | Export nhẹ; không import torch/encoder khi chỉ import package |
| `contracts.py` | Các kiểu ở trên; protocol encoder/reranker | Kiểu dữ liệu độc lập engine; `encode(texts, *, remaining_seconds) -> ndarray[N,1024]` và `score(query, passages, *, remaining_seconds) -> list[float]` |
| `encoders.py` | `VietnameseEmbeddingAdapter`, `VietnameseRerankerAdapter` | Verify local model, tokenize/batch/inference, pooling/logit và deadline giữa các batch |
| `corpus.py` | `build_corpus(split, ingestor, business_id)` | Chỉ nhận nguồn cho phép; tạo documents có scope/provenance từ file và raw catalog; trả danh sách document với text/metadata |
| `chunking.py` | `chunk_documents(documents, tokenizer, settings)` | Chia theo mục và embedding-token budget, giữ điều kiện/ngoại lệ; tạo chunk ID ổn định |
| `index.py` | `LocalDenseIndex`: build/load/search | Vector index local chính xác bằng NumPy; atomic write, checksum/manifest, filter scope trước search |
| `strategies.py` | `NullRetriever`, `DenseRetriever`, `DenseRerankRetriever` | Luồng encode query → top-k → rerank nếu bật; giữ đúng chunk-score alignment |
| `factory.py` | `build_retriever(settings, split, ingestor)` | Chọn đúng strategy, lazy import encoder chỉ khi mode cần; mode lạ báo lỗi |
| `context.py` | `fit_context(chunks, *, build_messages, backend, settings)` | Dùng callback dựng prompt đầy đủ rồi đếm token; trả messages đã fit và diagnostics |
| `identity.py` | `retrieval_identity(settings, split, business_ids)` | Tính hash nguồn/model manifest/chunker/config trước load model để kiểm resume và cache |

Không tạo `retrieval/config.py`. Dùng `RAGSettings` do Hải khai báo trong `src/datamate/config.py` để hai nhánh có đúng một schema cấu hình.

## 4. Điểm ghép vào code hiện có — Thịnh sửa

| File đang có | Chỉnh sửa cụ thể |
|---|---|
| `src/datamate/models/registry.py` | Thêm hai checkpoint trên; mở rộng `inventory(main_key, assistant_key=None, *, extra_keys=())`, thêm `inventory_from_settings(settings)`; giữ API hai đối số cũ |
| `src/datamate/models/artifacts.py` | Staging đủ cấu hình/tokenizer/module Sentence Transformers; seal/hash toàn bộ file cần offline; verify manifest generator cũ còn hợp lệ |
| `src/datamate/application.py` | Composition với một Ingestor dùng chung, full inventory, backend và retriever; prepare index trong budget; identity/log/lifecycle như mục 7 |
| `src/datamate/agent/runner.py` | Thêm dependency retriever/ingestor tùy chọn; một retrieval hook trước initial_messages; trace lỗi/timing; giữ loop plan/SQL hiện tại |
| `src/datamate/agent/prompts.py` | `initial_messages(..., contract=None, *, retrieved_context=None)`; chỉ thêm trường khi có đoạn |
| `tools/infer_team200.py` | Nếu dùng công cụ này: metadata đủ active models/revisions/hash và runtime identity; giữ định dạng benchmark hiện có |
| `tools/package_solution.py` | Đóng gói đủ active models/RAG config, rewrite path offline, manifest/extras cần cho Dockerfile Hải |
| `configs/experiments/thinh/*.json` | Tạo preset none/dense/dense_rerank cùng generator, backend, prompt nền |
| `docs/reports/thinh-rag-<run>.md` | Sau khi đo: kết quả trước/sau, lệnh, môi trường, lỗi còn lại và lợi ích reranker |

Không sửa `pyproject.toml/config.py` từ nhánh RAG. Yêu cầu Hải thêm extra `rag` với Sentence Transformers/NumPy và bộ Transformers/PyTorch tương thích; Hải pin dependency sau smoke chung. Không tự hạ Transformers chỉ theo version cũ ghi trong config encoder.

## 5. Model adapters và corpus/index

### Embedding

`VietnameseEmbeddingAdapter` dùng Sentence Transformers với đường dẫn local đã verify, `local_files_only=True`, `trust_remote_code=False`, eval/inference mode. Giữ **CLS pooling + L2 normalize**, vector **1024 chiều** như cấu hình checkpoint; không đổi sang mean pooling. Bắt đầu giới hạn **2048 token** theo phần mô tả huấn luyện, dù config có 8192; thay giới hạn phải đo và ghi preset. Xem [cấu hình checkpoint đã kiểm](../rag-model-review-2026-11-09.md).

`encode()` trả NumPy float32, shape `[N,1024]`, vector hữu hạn và chuẩn hóa; từ chối NaN/zero vector. Vector document/query dùng cùng adapter/config. Dot product trên vector chuẩn hóa dùng làm similarity. Chỉ đổi dtype tính toán khi GPU hỗ trợ và đã smoke; dtype phải vào identity.

### Reranker

`VietnameseRerankerAdapter` dùng `AutoTokenizer` + `AutoModelForSequenceClassification` từ artifact local, eval/inference mode. Mỗi cặp query–passage nhận **một logit**; flatten thành score theo đúng thứ tự đầu vào, không dùng như model embedding. [Config tác giả tại revision đã khóa](https://huggingface.co/AITeamVN/Vietnamese_Reranker/blob/f536976248403314225d7fdfdbc87f0e9516a54e/config.json).

Giới hạn cặp khởi đầu **2304 token gồm special tokens**, query tối đa 256; passage tối đa 2048 và phải giảm thêm nếu cần để tổng không vượt 2304. Nếu rút query cho retrieval/reranking, ghi số token bị cắt; câu hỏi gốc trong prompt generator giữ nguyên. Không ép logit thành bằng chứng đáp án đúng.

### Corpus phiên bản đầu

| Đưa vào | Cách lấy / giới hạn |
|---|---|
| `rulebook.md` của split | Scope global trong đúng split, chunk theo mục; vẫn giữ bản đầy đủ trong prompt nền |
| `README.md` doanh nghiệp | Scope đúng business; giữ các điều kiện/ngoại lệ cùng đoạn |
| Schema raw / mô tả bảng-cột có nguồn | Dùng `Ingestor.catalog(business_id, normalize=False)` hiện có; giữ tên file/sheet/bảng/cột thật |
| Danh mục tĩnh có khóa | Chỉ thêm khi xác định được nguồn/schema thực tế; giữ mã/tên/size và provenance, không tự đoán tên cột |

**Bản đầu không index** gold answers, lời giải/trace/review/log benchmark, output của câu đang đánh giá hoặc binding do LLM preparation/schema repair vừa sinh. Như vậy index raw-schema không bị thay bằng suy đoán của model. Nếu sau này index semantic binding thay đổi trong runtime, phải bổ sung invalidation theo catalog fingerprint trước khi dùng.

Tách split/business **trước top-k**, không search toàn bộ rồi lọc sau. Đoạn global của split chỉ chia sẻ trong split đó. Query có `as_of_date` nhưng giải nghĩa thời gian vẫn theo calendar/rulebook hiện tại.

**SQL/công cụ tiếp tục tính trên toàn bộ dòng thỏa điều kiện.** Không lấy top-k giao dịch rồi cộng tiền. Tên/mã/size phải đối chiếu khóa chính xác bằng tools; semantic top-1 chỉ là gợi ý nguồn, không tự chứng minh cùng thực thể.

### Chunking, index và cache

1. Chia theo heading/bảng/cụm mô tả, dùng tokenizer embedding để giới hạn chunk; overlap không làm mất phủ định hoặc điều kiện. ID chunk = hash scope + source hash + vị trí + phiên bản chunker.
2. Bản đầu dùng NumPy exact search, đủ cho corpus metadata nhỏ; chưa thêm vector DB/service hoặc hybrid/BM25. Search trả `candidate_k`, rerank nếu bật, chọn tối đa `context_k`; tie-break bằng chunk ID cho kết quả ổn định.
3. Cache ở `<cache_dir>/retrieval/<fingerprint>/` hoặc index_dir đã kiểm ngoài dataset. Tách `vectors.npy`, `chunks.jsonl`, `index_manifest.json`; không pickle object model.
4. Manifest chứa hash nguồn/scope, raw catalog + ingest version, chunker/code version, model/tokenizer manifest, pooling/normalization/dtype, config chunk/index, shape và hash file vector/chunk.
5. Ghi thư mục tạm rồi atomic rename; chỉ đọc lại khi checksum/identity khớp. Đổi nguồn/model/tokenizer/config phải build cache mới, không dùng chỉ tên business làm cache key.
6. Nếu cache query, khóa gồm text query + as_of_date + split/business + index/model/config identity; không chỉ question ID. Bản đầu có thể chưa cache query.
7. `prepare()` chỉ chuẩn bị doanh nghiệp trong lượt chạy, tái dùng cache hợp lệ. Thời gian load encoder/build hoặc đọc index đều vào startup/wall time; báo cả cache nóng/lạnh khi khác nhau.

## 6. Cấu hình Thịnh sử dụng

Schema do Hải thêm trong PR nền H; Thịnh review đầy đủ trước merge. Tạo ba file `configs/experiments/thinh/rag-none.json`, `rag-dense.json`, `rag-dense-rerank.json` từ cùng preset generator, giữ tên run riêng. Ví dụ phần RAG của preset cuối:

```json
{
  "backend": "transformers",
  "rag": {
    "mode": "dense_rerank",
    "embedding_key": "vietnamese-embedding",
    "embedding_dir": "models/vietnamese-embedding",
    "reranker_key": "vietnamese-reranker",
    "reranker_dir": "models/vietnamese-reranker",
    "chunk_tokens": 384,
    "chunk_overlap": 48,
    "candidate_k": 20,
    "context_k": 5,
    "context_tokens": 2048,
    "embedding_max_tokens": 2048,
    "rerank_max_tokens": 2304,
    "rerank_query_max_tokens": 256,
    "embedding_batch_size": 8,
    "rerank_batch_size": 4,
    "timeout_seconds": 8,
    "device": "cuda:0",
    "dtype": "auto",
    "index_dir": null
  }
}
```

Đây là **điểm thử**, không phải siêu tham số bắt buộc. Chỉnh qua JSON và lưu lại, không hard-code trong runner. `none` bỏ qua cả hai model/path; `dense` chỉ nạp embedding; `dense_rerank` mới nạp reranker. Backend demo chỉ dùng `none`.

## 7. Luồng ghép và xử lý lỗi

### Composition / lifecycle — `application.py`

1. Validate Settings, đường cache/output/index ngoài dataset, `inventory_from_settings()` và artifact cần dùng **trước load weights**.
2. Giữ `create_agent(settings)` tương thích; thêm keyword tùy chọn `business_ids=None` và `remaining_seconds=None`. Batch truyền business của tập câu đã chọn và wall budget còn lại; `ask` truyền một business. Khi bật RAG mà không truyền danh sách, dùng các business của split.
3. Tạo một `DatasetSplit/Ingestor`, backend từ factory Hải, retriever từ factory Thịnh; gọi `retriever.prepare()` với budget sau khi trừ thời gian đã load. Với none, không thêm ingestion/index.
4. Truyền `Agent(settings, backend, split, *, ingestor=..., retriever=...)`. Constructor cũ vẫn dùng được: ingestor thiếu thì tạo như hiện tại, retriever thiếu dùng NullRetriever khi mode none; mode bật mà thiếu retriever phải báo lỗi.
5. Agent có `close()` đóng retriever/backend an toàn. `run_batch()` và CLI ask dùng `try/finally`; khởi tạo hỏng giữa chừng cũng đóng thành phần đã tạo. Công cụ nhận Agent từ caller không tự đóng sau mỗi câu/batch; caller đóng khi kết thúc phiên.
6. `run_batch()` tiếp tục bắt đầu timer trước hash/load. Hết budget trong prepare phải dừng rõ ràng, không âm thầm chạy bản none rồi ghi là RAG. Thông báo Hải cập nhật caller CLI/notebook/diagnostic theo chữ ký này.

### Retrieval hook — `agent/runner.py` và `prompts.py`

- Vị trí hook: **sau catalog/preparation/preflight, trước initial_messages và prompt_snapshot**. Một lần truy xuất cho mỗi câu trong bản đầu; không gọi lại sau mỗi tool step.
- Tạo RetrievalRequest từ query gốc, fingerprint, business và as_of_date. Ngân sách retrieval là `min(rag.timeout_seconds, deadline - now)`, dùng **deadline hiện tại của Agent**, không cộng thêm thời gian miễn phí.
- Đưa chunks vào `fit_context()`. Callback `build_messages(context)` gọi `initial_messages(..., retrieved_context=context)`, nên token count bao gồm JSON wrapper, nguồn, chat template và generation prompt.
- Đảm bảo `count_tokens(full_messages) + settings.max_new_tokens <= settings.max_context_tokens`, đồng thời phần context thêm vào ≤ `rag.context_tokens`. Bỏ nguyên đoạn điểm thấp trước; không cắt rulebook/README/câu hỏi gốc để nhường RAG.
- Trước các lượt generate sau, nếu tool history làm prompt vượt giới hạn, giảm/bỏ block RAG đang giữ rồi đếm lại; không retrieval lần nữa hoặc xóa tool history. Nếu bỏ hết RAG vẫn quá context, dùng đường báo lỗi overflow hiện có. Log context thực tế của từng lượt.
- Chunks chỉ là dữ liệu tham khảo, không là system instruction hoặc evidence của phép tính. Giữ thứ tự ưu tiên rulebook và tool verification. Không để đoạn retrieved điều khiển thao tác ngoài pipeline.
- `none`/rỗng/fallback phải cho **messages giống baseline**, không thêm field rỗng, không đổi system prompt. Cảnh báo về dữ liệu RAG chỉ xuất hiện khi thật sự thêm context.
- Raw-schema được index từ nguồn gốc; binding do schema repair sinh không được lấy từ RAG để thay catalog. Tools vẫn nhận catalog hiện hành, không dùng score làm evidence đủ dữ liệu.

### Deadline và fallback

| Tình huống | Hành vi |
|---|---|
| Không có đoạn phù hợp | Context rỗng, tiếp tục baseline; không suy ra unanswerable |
| Lỗi query encode/search/rerank có thể phục hồi | Log phase/error, bỏ RAG cho câu này; tiếp tục baseline nếu còn thời gian |
| Hết budget retrieval | Dừng trước batch kế tiếp, bỏ context nếu không có kết quả hợp lệ; thời gian thực vẫn bị trừ khỏi deadline câu |
| Thiếu/sai hash checkpoint, index hỏng không rebuild được, OOM/encoder không phục hồi | Báo lỗi khởi tạo hoặc ném `RetrievalFatalError` để dừng batch, không coi là fallback bình thường |
| `BackendUnavailableError` từ generator Hải | Re-raise qua catch ở runner, kể cả trong repair helper; application dừng và giữ output đã flush |

Encoder chạy batch nhỏ có kiểm deadline trước/sau mỗi batch; không tạo background Future tiếp tục giữ GPU sau timeout. Kernel đang chạy có độ trễ hoàn tất, phải đo độ trễ thực và chọn batch phù hợp, không báo đã có timeout cứng khi chưa kiểm. Trace lưu timing encode/search/rerank/pack, IDs/scores/nguồn/context tokens, fallback và index/model identity.

## 8. Inventory, resume và Docker

- `inventory_from_settings()` cộng generator + assistant nếu có + embedding theo dense + reranker theo dense_rerank, deduplicate theo checkpoint; kiểm trần trước load. `inventory(main, assistant)` cũ vẫn có nghĩa inventory của riêng cặp sinh.
- Không thêm trường vào `ModelSpec.to_dict()` làm manifest generator đã seal khác đi; role embedding/reranker khai báo riêng hoặc có migration tường minh. Artifact phải giữ tokenizer, `modules.json`, pooling/Normalize configs và license.
- Hash weights khi stage thật phải khớp evidence: embedding `1f2debafdf03659e8273022a3e902b94deec73cd20c2b7262ab7e21630163f6d`; reranker `1cecccab406ec6d3f13da8ccdc5e853a1372e51845322163655ab530d3730071`. Đếm tham số và verify toàn bộ file trước dùng.
- `run_identity.json` phải gồm full inventory/artifact manifests, `runtime_identity(settings)` của Hải, `retrieval_identity()` và toàn source/input hashes. Tính phần ổn định **trước resume/model load**; đổi corpus/model/chunker/mode/config phải từ chối resume vào run cũ.
- Manifest cache nóng/lạnh không làm đổi identity cấu hình; khi reuse vẫn verify hash file index, log fingerprint thực tế và chi phí chuẩn bị vào runtime/summary. Metadata Team200 nếu dùng cũng phải khai báo đủ models.
- `tools/package_solution.py` dùng full inventory, chỉ copy model thực sự dùng, rewrite path RAG thành `/app/models/<key>` và index/cache thành `/work/...`. Kiểm manifest trước và sau copy; ghi extras cần cài để Hải cập nhật Dockerfile/wheelhouse.
- Không đóng gói index dựa trên public/private test đã thu thập ngoài luồng inference. Chuẩn bị index từ split hiện có trong container với dataset read-only; tính chi phí vào thời hạn.
- Qwen 9B + hai encoder = **10.788.614.897**; Gemma 12B + hai encoder = **13.095.240.753**; thêm Gemma assistant = **13.518.097.717**. Upscaled 14B + hai encoder = **15.734.995.569**, phải từ chối. Load/unload lần lượt hoặc quantization không giảm số tính.
- Khi ghép vLLM, thống nhất với Hải VRAM/context/dtype/batch và thứ tự load đã smoke; engine không được chiếm hết chỗ của hai encoder. Không chạy hai generator lớn đồng thời chỉ để so benchmark.

## 9. Test và bàn giao

Đây là các kiểm thử **cần viết/chạy khi triển khai**, không phải kết quả đã có:

| Test của Thịnh | Tình huống chính |
|---|---|
| `tests/retrieval/conftest.py` | Fake encoder/reranker/index riêng, không tranh sửa root conftest của Hải |
| `tests/retrieval/test_encoders.py` | Shape 1024, CLS/normalize, một score/cặp, alignment/truncation; model giả cho CPU |
| `tests/retrieval/test_corpus_index.py` | Phạm vi split/business trước top-k, provenance, nguồn loại trừ, cache hash/atomic/invalidation |
| `tests/retrieval/test_context.py` | Token budget bằng generator counter, bỏ đoạn đủ nguyên nghĩa, context rỗng giữ nguyên messages |
| `tests/test_agent_retrieval.py` | Hook/deadline/fallback, không index binding sửa bằng LLM, fatal engine được truyền ra |
| `tests/test_rag_inventory.py` | none/dense/rerank/assistant, dedup, từ chối >15B, manifest cũ tương thích |
| `tests/test_rag_solution_package.py` | Stage đủ active artifacts, rewrite path/extras, resume từ chối khi model/corpus/config đổi |
| Test hiện có | Thịnh cập nhật `tests/test_agent_application.py`, `test_team200_inference.py`, `test_team_layout.py` nếu điểm ghép thay đổi |

Chạy kiểm thử CPU phù hợp, rồi smoke hai checkpoint thật trên GPU để xác nhận tokenizer/pooling/logit/dtype, VRAM và context. Sau đó đo trước/sau theo mục 2, kiểm bản ghép cùng Hải và chạy thử Docker **mạng tắt**. Không suy ra đã tăng chất lượng chỉ từ unit test.

Bàn giao: PR vào `main`, ba preset, manifest checkpoint/index, yêu cầu dependency cho Hải, báo cáo benchmark, lệnh chạy và cách tắt RAG. Chủ dự án chốt preset kết hợp rồi cùng nhóm đóng gói Docker image/checksum cuối theo [hướng dẫn Docker](../docker.md).

Tuân thủ [ngữ cảnh cuộc thi](../competition-context.md): checkpoint trọng số công khai trước 01/10/2026, pin revision/hash/license; tổng mọi model inference ≤15B. Không dùng test để training/thu thập đáp án; `/data` chỉ đọc, output `/result/submission.jsonl`, log `/result/logs/`, scratch/index ngoài dataset. Toàn private test ≤6 giờ kể cả load/build index; base image/guideline BTC còn phải xác minh.
