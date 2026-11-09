# Plan kỹ thuật tích hợp vLLM — Hải

Cập nhật **09/11/2026** theo bố cục repo hiện tại và yêu cầu làm trên hai nhánh riêng. **Người thực hiện: Hải; nhánh: `codex/hai-vllm`.** Phối hợp với [Thịnh — RAG](plan-integrate-rag-2026-11-09.md).

Đây là đặc tả để code tiếp, **chưa phải các file/chức năng đã triển khai hoặc kết quả GPU đã đo**. Baseline hiện là pipeline 6 / Transformers; giữ backend này để đối chứng và chuyển lại khi cần. Full-checkpoint inference chạy trên Colab/Kaggle/GPU thuê; local dùng edit/test CPU.

## 1. Mục tiêu và kiến trúc

Dùng **Adapter + Strategy** trong modular monolith:

- **Adapter:** `VLLMBackend` chuyển giao thức DataMate sang API engine vLLM local. `TransformersBackend` vẫn thực hiện cùng contract.
- **Strategy:** `Settings.backend = transformers | vllm | demo`; `build_backend()` chọn implementation. `create_agent()` truyền backend vào Agent.
- Agent, preparation, schema repair và RAG gọi interface, không import engine vLLM. Giữ logic plan, tools/SQL, projection/type, scorer và định dạng submission.

```text
CLI / notebook / Docker
        -> application.create_agent()                  [Thịnh]
        -> models.backends.build_backend(settings)     [Hải]
             -> TransformersBackend | VLLMBackend
        -> Agent(..., backend, retriever=...)           [Thịnh]
```

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

Hải tự chọn benchmark, số câu và công cụ chạy; **không bắt buộc Team200 hoặc cùng bộ với Thịnh**.

1. Chạy/chấm Transformers trước khi tích hợp; lưu tập câu, cách chấm, preset, commit và môi trường.
2. Chạy vLLM trên **đúng tập đó**, cùng generator/revision, prompt, token budget và GPU/điều kiện tương đương. Đổi môi trường đáng kể thì đo lại baseline.
3. Báo chất lượng đáp án và tổng thời gian gồm load/chuẩn bị; token/giây chỉ là số phụ.

**Đạt:** nhanh hơn, chất lượng không giảm; không áp tỷ lệ tăng tốc cố định. Không chọn lại câu có lợi sau khi xem điểm. Báo cáo để ở `docs/reports/hai-vllm-<run>.md`, output lớn ở `outputs/runs/hai/<run>/`.

## 3. Contract Hải cung cấp trong PR nền H

### 3.1. `src/datamate/models/contracts.py` — tạo mới

```python
from typing import Protocol

class ChatBackend(Protocol):
    counts_as_model: bool
    supports_token_budget: bool
    last_generation: dict | None

    def generate(self, messages: list[dict], remaining_seconds: float,
                 *, max_new_tokens: int | None = None) -> str: ...
    def count_tokens(self, messages: list[dict]) -> int: ...
    def runtime_info(self) -> dict: ...
    def close(self) -> None: ...
```

Quy ước bắt buộc:

| Thành phần | Ý nghĩa |
|---|---|
| `generate()` | Trả text assistant cho parser hiện tại; không trả object vLLM, prompt gốc hoặc đáp án đã tự parse |
| `max_new_tokens=None` | Dùng mặc định của Settings; override dùng cho preparation/schema repair phải có hiệu lực |
| `remaining_seconds` | Thời gian còn lại tại thời điểm gọi; ≤0 không đưa request mới vào engine |
| `count_tokens()` | Số token input thực tế từ **cùng formatter/tokenizer** dùng khi generate, có generation prompt và cờ thinking |
| `runtime_info()` | Dict JSON-serializable: backend, contract/protocol version, checkpoint/revision, phiên bản package, GPU/dtype, cấu hình engine, chat-template hash và EOS/pad hiệu lực |
| `last_generation` | Dict của lượt gần nhất: input/output tokens, seconds, stop_reason, max_new_tokens, EOS/pad hiệu lực, hash raw output; không giữ object/tensor engine |
| `close()` | Hủy request còn sống và giải phóng engine/worker; gọi lặp lại an toàn |

Thêm `GenerationTimeoutError` và `BackendUnavailableError` kế thừa `DataMateError` tại module này. Timeout của một request là lỗi có thể kết thúc câu; engine chết/OOM không phục hồi là lỗi nghiêm trọng phải truyền ra để dừng batch, không biến tất cả câu sau thành abstain.

Giữ re-export `ChatBackend` tại `models/backends.py` cho import cũ. Cập nhật Transformers, DemoBackend, wrapper và fake backend theo contract. Demo không cần tokenizer thật: `count_tokens()` phải báo không hỗ trợ nếu bị gọi; `rag.mode != none` với demo bị validation từ chối. `close()` của demo là no-op.

### 3.2. `models/chat_format.py` — tạo mới

Tách helper dùng chung: `encode_messages(tokenizer, messages) -> list[int]`, token counter và xử lý EOS/pad. Giữ đường import helper cũ bằng re-export nếu test/công cụ đang dùng.

- Giữ `apply_chat_template`, `add_generation_prompt=True`, `enable_thinking=False` và xử lý special tokens của baseline. Không tokenize lại chuỗi đã có special token theo cách thêm token lần hai.
- Dùng checkpoint generation config + tokenizer và helper `generation_token_settings()` hiện có; xác minh EOS/turn terminator nằm trong vocabulary. Không hard-code ID token của một model cho mọi model.
- `count_tokens()` và `generate()` dùng cùng dãy input IDs. Đối chiếu cả Qwen 9B và checkpoint đang chọn trong benchmark; không thay checkpoint để né lỗi adapter.
- Cập nhật adapter Transformers thật gọn; giữ `TransformersBackend` trong `backends.py` để công cụ kiểm version qua AST hiện tại còn tìm được, hoặc sửa đồng bộ công cụ đó.

### 3.3. `models/runtime.py` — tạo mới

Cung cấp `runtime_identity(settings) -> dict` đọc được **trước khi load weights**: backend, option ảnh hưởng generation, contract/formatter version, phiên bản package đang cài. Đọc package bằng metadata, không import vLLM để chạy demo/Transformers.

Phần identity ổn định không chứa thời gian, request ID hoặc GPU-memory-used. `runtime_info()` bổ sung snapshot phần cứng thực tế vào log. Thịnh dùng helper này trong application/resume; Hải dùng trong cache identity của preparation/schema repair. File mới của adapter/formatter và option engine phải tham gia identity, tránh dùng lại cache tạo bằng engine khác.

## 4. Cấu hình chung — Hải sửa `src/datamate/config.py`

Thêm hai dataclass `VLLMSettings` và `RAGSettings` **ngay trong config.py**, không tạo thêm bản cấu hình RAG trùng ở package retrieval. `Settings` có `vllm`/`rag` bằng `default_factory`. Chốt trường với Thịnh trong PR nền H:

| Nhóm | Trường / giá trị khởi đầu để thử |
|---|---|
| `vllm` | `tensor_parallel_size=1`, `gpu_memory_utilization=0.70`, `max_num_seqs=1`, `enable_prefix_caching=False`, `dtype="auto"` |
| `rag` chọn mode/model | `mode="none"`; `embedding_key="vietnamese-embedding"`, `embedding_dir="models/vietnamese-embedding"`; `reranker_key="vietnamese-reranker"`, `reranker_dir="models/vietnamese-reranker"` |
| `rag` chunk/top-k | `chunk_tokens=384`, `chunk_overlap=48`, `candidate_k=20`, `context_k=5`, `context_tokens=2048` |
| `rag` encoder | `embedding_max_tokens=2048`, `rerank_max_tokens=2304`, `rerank_query_max_tokens=256`, `embedding_batch_size=8`, `rerank_batch_size=4` |
| `rag` tài nguyên | `timeout_seconds=8`, `device="cuda:0"`, `dtype="auto"`, `index_dir=None` → suy ra từ `cache_dir/retrieval` |

Các số trên là **preset thử đầu**, chỉnh theo GPU/benchmark rồi ghi lại; không phải ngưỡng hiệu năng bắt buộc. `max_context_tokens` và `max_new_tokens` vẫn là trường gốc, không tạo hai bản trong config vLLM.

Yêu cầu kỹ thuật:

- `Settings.load()` nhận nested JSON; khởi tạo Python bằng dict hoặc dataclass đều chuẩn hóa và validate được. `to_dict()` trả nested dict, round-trip ổn định, preset cũ thiếu hai nhóm vẫn chạy.
- Kiểm tra type/range, enum, overlap < chunk size, chunk size ≤ embedding limit, `context_k <= candidate_k`, query limit < rerank pair limit, output budget < context và đường ghi ngoài dataset; báo trường sai trước khi load GPU. Không bỏ qua key lạ.
- `rag.mode=none` không đòi file encoder; `dense` chỉ dùng embedding; `dense_rerank` dùng cả hai. Thịnh thực hiện kiểm tổng inventory/file artifact ở composition.
- Không mặc định chuyển mọi chế độ quantization/speculative hiện có của Transformers sang vLLM. Lập bảng tổ hợp hỗ trợ của phiên bản đã kiểm, báo lỗi sớm khi không hỗ trợ, không âm thầm rơi về backend khác.
- CLI `--backend` thêm `vllm`; option chi tiết lấy từ JSON để tránh trùng hàng loạt cờ.

Preset riêng cần tạo: `configs/experiments/hai/transformers-baseline.json` và `vllm.json`. Cùng generator đang đo; cả hai đặt `rag.mode=none` khi đánh giá riêng vLLM. Ví dụ phần mới trong preset vLLM:

```json
{
  "backend": "vllm",
  "vllm": {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.70,
    "max_num_seqs": 1,
    "enable_prefix_caching": false,
    "dtype": "auto"
  },
  "rag": {"mode": "none"}
}
```

## 5. Các file Hải cần code

`Mới` là file chưa tồn tại; `Sửa` là điểm ghép vào runtime hiện tại. Contract/config ở trên làm trước; bảng này là phần implementation tiếp theo.

| File | Loại | Chức năng / việc cần hoàn thành |
|---|---|---|
| `src/datamate/models/vllm_backend.py` | Mới | `VLLMBackend` theo ChatBackend: verify artifact, tạo engine một lần, generate/count/runtime/close |
| `src/datamate/models/backends.py` | Sửa | Factory tường minh cho ba backend; chỉ import VLLMBackend trong nhánh `vllm`; backend lạ phải báo lỗi |
| `src/datamate/agent/preparation.py` | Sửa | Cache identity gồm backend/runtime/formatter; truyền token override; lỗi engine nghiêm trọng phải được re-raise |
| `src/datamate/agent/schema_repair.py` | Sửa | Dùng identity mới, cùng contract và nguyên tắc lỗi; không sửa prompt/logic ánh xạ nghiệp vụ |
| `src/datamate/cli.py` | Sửa | Chấp nhận backend mới; lệnh `ask` truyền `business_ids/remaining_seconds` cho `create_agent()` và đóng Agent trong `finally` |
| `tools/diagnose_train.py` | Sửa | `runtime_snapshot()` dùng `runtime_info()`, bỏ phụ thuộc `backend.model/.torch`; TimedBackend chuyển tiếp đủ contract |
| `tools/colab_runtime_check.py` | Sửa | Nhận biết contract/adapter đúng version; không yêu cầu import vLLM khi chạy baseline |
| `pyproject.toml` + lock GPU mới | Sửa/Mới | Extra vLLM và extra RAG theo yêu cầu Thịnh; khóa bộ dependency đã smoke chung trên Linux GPU |
| `tools/create_inference_notebook.py`, `create_team200_colab.py` | Sửa | Chọn preset/extras, ghi môi trường; sinh lại notebook tương ứng, không đặt engine trong notebook |
| `tools/package_colab.py` | Sửa nếu cần | Source bundle chứa module/config mới; vẫn loại dataset/weights/output/archive |
| `docker/Dockerfile.competition` | Sửa | Cài từ wheelhouse offline, đúng extras do build context chỉ định; không tải lúc inference |
| `configs/experiments/hai/*.json` | Mới | Preset baseline/vLLM có đường dẫn và run name riêng |
| `docs/reports/hai-vllm-<run>.md` | Mới sau đo | Lệnh tái chạy, môi trường/phiên bản, bộ câu, chất lượng/thời gian trước và sau |

Không chuyển cả `models/` thành vùng riêng của Hải: `registry.py/artifacts.py` do Thịnh sửa. Không chỉnh notebook training hoặc các ZIP lịch sử đã khóa để thêm vLLM.

## 6. Chi tiết adapter vLLM

### Khởi tạo và dependency

1. Trên GPU Linux, đối chiếu `architectures` trong **checkpoint thực tế** với [danh sách model hỗ trợ](https://docs.vllm.ai/en/latest/models/supported_models/), chọn phiên bản vLLM/PyTorch/CUDA tương thích rồi khóa bản đã chạy. Chưa chốt số phiên bản khi chưa smoke.
2. Verify checkpoint local bằng `verify_artifact()`; tokenizer/config cũng phải nằm local. Giữ offline và `trust_remote_code=False`. Không tải model/tokenizer từ mạng khi inference.
3. Engine dùng đường dẫn checkpoint, `max_model_len=settings.max_context_tokens` và các option đã validate. Không load thêm nguyên model Transformers chỉ để lấy tokenizer/generation config.
4. Mỗi backend giữ một engine, tái dùng qua các câu và lượt tool. Bản đầu chạy tuần tự, `max_num_seqs=1`; chưa đổi batch/loop Agent để tránh lẫn hiệu ứng tối ưu.
5. Dùng engine async có API hủy request, bọc thành contract đồng bộ hiện tại bằng event loop/worker sống suốt vòng đời backend. Chi tiết API phải đối chiếu phiên bản đã khóa, tham khảo [AsyncLLM](https://docs.vllm.ai/en/latest/api/vllm/v1/engine/async_llm/). Trong Colab không gọi `asyncio.run()` trực tiếp trên event loop đang chạy.

### Generate, context và timeout

- Dãy token input lấy từ shared formatter. Chế độ baseline tương đương greedy: `temperature=0`, một output, seed đã ghi; ánh xạ token budget thành sampling params của phiên bản đang dùng.
- Kiểm `input_tokens + effective_max_new_tokens <= max_context_tokens` trước khi submit. Quá context báo lỗi có thông tin; không tự cắt rulebook, câu hỏi hoặc tăng giới hạn.
- EOS/turn terminator lấy từ helper đã kiểm; đối chiếu cách vLLM trả stop token và special tokens với parser hiện có. Chỉ decode phần output, không để adapter tự thêm reasoning/JSON sửa lỗi.
- Mỗi request có ID nội bộ duy nhất, không suy luận đáp án từ question ID. Hết `remaining_seconds` phải abort request và xác nhận thu hồi khỏi engine; dùng timeout Future mà bỏ GPU chạy nền là chưa đạt.
- Nếu abort thất bại/engine hỏng, ném `BackendUnavailableError`, dọn worker trong `close()`. Thịnh cập nhật catch ở runner/application để lỗi này dừng batch và giữ các dòng đã flush.
- `self_review` vẫn do Agent điều khiển qua cùng generate. Speculative/quantization chỉ bật khi adapter có đường hỗ trợ đã thử; không tự ánh xạ bitsandbytes của Transformers sang một định dạng khác.
- Prefix caching, grammar và batching là bước thử sau khi đường cơ bản đúng. Mỗi thay đổi có preset/run riêng; không sửa baseline cùng lúc rồi gán toàn bộ cải thiện cho engine.

### Phần ghép do Thịnh thực hiện

Hải bàn giao API và test stub, Thịnh sửa đúng file của mình:

| Điểm ghép | Yêu cầu Hải cần nêu rõ trong PR |
|---|---|
| `application.create_agent()` | Gọi factory; nhận thêm keyword `business_ids/remaining_seconds`; kiểm inventory trước load, prepare retriever trong budget; trả Agent có `close()` |
| `application.run_batch()` | Thêm runtime identity trước resume, snapshot sau load; `finally` đóng Agent; load/prepare nằm trong wall time |
| `agent/runner.py` | Re-raise BackendUnavailableError ở các catch liên quan; timeout thường có log và kết thúc câu theo ngân sách |
| `tools/infer_team200.py` | Nếu dùng benchmark này, metadata/resume dùng runtime schema mới và toàn inventory; không để sót RAG models |
| `tools/package_solution.py` | Ghi extras/profile cần cài vào build context để Dockerfile của Hải sử dụng |

### VRAM khi ghép RAG

Hải và Thịnh ghi cùng một resource profile: GPU/dung lượng, dtype encoder, batch, context và phần VRAM dành cho vLLM. `gpu_memory_utilization=0.70` chỉ là điểm thử, không bảo đảm vừa mọi GPU. Kiểm phần còn lại đủ embedding/reranker, peak activation và dự phòng; khởi tạo theo thứ tự đã kiểm để profiling engine không chiếm hết VRAM. Nếu không vừa, giảm budget/context/batch hoặc dùng GPU phù hợp rồi đo lại, không âm thầm đổi model.

## 7. Test và bàn giao

Đây là các kiểm thử **cần viết/chạy khi triển khai**, không phải kết quả đã có:

| Test của Hải | Tình huống chính |
|---|---|
| `tests/test_backend_contract.py` | TF/demo/fake/wrapper đúng chữ ký, override, lifecycle; lazy import không cần vLLM |
| `tests/test_vllm_adapter.py` | Engine giả: sampling, token IDs, EOS, context overflow, abort, engine chết; không cần weights thật |
| `tests/test_settings_extensions.py` | JSON cũ/mới, nested round-trip, key/range sai, mode không hỗ trợ |
| `tests/test_runtime_identity.py` | Đổi engine/formatter/package/config làm đổi identity; không có field biến động |
| Test hiện có | Hải cập nhật `tests/conftest.py` nếu fake backend chung cần đổi; `test_generation_protocol.py`, `test_model_loading.py`, `test_train_diagnostic.py` và nhóm `test_colab_*.py` |

Chạy kiểm thử CPU phù hợp, sau đó smoke checkpoint thật trên GPU: generate action, preparation/repair có token override, EOS/context, timeout và không còn request nền. Xong mới đo trước/sau theo mục 2 và smoke bản ghép với RAG. Giữ output thô/log/preset có thể tái chạy; PR nêu check nào chưa chạy. `tests/test_team_layout.py` do Thịnh giữ để cập nhật packaging; nếu thay notebook làm test ở đó cần đổi, đề nghị Thịnh sửa thay vì cùng sửa file.

Bàn giao: PR vào `main`, lock môi trường GPU, hai preset, báo cáo benchmark, lệnh CLI/notebook và hướng dẫn chuyển về Transformers. Cùng Thịnh build/chạy thử Docker **mạng tắt**; nộp Docker image/checksum ở bước cuối theo [hướng dẫn Docker](../docker.md), không coi gói source ZIP là image.

Tuân thủ [ngữ cảnh cuộc thi](../competition-context.md): tổng mọi model inference ≤15B; checkpoint trọng số công khai trước 01/10/2026, pin revision/hash/license. Không dùng test để training/thu thập đáp án; `/data` chỉ đọc, output `/result/submission.jsonl`, log `/result/logs/`, scratch `/work`. Toàn private test ≤6 giờ kể cả load/index; xác minh base image/guideline BTC trước chốt image.
