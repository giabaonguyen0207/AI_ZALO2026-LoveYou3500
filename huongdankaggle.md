# Chạy và so sánh hoàn toàn trên Kaggle

Làm theo thứ tự dưới đây. Toàn bộ code trong tài liệu chạy ở **Code cell của Kaggle**. Mỗi notebook GPU chỉ cần gắn **một Input là kit**; nó tự cài thư viện, tải model rồi chạy. Không cần notebook chuẩn bị, Dataset model hay bộ thư viện riêng.

Thời gian tải/cài được ghi trong log, **không cộng vào thời gian so sánh hiệu năng**. Phép đo bắt đầu khi gọi pipeline: báo riêng thời gian load và thời gian xử lý câu hỏi. Vòng agent và logic nghiệp vụ giữ nguyên.

## 1. Upload bộ chạy một lần

Trên Kaggle, tạo **Dataset private** rồi upload **một file `kaggle-runtime-kit.zip`** đã chuẩn bị ở `outputs/packages/` trong dự án. Thao tác bằng trình duyệt, không cần gõ lệnh. Nếu đã có Dataset kit, cập nhật chính Dataset đó bằng file ZIP mới, rồi chọn phiên bản mới trong hai notebook GPU.

Kaggle giải nén archive khi upload Dataset, giữ cấu trúc thư mục. Vì vậy kit mới chứa thẳng `current/` (source mới), `baseline/` (source cũ), `benchmark/` (dataset/IDs) và `kaggle_kit.json`; **không chứa ZIP lồng và không để hai source đè lên nhau**. Code tìm manifest thư mục, đồng thời nhận được cả trường hợp Input vẫn giữ file ZIP. Không phụ thuộc tên thư mục mà Kaggle tạo. [Tài liệu upload Kaggle](https://www.kaggle.com/docs/datasets).

Nếu cập nhật dataset đã có, tạo phiên bản mới và chọn phiên bản đó ở các notebook dưới. Chỉ gắn một phiên bản kit. Hai ZIP source riêng vẫn được giữ để kiểm tra, nhưng không phải upload chúng cho quy trình này.

Lộ trình cần theo:

| Thứ tự | Làm gì? | Có cần canh không? | Khi nào đi tiếp? |
|---|---|---|---|
| 2 | Tạo hai notebook GPU, gắn kit | Chỉ thao tác cấu hình; chưa chạy | Hai notebook có cùng phiên bản kit, T4 × 2 và Internet bật |
| 3 | Chạy thử 4 câu, GPU | **Có**, đây là lượt phát hiện lỗi | Cả Torch và vLLM đều có pilot status=passed |
| 4 | Chạy chính 24 câu, GPU | Sau khi Submit Save & Run All có thể rời máy | Hai version chạy xong; đủ 24 câu thì đọc được báo cáo cả khi needs_review |
| 5 | Mở kết quả hai version | Chỉ kiểm status và tên file | Có đủ hai main_result.zip |
| 6 | So sánh, CPU | Chạy ngắn; không cần GPU | Đọc comparison.txt / comparison.csv |

**Pilot và main là hai lượt khác nhau.** Main không chạy lại bốn câu pilot. Save & Run All tạo phiên sạch, tự cài thư viện/tải model/đọc Input/đo context trước khi chạy; không dùng các biến hay venv còn sót từ lượt tương tác. Quick Save không chạy lại notebook. Đây là hai chế độ khác nhau trong [hướng dẫn chính thức Kaggle](https://www.kaggle.com/docs/notebooks).

## 2. Tạo hai notebook GPU, không chuẩn bị tài nguyên riêng

Tạo notebook **hai-torch-runtime**:

1. Add Input **dataset kit ở bước 1**, chọn phiên bản mới nhất vừa upload. Không cần gắn Output của notebook prepare_model cũ.
2. Chọn Accelerator **GPU T4 × 2**, bật **Internet**.
3. Import `notebooks/DataMate_Kaggle_Runtime_Benchmark.ipynb`, hoặc copy hai cell ở bước 3.
4. Chưa Save & Run All ngay; chạy pilot ở bước 3 trước để phát hiện lỗi.

Tạo notebook **hai-vllm-runtime** tương tự, cùng phiên bản kit, cùng T4 × 2 và Internet bật. Trong cell cấu hình của notebook này đổi `ARM = 'vllm'`. Hai notebook dùng hai phiên GPU riêng; không chạy hai engine tranh cùng hai card trong một notebook.

Nếu đã có notebook/output **hai-model-prepare**, giữ lại tùy nhu cầu khác; quy trình kiểm thử này không dùng đến và không cần tải lại nó thành Dataset. Bản kit mới không còn bước chuẩn bị CPU. Không gắn kit cũ và mới cùng lúc.

## 3. Chạy thử bốn câu và theo dõi

Trong notebook **hai-torch-runtime** đã tạo ở bước 2, dùng hai Code cell dưới. Cả hai notebook GPU đều chỉ gắn kit; model sẽ tự tải đúng revision đã khóa khi chạy Cell 2.

**Cell 1:** chỉ sửa hai dòng đầu nếu cần. Lượt này để `torch` và `pilot`.

<!-- notebook:benchmark:start -->
```python
ARM = 'torch'     # notebook Torch: 'torch'; notebook vLLM: 'vllm'
PHASE = 'pilot'   # lần thử: 'pilot'; lần chạy chính: 'main'

SETTINGS = {}     # để nguyên profile ban đầu trên hai T4
```

**Cell 2:** copy nguyên block, bấm nút chạy bên trái cell. Chờ nó kết thúc trước khi làm bước 4.

```python
TASK = 'benchmark'
OPTIONS = dict(SETTINGS, arm=ARM, phase=PHASE)

import hashlib, json, runpy, shutil, traceback, zipfile
from pathlib import Path

try:
    inputs = Path(OPTIONS.get('input_root', '/kaggle/input'))
    bootstrap = Path(OPTIONS.get('bootstrap_root', '/tmp/datamate-bootstrap'))
    manifests = list(inputs.rglob('kaggle_kit.json'))
    if not manifests:
        packets = list(inputs.rglob('kaggle-runtime-kit.zip'))
        assert len(packets) == 1, 'Add Input cần đúng một bộ kaggle-runtime-kit mới'
        with packets[0].open('rb') as handle:
            packet_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        expanded = bootstrap / 'upload' / packet_hash[:16]
        expanded.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(packets[0]) as archive:
            assert all((expanded/m.filename).resolve().is_relative_to(expanded.resolve()) for m in archive.infolist())
            archive.extractall(expanded)
        manifests = [expanded/'kaggle_kit.json']
    assert len(manifests) == 1, 'Chỉ gắn một phiên bản kit, không gắn trùng'
    kit = manifests[0].parent
    metadata = json.loads(manifests[0].read_text())
    assert metadata['format'] == 'datamate-kaggle-kit-v1'
    for relative, expected in metadata['files'].items():
        path = (kit/relative).resolve()
        assert path.is_relative_to(kit.resolve()), 'Đường dẫn kit không hợp lệ'
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected, relative
    source_hash = metadata['source_archives']['current']['sha256']
    toolkit = bootstrap / 'toolkit' / source_hash[:16]
    if not toolkit.exists():
        shutil.copytree(kit/'current', toolkit)
    workflow = runpy.run_path(str(toolkit/'tools/kaggle_workflow.py'))
    OPTIONS['current_zip_sha256'] = source_hash
    OPTIONS['kit_root'] = str(kit)
    STATUS = workflow['entrypoint'](TASK, **OPTIONS)
    print(json.dumps(STATUS, ensure_ascii=False, indent=2))
except BaseException:
    output = Path(OPTIONS.get('output_root', '/kaggle/working'))
    output.mkdir(parents=True, exist_ok=True)
    (output/'bootstrap_error.log').write_text(traceback.format_exc(), encoding='utf-8')
    raise
```
<!-- notebook:benchmark:end -->

Cell 2 tự chạy hết theo thứ tự sau; không cần bấm thêm cell giữa chừng:

| Giai đoạn in trong cell/log | Việc đang làm | Có tính vào số đo hiệu năng không? |
|---|---|---|
| `install_libraries`, `record_library_versions` | Cài môi trường riêng, ghi phiên bản; cả hai bên cài cùng nhóm thư viện để so sánh | Không |
| `check_two_t4_devices` | Kiểm hai GPU T4 và phép tính FP16 nhỏ | Không |
| `download_and_verify_model`, `model_ready` | Tải Qwen9B đúng revision, kiểm weights/manifest | Không |
| `measure_actual_context_without_weights` | Đếm token qua prompt thật, chọn context phù hợp trước khi nạp model | Không |
| `run_pilot_questions` hoặc `run_main_questions` | Gọi pipeline gốc: nạp weights lên GPU rồi chạy các agent loop | Có; tách riêng load và inference trong kết quả |

Không cần tìm/sửa đường dẫn tên dataset Kaggle: code tìm kit theo manifest. Thư viện và model nằm dưới `/tmp/datamate-kaggle/`, không nằm trong Output. Nếu pilot và main chạy cùng một phiên, checkpoint tạm có thể được dùng lại; saved version sạch tự tải lại. Đây là thao tác tự động, không có bước chuẩn bị riêng.

Lượt thử cần theo dõi để phát hiện lỗi sớm. Thư viện/download chậm không phải tốc độ agent. Nếu cell báo lỗi, **không chuyển sang main**; xem file status/log dưới `/kaggle/working` trong vùng Files của phiên đang chạy. Nếu lỗi trước khi nạp được công cụ, có `bootstrap_error.log`.

Trong notebook thứ hai **hai-vllm-runtime**, gắn **cùng kit/phiên bản**, cùng T4×2 và Internet bật. Copy đúng hai cell trên, đổi **duy nhất** `ARM = 'vllm'`, giữ `PHASE = 'pilot'`, rồi chạy Cell 1 và Cell 2. Nếu Kaggle chưa cấp được cả hai phiên GPU đồng thời, chạy thử lần lượt.

Kiểm kết quả lượt thử:

| Notebook | File phải mở | Điều kiện đi tiếp |
|---|---|---|
| Torch | `torch_pilot_status.json` | `state: "passed"` và `questions: 4` |
| vLLM | `vllm_pilot_status.json` | `state: "passed"` và `questions: 4` |

`passed` xác nhận lượt đo sạch theo kiểm tra vận hành, **chưa xác nhận đáp án đúng**. `needs_review` nghĩa là pipeline đã ghi đủ câu nhưng có cảnh báo, chẳng hạn timeout hoặc toàn bộ abstain; mở status/metrics và log để xem trước khi tốn quota cho main. `failed` dành cho lỗi thực thi hoặc không hoàn thành đủ câu; mở `torch_pilot_run.log` hoặc `vllm_pilot_run.log`.

## 4. Gửi lượt chính 24 câu, rồi có thể làm việc khác

Chỉ làm sau khi cả hai pilot đạt. Thực hiện lần lượt trên hai notebook GPU:

1. Dừng **phiên tương tác** vừa dùng để chạy pilot trong bảng Session. Việc này giải phóng GPU đang giữ. Không bấm Cancel vào một saved version đang chạy.
2. Trong Cell 1, đổi `PHASE = 'main'`. Giữ ARM đúng notebook. Không bấm chạy lại Cell 2 trong phiên tương tác.
3. Chọn **Save Version → Save & Run All**. Trong cài đặt của version, kiểm **GPU T4 × 2**, Internet bật và **một Input là kit** đã gắn từ trước. Submit version.
4. Khi version hiện Queued/Running, có thể đóng trang và làm việc khác. Phiên Save & Run All chạy riêng, từ đầu đến cuối; không cần chờ ở trình duyệt để bấm cell tiếp theo.
5. Gửi tương tự notebook còn lại. Hai saved runs có thể chạy song song nếu Kaggle cấp đủ GPU/quota; nếu không, chờ lượt đầu xong rồi gửi lượt thứ hai.

**Không dùng Quick Save cho bước này:** nó chỉ lưu trạng thái notebook, không thực thi 24 câu. Không bấm Run All thêm trong editor trong lúc saved run đang chạy, vì sẽ tạo thêm việc tiêu quota.

Main dùng 24 câu cố định; cùng checkpoint, FP16 và pipeline cũ. Budget của riêng run_batch là 45 phút; cài thư viện, tải model và đo context diễn ra trước, có timeout riêng và không nằm trong thời gian so sánh. Đây là mức dừng, không phải thời gian dự kiến. Tổng thời gian phiên/quota Kaggle vẫn bao gồm các bước tải/cài vì GPU đã được cấp cho phiên. Source mới chỉ thay engine/điều phối và đo đạc, không đổi plan/tool/SQL/review/final.

## 5. Quay lại: mở đúng version và lấy đúng tên file

Khi hai version kết thúc, vào từng notebook và **chọn đúng version main vừa gửi**, không đọc Output của một version pilot/phiên bản cũ. Kiểm các file do code tự tạo dưới `/kaggle/working`, được hiển thị trong Output:

| Torch main | vLLM main | Cần làm gì? |
|---|---|---|
| `torch_main_status.json` | `vllm_main_status.json` | `questions: 24`; `passed` là lượt sạch, `needs_review` vẫn đọc được báo cáo chẩn đoán |
| `torch_main_run.log` | `vllm_main_run.log` | Mở nếu status failed hoặc version báo lỗi |
| **`torch_main_result.zip`** | **`vllm_main_result.zip`** | Hai file cần cho bước so sánh |

Đây là tên file **chính code trong bộ này ghi ra**, không phải giả định về cách Kaggle đặt tên file download. Không cần biết Kaggle có tự đóng một ZIP tổng hay không: ở Output, chọn file có đúng tên trên.

Code cố ghi status/log và result ZIP cả khi gặp lỗi Python/subprocess. Nếu Kaggle giết toàn bộ kernel, hết đĩa hoặc phiên chưa chạy được bootstrap, có thể thiếu file hoặc status vẫn `running`; khi đó xem log thực thi của saved version và không coi lượt đó là hoàn tất. Không tự điền thời gian 0 hay dùng kết quả thiếu câu.

## 6. So sánh ngay trên Kaggle bằng CPU

**Không chạy file log.** `.log` là nhật ký mở ra đọc khi có lỗi. File `.zip` kết quả chứa các số đo để chương trình so sánh đọc tự động. Sau hai lượt main, việc tiếp theo là chạy **notebook so sánh CPU**, không phải chạy lại agent hay chép log vào code.

Tạo notebook thứ ba tên **hai-runtime-compare**:

1. Accelerator **None**; không cần GPU.
2. Add Input dataset kit ở bước 1, chọn phiên bản mới.
3. Add Input **Output của version main đã hoàn tất** từ `hai-torch-runtime` và `hai-vllm-runtime`. Kiểm trong Input thấy `torch_main_result.zip` và `vllm_main_result.zip`.
4. Import `notebooks/DataMate_Kaggle_Compare.ipynb`, hoặc copy một cell dưới và chạy. Code tự tìm/giải nén đúng hai file; không cần đoán đường dẫn mount của Kaggle.

<!-- notebook:compare:start -->
```python
TASK = 'compare'
OPTIONS = {}  # so timing; chưa đọc nhãn/answers

import hashlib, json, runpy, shutil, traceback, zipfile
from pathlib import Path

try:
    inputs = Path(OPTIONS.get('input_root', '/kaggle/input'))
    bootstrap = Path(OPTIONS.get('bootstrap_root', '/tmp/datamate-bootstrap'))
    manifests = list(inputs.rglob('kaggle_kit.json'))
    if not manifests:
        packets = list(inputs.rglob('kaggle-runtime-kit.zip'))
        assert len(packets) == 1, 'Add Input cần đúng một bộ kaggle-runtime-kit mới'
        with packets[0].open('rb') as handle:
            packet_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        expanded = bootstrap / 'upload' / packet_hash[:16]
        expanded.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(packets[0]) as archive:
            assert all((expanded/m.filename).resolve().is_relative_to(expanded.resolve()) for m in archive.infolist())
            archive.extractall(expanded)
        manifests = [expanded/'kaggle_kit.json']
    assert len(manifests) == 1, 'Chỉ gắn một phiên bản kit, không gắn trùng'
    kit = manifests[0].parent
    metadata = json.loads(manifests[0].read_text())
    assert metadata['format'] == 'datamate-kaggle-kit-v1'
    for relative, expected in metadata['files'].items():
        path = (kit/relative).resolve()
        assert path.is_relative_to(kit.resolve()), 'Đường dẫn kit không hợp lệ'
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected, relative
    source_hash = metadata['source_archives']['current']['sha256']
    toolkit = bootstrap / 'toolkit' / source_hash[:16]
    if not toolkit.exists():
        shutil.copytree(kit/'current', toolkit)
    workflow = runpy.run_path(str(toolkit/'tools/kaggle_workflow.py'))
    OPTIONS['current_zip_sha256'] = source_hash
    OPTIONS['kit_root'] = str(kit)
    STATUS = workflow['entrypoint'](TASK, **OPTIONS)
    print(json.dumps(STATUS, ensure_ascii=False, indent=2))
except BaseException:
    output = Path(OPTIONS.get('output_root', '/kaggle/working'))
    output.mkdir(parents=True, exist_ok=True)
    (output/'bootstrap_error.log').write_text(traceback.format_exc(), encoding='utf-8')
    raise
```
<!-- notebook:compare:end -->

Nếu không tìm thấy đúng một file mỗi tên, cell sẽ báo các file đang thấy. Quay lại Add Input chọn đúng version main, bỏ nguồn bị gắn trùng. Không sửa tên file của pilot thành main để vượt kiểm tra.

Kết quả so sánh được in ngay trong cell và ghi rõ:

| File trong `/kaggle/working` của notebook CPU | Nội dung |
|---|---|
| **`comparison.txt`** | Kết luận dễ đọc: điều kiện nào chưa qua, thời gian và speedup |
| **`comparison.csv`** | Bảng Transformers/vLLM để xem hoặc đưa vào báo cáo |
| `comparison.json` | Đầy đủ số liệu, điều kiện so sánh và ID bất đồng |
| `comparison_status.json`, `comparison_run.log` | Trạng thái/lỗi của việc ghép kết quả |
| `comparison_result.zip` | Gói các file so sánh trên để lưu lại |

Ví dụ thao tác sau khi đã chạy GPU:

1. Vào version main của notebook Torch. Mở `torch_main_status.json`: xem số câu và `passed`/`needs_review`. Kit mới nhận cả hai để tạo báo cáo; trạng thái cảnh báo không bị đổi thành passed.
2. Làm tương tự notebook vLLM. Với ZIP cũ có `failed` sau bước `run_main_questions` nhưng metrics/submission vẫn đủ 24 câu, cập nhật kit mới rồi chạy notebook so sánh CPU; không cần chạy lại GPU chỉ để đọc các số đo đã có. ZIP thiếu metrics, thiếu câu hoặc hỏng vẫn không được coi là lượt so sánh sạch.
3. Mở notebook CPU **hai-runtime-compare**, Add Input Output của **hai version ở bước 1–2**. Hai file đầu vào cần có là `torch_main_result.zip` và `vllm_main_result.zip`. Không cần tự giải nén hay chọn từng log con.
4. Chạy cell của `DataMate_Kaggle_Compare.ipynb`. Cell tự đọc số đo trong hai ZIP; nó không nạp model và không xử lý lại 24 câu.
5. Đọc phần chữ in cuối cell hoặc mở `comparison.txt`. Muốn lấy bảng đưa cho nhóm thì dùng `comparison.csv`. File `comparison.json` dành cho xem chi tiết khi có điều kiện chưa đạt hoặc đáp án khác nhau.

Nếu đọc `comparison.txt` thấy “Chưa đủ điều kiện” hoặc một điều kiện chưa đạt, chưa được dùng ratio thời gian để khẳng định vLLM nhanh hơn. Đọc lý do được ghi bằng tiếng Việt và mở log/metrics của lượt tương ứng.

Muốn lưu Output để quay lại sau, dùng Save Version → Save & Run All của notebook CPU này; nó tự chạy lại từ hai Input đã lưu, không chạy lại GPU. Có thể tải trực tiếp `comparison_result.zip` và hai main_result.zip từ Output nếu muốn giữ bản riêng. Không có bước bắt buộc chạy so sánh trên máy cá nhân.

## 7. Đọc kết quả theo thứ tự này

1. Mở **comparison.txt**, xem điều kiện so sánh có hợp lệ không. Khác IDs/dataset/checkpoint/precision/thư viện/prompt đầu, sai backend, thiếu câu hoặc có deadline thì không được kết luận speedup sạch. Báo cáo vẫn giữ tất cả câu và phân loại nguyên nhân; `needs_review` không phải engine bị crash.
2. Xem **Inference giây** và **Tổng run_batch giây**. Speedup = thời gian Torch / thời gian vLLM: >1 là vLLM nhanh hơn trong lượt này, <1 là chậm hơn. Tổng run_batch còn tính cả kiểm hash/load model; **không gồm tải model, cài thư viện hay context preflight**. Không lấy `elapsed_seconds` trong status hoặc thời lượng toàn notebook làm tốc độ inference: các số đó có cả phần chuẩn bị tự động.
3. Xem câu/phút, latency p50/p95, số calls và token output. Sinh ít token, abstain nhiều hoặc làm ít lượt model có thể khiến thời gian ngắn hơn; đó chưa tự là tối ưu tốt.
4. Xem **Token cache hit**: số lấy trực tiếp từ vLLM, không suy từ việc có hai request chạy đồng thời. `null` là chưa có số đo, không phải 0%. Generate đồng thời cao nhất chỉ là overlap ở caller, không phải batch size nội bộ engine.
5. Không có nhãn thì chỉ kết luận timing/agreement. Hai đáp án giống nhau không đồng nghĩa đều đúng. Muốn kết luận nhanh hơn mà không giảm chất lượng cần chấm nhãn TRAIN sau inference; không đưa nhãn vào Input của agent.

Bản kiểm thử mới tách tỷ số thời gian quan sát của toàn pipeline khỏi speedup sạch. Khi có timeout, tỷ số quan sát vẫn được in nhưng `inference_speedup` giữ null. `comparison.json` có `per_question`: thời gian trước lần gọi model đầu, tổng thời gian gọi model, phần thời gian còn lại, khác biệt có deadline hay hai bên trả đáp án khác nhau, action phân kỳ đầu tiên và khác biệt phản hồi tool. Phần ngoài generate gồm chuẩn bị dữ liệu, chờ lock và chạy tool; số đo này không tự phân biệt từng nguyên nhân đó. Không bỏ các câu chậm/lỗi khỏi bảng.

Mặc định vẫn 90 giây/câu. Nếu cần một phép đo với ngân sách khác, có thể đặt `SETTINGS = {'question_timeout_seconds': 180}` ở **cả hai** notebook; đó là lượt thử khác, phải giữ nguyên cùng giá trị ở hai bên. Không đổi budget một bên hoặc chuyển đồng hồ ra sau bước chuẩn bị dữ liệu. Phép đo hiện tại là toàn pipeline với cách điều phối khác nhau; chưa phải phép đo engine trên chuỗi request cố định.

Hai result ZIP còn có thư mục `details/...` chứa `benchmark_metrics.json`, `benchmark_identity.json`, `model_calls.jsonl`, `question_timing.jsonl`, `gpu_samples.jsonl`, `submission.jsonl` và log từng câu. Không cần tự tải từng file để so sánh: notebook CPU đọc chúng trong ZIP. Nếu debug một ID, tìm `id` trong log JSON; vLLM cache count ở metadata generation. GPU/RAM là số lấy mẫu, có thể bỏ lỡ đỉnh ngắn.

## 8. Cấu hình và giới hạn của phép đo này

- Bộ chính có 24 câu, 4 câu/doanh nghiệp ở sáu doanh nghiệp; pilot 4 câu khác. Còn 12 câu reserve trong ZIP nhưng quy trình trên **không tự chạy** chúng. Bộ nhỏ dùng chẩn đoán hiệu năng, không đại diện điểm toàn cuộc thi.
- `768` là output tối đa mỗi lượt; context preset cũ là 16K. Phần prompt tĩnh đã đếm bằng tokenizer pin dài khoảng 7.365–10.800 token, chưa catalog/history. Mỗi saved run đo request đầy đủ, cộng 8K dự phòng history, chọn 16–32K; nếu không vừa thì dừng, không cắt câu/rulebook hoặc bỏ câu dài.
- Qwen9B FP16 có payload khoảng 17,98 GiB toàn model; trên hai T4, vLLM TP2 thử chia weights giữa hai card. Profile đầu: 2 loop/2 seqs, prefill 2048, eager, VRAM fraction 0.88, không RAG. Đây là cấu hình thử, chưa được kiểm GPU thật tại máy phát triển; pilot mới quyết định có chạy được hay không.
- Baseline là source cũ; bộ đo ép dtype FP16 tại loader và ghi override, vì kiểm BF16 của Torch mới có thể tính emulation. Giữ device_map=auto, SDPA, inference_mode và KV cache vốn có; không thay agent loop. vLLM cũng dùng Torch bên dưới.
- Tắt prefix caching, đổi precision, đổi context hay số loop cần đo thành lượt khác với cấu hình được ghi lại; không đổi riêng một bên rồi gọi là speedup thuần engine. Nếu chênh lệch nhỏ, đo lại cả hai trên cùng tập trước khi kết luận.

Nếu pilot lỗi: đọc đúng *_pilot_run.log. CUDA OOM thì kiểm hai card/tiến trình cũ, context và số seqs; lỗi context thì xem context-preflight trong details, không tự xóa history; lỗi CUDA/kernel/NCCL thì sửa stack/hardware. Không tiêu tiếp lượt main khi pilot chưa đạt. Các thao tác nâng cấu hình ngoài profile đầu xem phần 5 của docs/vllm-runtime-addendum.md.

## 9. Phạm vi đã kiểm chứng

Đã kiểm bằng CPU/dữ liệu giả: dataset cố định và không nhãn; observer chạy được source baseline cũ; ghi cache metadata; các điều kiện so sánh. Bổ sung kiểm artifact/status/log, tải model vào thư mục tạm và thứ tự tải/cài trước phép đo cho luồng tự động. Chưa chạy checkpoint thật trên T4 hay kiểm thao tác Save Version trong tài khoản Kaggle; không khẳng định vLLM đã nhanh hơn.

Quy trình lưu version và tái dùng Output làm Input dựa trên [tài liệu Kaggle](https://www.kaggle.com/docs/notebooks). Các tên file result/status/log trong hướng dẫn là hợp đồng của code đi kèm, không phụ thuộc tên mà Kaggle tự đặt cho một gói tải xuống.

## 10. Nếu gặp lỗi tạo môi trường

Kit mới tự tạo venv bằng `--without-pip`, kiểm isolation và cài pip bằng script chính thức PyPA khi cần; không gọi ensurepip. Nếu log vẫn in `create_python_environment → ensurepip`, notebook đang dùng kit cũ: cập nhật Dataset kit ở bước 1, chọn phiên bản mới ở cả hai notebook rồi chạy lại. Không cần thêm cell vá hay mở notebook prepare_model.

Lỗi tải/cài xem `torch_pilot_run.log` hoặc `vllm_pilot_run.log`, theo notebook đang chạy. Lỗi trước khi khởi động công cụ xem `bootstrap_error.log`. Không chuyển sang main khi pilot chưa đạt.
