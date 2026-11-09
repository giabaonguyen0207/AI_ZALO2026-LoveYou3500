# Rà soát Vietnamese_Embedding và Vietnamese_Reranker

Cập nhật bố cục: Qwen 4B đã ngừng hỗ trợ trong runtime. Dòng kiểm kê 4B bên dưới được giữ như kết quả rà soát lịch sử; cấu hình hiện hành xem [model-comparison](model-comparison.md).

Ngày kiểm tra: **09/11/2026**. Nguồn mới: repository công khai của AITeamVN trên Hugging Face, metadata API, lịch sử revision, cấu hình và header safetensors. Đây là đánh giá model ứng viên theo yêu cầu người dùng, chưa phải quyết định triển khai RAG của đội.

**Kết luận:** hai model phù hợp để thử retrieval tiếng Việt và reranking trong DataMate. Có checkpoint trọng số từ trước 01/10/2026, khai báo Apache 2.0, và tổng tham số cho phép ghép với Qwen 9B hoặc Gemma 12B hiện có. Chưa có bằng chứng chúng tăng PA hoặc giảm tổng thời gian của DataMate. Các điều kiện cuộc thi bên dưới đối chiếu với bản tổng hợp của người dùng, chưa phải xác nhận mới từ ban tổ chức.

## Checkpoint và nguồn đã kiểm tra

| Thành phần | Số phần tử tensor | Lần upload trọng số đã xác minh (UTC) | Giấy phép công bố |
|---|---:|---|---|
| AITeamVN/Vietnamese_Embedding | 567.754.752 | 17/03/2025 06:59:35 | Apache 2.0 |
| AITeamVN/Vietnamese_Reranker | 567.755.777 | 30/04/2025 07:46:11 | Apache 2.0 |
| Tổng hai model | **1.135.510.529** | | |

Revision đề xuất khóa khi chuẩn bị artifact:

- Embedding: **dea33aa1ab339f38d66ae0a40e6c40e0a9249568**, commit ngày 25/08/2025 13:32:21 UTC. SHA256 LFS của model.safetensors: **1f2debafdf03659e8273022a3e902b94deec73cd20c2b7262ab7e21630163f6d**.
- Reranker: **f536976248403314225d7fdfdbc87f0e9516a54e**, commit ngày 06/05/2025 07:15:35 UTC. SHA256 LFS của model.safetensors: **1cecccab406ec6d3f13da8ccdc5e853a1372e51845322163655ab530d3730071**.

Đã đối chiếu SHA256 LFS của trọng số tại hai revision này với các commit upload gốc tương ứng **d4a60ad49d14ebb0da322b2dff0a46ed04cb7cb2** và **37eada2458a2c3b4ff304b5c73ef53773489728f**: khớp nhau. Vì vậy bằng chứng thời hạn dựa trên checkpoint trọng số cụ thể, không chỉ năm ghi trên model card. Metadata cho thấy cả hai repository public và không gated tại thời điểm kiểm tra. Giấy phép Apache 2.0 được công bố trong metadata/model card; khi đóng gói cần giữ thông tin giấy phép và attribution thích hợp.

Đã đọc HTTP Range 0–262143 của mỗi file, giải mã header và cộng tích shape của mọi tensor. Kết quả khớp số tham số API. **Chưa tải toàn bộ trọng số, chưa kiểm SHA256 toàn bộ payload local, chưa load model/tokenizer hoặc inference CPU/GPU.** Số đếm này là inventory theo checkpoint/header, chưa phải đo bộ nhớ runtime.

Bằng chứng máy đọc được: [model-evidence-rag-2026-11-09.json](model-evidence-rag-2026-11-09.json). Raw metadata và header được lưu tại outputs/research/rag-models-2026-11-09/; SHA256 các file raw nằm trong evidence JSON.

Nguồn chính: [embedding model card](https://huggingface.co/AITeamVN/Vietnamese_Embedding), [embedding history](https://huggingface.co/AITeamVN/Vietnamese_Embedding/commits/main), [embedding weight upload](https://huggingface.co/AITeamVN/Vietnamese_Embedding/commit/d4a60ad49d14ebb0da322b2dff0a46ed04cb7cb2), [reranker model card](https://huggingface.co/AITeamVN/Vietnamese_Reranker), [reranker history](https://huggingface.co/AITeamVN/Vietnamese_Reranker/commits/main), [reranker weight upload](https://huggingface.co/AITeamVN/Vietnamese_Reranker/commit/37eada2458a2c3b4ff304b5c73ef53773489728f). URL API tại revision cụ thể nằm trong evidence JSON.

## Kiểm kê khi ghép với model dự án

Số tham số generator lấy từ registry hiện có của dự án, không xác minh lại các generator trong lượt kiểm tra này. B là một tỷ tham số, không phải dung lượng file.

| Generator + cả hai model RAG | Tổng tham số chính xác | Tổng theo B | Trong 15B |
|---|---:|---:|---|
| Qwen 3.5 4B | 5.795.375.617 | 5,795 | Có |
| Qwen 3.5 9B | 10.788.614.897 | 10,789 | Có |
| Gemma 4 12B | 13.095.240.753 | 13,095 | Có |
| Gemma 4 12B + MTP assistant | 13.518.097.717 | 13,518 | Có |
| Pinkstackorg upscaled 14B V2 | 15.734.995.569 | 15,735 | **Không** |

Self-review bằng cùng checkpoint không cộng thêm một model. Quantization, lần lượt load/unload hoặc chạy model ở process khác không giảm inventory cần khai báo. Bất kỳ model bổ sung nào cũng phải được cộng lại.

## Chi tiết kỹ thuật cần giữ đúng

- Embedding là bản fine-tune BGE-M3 cho tiếng Việt; model card nêu vector 1024 chiều và train ở 2048 token. Cấu hình pooling thực tế dùng CLS và module Normalize. Nếu triển khai trực tiếp bằng Transformers, cần tái tạo đúng pooling/normalization; không tự thay bằng mean pooling. Dot product với vector đã chuẩn hóa tương đương cosine.
- Có khác biệt đáng chú ý: sentence_bert_config.json đặt max_seq_length=8192, trong khi model card và ví dụ sử dụng nêu 2048. Không suy ra chất lượng retrieval 8192 token đã được chứng minh. Bắt đầu với đoạn nằm trong 2048 token theo tokenizer của embedding là đề xuất thận trọng, rồi đo trên dữ liệu đội.
- Reranker có kiến trúc XLMRobertaForSequenceClassification và một nhãn đầu ra. Dùng cặp query–passage để lấy một logit liên quan, qua AutoModelForSequenceClassification hoặc CrossEncoder. Phần hướng dẫn tự động trên trang và một số mục trong model card gọi nó là embedding/vector 1024 chiều có thể gây nhầm; cấu hình checkpoint và ví dụ Usage của tác giả cho thấy đây là mô hình chấm cặp.
- Model card reranker nêu train với 2304 token, phân bổ 256 cho query và 2048 cho passage. Khi tokenize cặp phải tính cả special token trong giới hạn tổng, tránh cắt mất điều kiện quan trọng. Điểm liên quan không phải xác suất đáp án nghiệp vụ đúng.
- Các kiến trúc XLM-R này dùng được theo API Transformers chuẩn về mặt cấu hình; chưa kiểm chứng khả năng load và chạy trên dependency/GPU hiện tại. Không có yêu cầu phải chuyển chúng sang cùng engine vLLM của generator. Cần khóa và kiểm tra bộ dependency dùng chung trước khi merge, không hạ phiên bản Transformers của generator chỉ theo phiên bản ghi trong cấu hình model cũ.

Nguồn cấu hình: [embedding upload/config/pooling/normalize](https://huggingface.co/AITeamVN/Vietnamese_Embedding/commit/d4a60ad49d14ebb0da322b2dff0a46ed04cb7cb2), [reranker config tại revision khóa](https://huggingface.co/AITeamVN/Vietnamese_Reranker/blob/f536976248403314225d7fdfdbc87f0e9516a54e/config.json), [cách dùng cross-encoder của model gốc](https://huggingface.co/BAAI/bge-reranker-v2-m3). Cấu hình JSON đã đọc trực tiếp được giữ trong evidence.

## Phù hợp với DataMate ở đâu

Đã đọc rulebook train và README thực tế của biz_101 trước khi đề xuất. Dữ liệu có phép tổng hợp chính xác, khóa khách/sản phẩm, đơn vị và điều kiện trạng thái. Hai model có thể hỗ trợ tìm mô tả cột/bảng, quy ước trong README, tên thực thể và đoạn tài liệu liên quan. Đây là suy luận về vai trò phù hợp, chưa phải kết quả benchmark.

Đề xuất để retrieval hỗ trợ chọn nguồn và hiểu nghĩa, còn SQL/công cụ hiện tại tính trên đầy đủ các dòng thỏa điều kiện. Lấy top-k giao dịch tương tự rồi cộng tiền sẽ không bảo đảm tổng đúng. Với tên/mã/size cần giữ tra cứu chính xác và đối chiếu khóa trước, semantic retrieval chỉ bổ sung ứng viên.

Giữ rulebook đầy đủ và cơ chế giải thời gian theo as_of_date hiện có trong lần thử đầu. Không để một đoạn được rerank cao thay thế quyền ưu tiên của rulebook split. Index phải tách phạm vi split/doanh nghiệp, có nguồn bảng/cột/tài liệu, và không chứa đáp án public/private test. Ví dụ giải từ train nếu đưa vào retrieval cần tách khỏi câu/doanh nghiệp dùng đánh giá để tránh benchmark bị rò rỉ.

Model card báo benchmark trên Legal Zalo 2021; kết quả đó không chứng minh PA, xử lý số liệu hay tốc độ của DataMate. RAG cũng không tự sửa lỗi projection/type hoặc lỗi tính toán của công cụ.

## Điểm ghép code và cách kiểm chứng tiếp theo

Đây là đề xuất giao diện, chưa được triển khai:

- Để generator tiếp tục đi qua ChatBackend; tách interface embedding/retriever và reranker với đầu ra vector hoặc danh sách đoạn có score/provenance. Không ép hai encoder vào giao thức sinh action JSON của ChatBackend.
- Mở rộng inventory và artifact staging/verification: inventory() hiện chỉ nhận main_key và assistant_key, chưa cộng embedding/reranker. Chỉ kiểm kê generator sẽ báo thiếu hơn 1,135B tham số.
- Chuẩn bị checkpoint/tokenizer/cấu hình/module offline từ revision cố định; kiểm hash toàn bộ artifact local khi tải thật. Index và cache ghi ngoài /data. Hash model, corpus, cách chunk và cấu hình retrieval phải tham gia run/resume identity.
- So sánh cùng generator/backend/prompt nền trên cùng tập train đánh giá đã khóa: baseline; thêm embedding; thêm reranker. Đo PA, đúng/sai/abstain, nguồn liên quan bị bỏ sót, latency, VRAM và tổng thời gian có load/index. Đóng gói thử chạy với mạng tắt trước khi kết luận đáp ứng yêu cầu offline/thời hạn.

Chưa thay đổi runtime, registry, dependency, checkpoint, notebook hay các ZIP đã khóa. Chưa đo PA, tốc độ, VRAM hoặc thời gian toàn bộ split với hai model này.
