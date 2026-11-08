# Thử Pinkstackorg Qwen 3.5 upscaled 14B V2 trên Colab

Đã thêm lựa chọn inference `qwen35-upscaled14b-v2` ngày 06/11/2026. Key `qwen35-9b` tiếp tục trỏ tới **Qwen/Qwen3.5-9B chính thức**; đổi sang key đó không chọn model Pinkstackorg.

**Kết quả chạy thật trước sửa, nhận ngày 06/11/2026:** 10 câu đều abstain do lỗi EOS/parser, trung bình 49,06 giây/câu. Xem [chẩn đoán đầy đủ](inference-review-qwen14b-2026-11-06.md). Sau đó đã triển khai sửa EOS/parser/plan cho cả Qwen 9B và 14B; dùng [ZIP/notebook và hướng dẫn mới](qwen-protocol-fix-2026-11-06.md). Chưa có benchmark checkpoint đầy đủ trên GPU sau sửa.

## Cấu hình đã kiểm tra

| Mục | Giá trị |
|---|---|
| Repository | `Pinkstackorg/Qwen-3.5-upscaled-14B-V2-noft` |
| Key của DataMate | `qwen35-upscaled14b-v2` |
| Revision cố định | `f4fe3d2ad8f2e14799624cdbafa5d46219bc1fb2` |
| Ngày commit revision | 24/03/2026, 11:13:01 UTC |
| Commit upload trọng số | `43418eac1081e1d44dbf1b94fce556b4976fc0a3`, 24/03/2026, 11:10:54 UTC |
| Tổng tensor parameters | **14.599.485.040**, gồm **456.010.480** tham số vision |
| Khoảng cách đến giới hạn 15B | 400.514.960 tham số; preset chỉ dùng một model |
| License repository công bố | Apache 2.0 |
| Trọng số | BF16, một file **29.199.107.448 bytes**, khoảng **27,2 GiB** |
| Loader | `AutoModelForImageTextToText` → `Qwen3_5ForConditionalGeneration` |

Đã đối chiếu LFS SHA256 của file trọng số giữa commit upload và revision được pin, kết quả giống nhau. Đã đọc tối đa 256 KiB đầu safetensors, đếm toàn bộ 1.078 tensor trong header và khớp số tham số do Hub báo. Không suy số tham số từ tên “14B” hoặc con số “15B” được làm tròn trên trang model.

Metadata và bằng chứng hash: [model-evidence-pinkstack-2026-11-06.json](model-evidence-pinkstack-2026-11-06.json). Nguồn: [model card](https://huggingface.co/Pinkstackorg/Qwen-3.5-upscaled-14B-V2-noft), [revision đã pin](https://huggingface.co/Pinkstackorg/Qwen-3.5-upscaled-14B-V2-noft/tree/f4fe3d2ad8f2e14799624cdbafa5d46219bc1fb2), [commit upload](https://huggingface.co/Pinkstackorg/Qwen-3.5-upscaled-14B-V2-noft/commit/43418eac1081e1d44dbf1b94fce556b4976fc0a3). Đây là kiểm chứng nguồn model theo ràng buộc đội đang dùng, không phải xác nhận chấp thuận của BTC.

## Chạy cùng cấu hình fast v5

1. Thay `datamate-colab-source.zip` trên Drive bằng gói mới tại `outputs/datamate-colab-source.zip`. Registry nằm trong source ZIP, không chỉ trong notebook.
2. Mở `notebooks/DataMate_Colab_Inference.ipynb`. Ở **ô cấu hình đầu tiên**, đặt:

```python
MODEL_KEY = "qwen35-upscaled14b-v2"
QUANTIZATION = "none"
```

3. Dùng runtime mới hoặc khởi động lại runtime rồi chạy lại từ ô đầu. `MODEL_SOURCE_DIR` được tính theo key ở đó. Đổi riêng key sau khi đã chạy setup có thể để đường dẫn vẫn trỏ vào thư mục Gemma/Qwen cũ và bị chặn vì manifest không khớp.
4. Ô load tự tải đúng revision vào thư mục Drive riêng `models/qwen35-upscaled14b-v2`, seal model, chép sang `/content/models`, kiểm tra hash và dùng tokenizer của chính checkpoint. Không cần tự đổi `repo_id` hoặc bỏ kiểm tra manifest.
5. Ô batch vẫn lấy cùng mẫu 100, seed 20261103. Có thể đặt `RUN_LIMIT=7` để kiểm tra GPU trước, sau đó `RUN_LIMIT=None` để resume đủ 100. Đặt `RUN_NAME_SUFFIX="random100-fast-v5-Nov6th"` nếu muốn tên lượt chạy thể hiện ngày mới. `RUN_NAME` tự thêm tiền tố model; không resume vào thư mục kết quả Gemma.

Giữ nguyên profile hiện tại để so sánh: context **16.384**, output **768**, **6 bước**, **90 giây/câu**, `schema_mode="code_first"`, `require_plan=True`, `auto_execute_plan=True`, schema repair tối đa 512 token/20 giây khi cần. Không thay prompt, metric, rulebook hay bộ câu hỏi trong lần thêm model này. Preset CLI tương ứng là `configs/qwen35-upscaled14b-v2.json`.

Với GPU khoảng 95 GiB như bundle Gemma vừa nhận, riêng trọng số BF16 khoảng 27,2 GiB có dư địa để thử cùng profile. Tổng bộ nhớ còn có cache/activation và phần khác; chưa đo VRAM thực tế của model này. Colab Pro có thể cấp GPU khác. Nếu đổi sang `4bit` để chạy trên GPU nhỏ, ghi rõ đây là thay đổi quantization khi so kết quả; quantization không đổi tổng tham số được tính theo giới hạn.

Đổi model giữ nguyên các lỗi agent vừa chẩn đoán ở fast v5. Mục tiêu lượt này là đo tác động của checkpoint trong cùng hệ thống. Không mặc định model upscaled nhiều tham số hơn sẽ nhanh hơn hoặc hiểu action tốt hơn; model card mô tả đây là bản merge/upscale và tác giả khuyến nghị fine-tune thêm.

## Phạm vi kiểm tra tích hợp ban đầu

- Chạy offline trên Transformers **5.15.0**, PyTorch **2.11.0+cu128**, trong giới hạn dependency hiện có của Colab.
- Khởi tạo model trên thiết bị **meta** để kiểm tra cấu trúc, không cấp phát trọng số đầy đủ: toàn bộ tên/shape tham số khớp header checkpoint, không thiếu hoặc thừa tensor. Đây không phải phép thử chạy model trên GPU.
- Tokenizer đúng revision, chat template nhận `enable_thinking=False` và tạo phần mở đầu assistant với khối think rỗng. Bảy prompt ban đầu thật của run fast v5 dài **7.924–11.055 token** với tokenizer này, đều vừa context 16K cộng 768 token đầu ra. Đây chưa bao gồm những lượt tool tiếp theo. Parser nhận action JSON có EOS Qwen trong fixture kiểm tra.
- **26 kiểm thử phần mềm đều qua** cho loader, model inventory, scoring contract và model cache; các ô code notebook parse được và ô loader vẫn khớp source helper. Kết quả chi tiết nằm trong `outputs/model-verification/qwen35-upscaled14b-v2-2026-11-06/`.
- **Chưa tải toàn bộ trọng số tại máy local, chưa generation, chưa đo tốc độ/VRAM/PA.** Lần tải trọng số đầy đủ sẽ xảy ra khi người dùng chạy ô chuẩn bị model trên Colab.

Notebook vẫn mặc định Gemma để giữ lựa chọn hiện tại của người dùng. Gói seed và source fine-tune Gemma riêng không được phát hành lại trong thay đổi này; tiếp tục dùng đúng cặp ZIP đã bàn giao cho SFT Gemma. Source inference mới có thêm registry entry nên không dùng nó thay cho source ZIP khóa theo hash của release SFT cũ.
