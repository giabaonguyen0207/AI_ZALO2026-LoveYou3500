# Kiểm tra đầu vào Colab · 02/11/2026

## Bổ sung: import trong phiên notebook đang chạy

Người dùng tiếp tục báo `ModuleNotFoundError: No module named 'datamate'` ngay đầu ô load model. Chưa nhận log ô cài dependency nên chưa khẳng định người dùng đã cài thành công hay đã chạy đủ ô. Notebook trước dùng `pip install -e` trong subprocess và không thêm `WORK/src` vào phiên Python hiện tại; đường dẫn editable `.pth` thường được xử lý khi Python khởi động ([tài liệu Python](https://docs.python.org/3/library/site.html)).

Cả hai notebook được sửa để cài từ đúng `cwd=WORK`, thêm `WORK/src` vào `sys.path`, xóa cache tìm module và in `DataMate import OK` sau khi cài. Hướng dẫn F11 có ô sửa trực tiếp cho notebook đang mở, không cần nén/upload lại dữ liệu. Kiểm tra hồi quy chạy ô thiết lập trong tiến trình Python cách ly, ban đầu không import được `datamate`, rồi import mã nguồn thật ngay trong cùng tiến trình. Các bước pip/CUDA được giả lập trong kiểm tra này; không tuyên bố đã chạy dependency/model trên Colab.

## Kiểm tra ZIP

Người dùng báo lỗi khóa CSV khi dùng Windows PowerShell `Compress-Archive`, sau đó báo assertion `Upload only the original TRAIN package` ở ô đầu của notebook inference. Chưa truy cập runtime hoặc ZIP trên Drive của người dùng; assertion cũ không cho biết tìm được 0 hay nhiều thư mục train.

Đã đọc `outputs/train.zip` trên E: vào ngày 02/11/2026: 162 file, một split `train/`, CRC hợp lệ, đủ toàn bộ file và từng SHA256 khớp `configs/training_sources.json`. Fingerprint nguồn: `5d598c5d603020c8077df7802cc6fb06b364af9914eb88425bd7664c639978ea`. ZIP SHA256: `b30c103b550cb9a6c0d5720f1056b92618be31e04841dd07be8b68be8fcf9a8f`. Không mặc định bản trên Drive giống bản local.

Đã thêm `tools/package_train.py`: dùng Python chuẩn, kiểm tra nguồn, tạo ZIP tạm, chỉ thay ZIP lỗi sau khi kiểm tra hash; dùng lại ZIP hợp lệ. Cập nhật cả notebook inference và LoRA để kiểm tra layout, file và hash ngay từ ZIP, chuẩn hóa đường dẫn Windows, giải nén vào thư mục mới để không quét lẫn các lần trước. Model/cache/kết quả trên Drive không bị xóa.

Kiểm tra tự động bao gồm ZIP ở các mức thư mục khác nhau, chạy lại khi còn thư mục cũ, dấu `\`, thiếu nhãn hoặc CSV, nhiều split, nhầm source ZIP, đường dẫn ra ngoài, tạo lại ZIP lỗi và dùng lại ZIP hợp lệ khi một CSV đang có tiến trình đọc. Đây là kiểm tra đóng gói/đọc dữ liệu trên máy local; chưa phải chạy model trên Colab, benchmark hoặc fine-tune.
