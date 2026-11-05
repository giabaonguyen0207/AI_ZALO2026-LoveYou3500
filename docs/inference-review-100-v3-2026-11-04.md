# Phân tích inference lần 3 trên 100 câu train ngẫu nhiên

Bundle người dùng cung cấp ngày 04/11/2026 nằm tại `outputs/diagnosis-2026-11-03/gemma4-12b-stress100-baseline-v3/`. Chấm lại `submission.jsonl` bằng nhãn train và bộ chấm local cho **49 đúng, 13 sai, 38 bỏ qua**, tương ứng **PA 36%**. Trong 62 câu có đáp án, 49 câu đúng, tương đương **79,03%**.

## Mẫu và khả năng so sánh

- Đủ 100 ID duy nhất, khớp cả nội dung câu hỏi, `as_of_date`, nhãn train và thứ tự trong danh sách đã bốc trước bằng seed `20261103`. Mẫu lấy đều từ 3.000 câu train và rơi vào 25 doanh nghiệp. Dòng `limitations` của summary còn mô tả cách lấy mẫu có tiêu chí của lượt cũ. Manifest và danh sách ID xác nhận lượt này lấy mẫu ngẫu nhiên.
- Submission, review và 100 log JSON khớp nhau. ZIP kiểm tra thành công và chứa cả 100 log Markdown. Hash của 351 đầu ra thô, identity hash, fingerprint train và hash của 34 môđun nguồn đều khớp.
- Lượt 2 và lượt 3 có cùng manifest checkpoint Gemma 4 12B, cùng BF16 trên RTX PRO 6000 Blackwell, cùng phiên bản runtime ghi nhận và cùng tham số sinh. Khác biệt giữa hash mã nguồn của hai lượt nằm ở phần ghi log. Đối chiếu bằng cách bỏ các dòng ghi log được bổ sung khỏi bản đọc trong bộ nhớ cho lại đúng hash lượt 2 của `agent.runner` và `application`.
- Hai mẫu chỉ giao một câu `q_27eebcccdf1f`, cả hai lần đều đúng `-9.65` và dùng hai lượt gọi model. Lượt 3 có 54 câu `number` và 24 câu `percent`, trong khi lượt 2 có 10 câu mỗi loại. Chênh lệch PA phản ánh hai phép đo trên hai mẫu khác nhau.
- Trong 18 câu thuộc các doanh nghiệp từng dành riêng để validation, kết quả là 7 đúng, 3 sai và 8 bỏ qua. 82 câu còn lại có 42 đúng, 10 sai và 30 bỏ qua. Đây là thống kê theo mẫu đã bốc, với số câu rất khác nhau giữa hai nhóm.

## Điểm và thời gian

| Chỉ số | Lượt 2, mẫu 72 câu | Lượt 3, mẫu 100 câu |
|---|---:|---:|
| Đúng / sai / bỏ qua | 56 / 1 / 15 | **49 / 13 / 38** |
| PA | 76,39% | **36,00%** |
| Tỷ lệ có đáp án | 79,17% | 62,00% |
| Lượt gọi model mỗi câu | 2,75 | 3,51 |
| Thời gian riêng batch | 18 phút 15 giây | 30 phút 26 giây |
| Trung vị mỗi câu | 10,87 giây | 12,33 giây |
| P95 mỗi câu | 32,65 giây | 46,87 giây |

| Type theo nhãn train | Đúng | Sai | Bỏ qua |
|---|---:|---:|---:|
| `number` | 25 | 11 | 18 |
| `percent` | 14 | 1 | 9 |
| `list` | 3 | 0 | 7 |
| `date` | 5 | 1 | 2 |
| `boolean` | 1 | 0 | 0 |
| `unanswerable` | 1 | 0 | 2 |

Trong 62 đáp án đã chốt, 49 lấy kết quả từ metric, gồm 44 đúng và 5 sai. Nhánh SQL tự do có 12 đáp án, gồm 4 đúng và 8 sai. Đáp án còn lại là một `unanswerable` đúng. Hai nhánh nhận các câu hỏi khác nhau nên tỷ lệ này hữu ích để khoanh vùng kiểm tra, với độ khó chưa được cân bằng.

38 câu bỏ qua dùng 179/351 lượt gọi model và 1.038 giây, khoảng **57% thời gian batch**. Có 19 câu dừng vì lặp action hoặc kết quả và 19 câu model chủ động `abstain`. Trong 351 lượt sinh, 349 kết thúc bằng EOS, hai lượt chạm trần 768 token, cả hai thuộc câu bỏ qua. Đầu vào dài nhất là 15.197 token. Lỗi quan sát chủ yếu nằm ở lựa chọn nghiệp vụ, truy vấn và vòng sửa lỗi. Thời gian batch được ghi riêng sau khi nạp model.

## Những lỗi lặp lại

**Đơn hủy.** Trong tám câu về số đơn hoặc tỷ lệ đơn hủy, hai đúng, ba sai và ba bỏ qua. Ba câu sai cùng truy vấn `status='cancelled'` trên `dm_orders`, bảng đã lọc bỏ đơn hủy theo chính sách bán hàng. Chạy trên `dm_orders_all` với cùng kỳ cho đúng 19, 41 và 37. Hai câu tỷ lệ đã dùng đúng mẫu số gồm mọi trạng thái, nhưng từ khóa của bộ kiểm tra không nhận “hủy đơn” hoặc “tỉ lệ huỷ đơn” nên chặn kết quả 2,18% và 1,14%. Ca còn lại có dữ liệu ngày hoặc trạng thái chưa giải được và SQL trả 1,73%, khác nhãn 1,62%. Ca đó cần điều tra nguồn dữ liệu trước khi chấp nhận SQL. [Bộ kiểm tra bảng truy vấn](../src/datamate/agent/tools.py).

**Kỳ thời gian.** Cả năm câu “từ đầu năm” bị bỏ qua. Bốn câu đã thử kỳ từ 01/01 đến `as_of_date` và bị bộ kiểm tra ép về riêng “hôm nay”. Câu thứ năm đi tìm sản phẩm `heets` từ chữ gõ nhầm “heets hnay”. Gọi trực tiếp công cụ theo đúng khoảng năm cho kết quả khớp cả năm nhãn. Cùng nhóm, một câu tăng trưởng lấy tháng 5 đến ngày 29 so với toàn tháng 4 và chốt `-0,50%`. Cùng số ngày đến 29/04 cho `+2,83%`. [Bộ giải kỳ](../src/datamate/agent/verification.py).

**Phân loại đại lượng và điều kiện lọc.** Bảy câu hỏi tỷ trọng doanh thu theo nhóm hàng, nhóm khách hoặc giới tính đều bỏ qua, trong đó một câu có nhãn `unanswerable`. Nhiều câu bị gán sang `revenue_growth` từ tín hiệu “doanh thu” và “%”. Hướng ngược lại cũng có lỗi. Với một số câu không nhận diện được đại lượng, `revenue` chạy thành công và chốt tổng tiền cho câu hỏi số lượng hoặc lãi gộp. SQL hợp lệ và cờ `verified` của công cụ xác nhận phép tính đã chạy, còn ý nghĩa của phép tính phụ thuộc action đã chọn.

**Ca bán và loại dữ liệu.** Hai câu doanh thu ca chiều và một câu đếm đơn ca sáng đều so sánh timestamp ISO đầy đủ với chuỗi chỉ có giờ. Hai câu đầu tính nhầm tổng, câu cuối chốt 0. Tách phần giờ và dùng ranh giới ca từ rulebook train cho đúng 175.175.700, 139.965.000 và 590.

**Ánh xạ schema của `biz_123`.** Năm câu của doanh nghiệp này có hai sai và ba bỏ qua. SQLite gốc có `don_hang`, `chi_tiet_don`, `khach_hang` và `thanh_toan`, README giải thích các trường `TriGia`, `SLQuyDoi` và `SoTienThu`. Trong catalog hiện tại, các bảng này bị gán `other`; bảng đơn, khách và sổ thu thiếu ánh xạ cần thiết. Các khả năng `sales`, `order_identity`, `customer_identity` và `payments` đều báo `False`. Với câu xếp ba khách mua cao nhất, log chặn `limit=3` vì bộ nhận diện suy ra top 1. Gọi trực tiếp metric với `limit=3` vẫn báo thiếu `customer_identity`, cho thấy cần xử lý cả ánh xạ nguồn và nhận diện top k. [README của doanh nghiệp](../train/train/data/biz_123/README.md).

## Sáu câu bỏ qua có kết quả SQL khớp nhãn

Chiếu kết quả đã ghi trong log theo type tham chiếu, chọn cột phù hợp và làm tròn phần trăm cho thấy sáu trường hợp dưới đây. Đây là đối chiếu sau chạy để định vị lỗi, dùng type tham chiếu chỉ trong bước phân tích.

| Câu | Giá trị từ SQL trong log | Điểm bị mất |
|---|---|---|
| `q_72f43cc7a0bc` | `2025-04-04` | “ngayf nào” làm nhận diện sai `best_day` thành `revenue`, sau đó SQL bị chặn ở final |
| `q_46d72d387508` | `2,18%` | Bộ kiểm tra từ chối `dm_orders_all` cho câu hỏi tỷ lệ hủy |
| `q_5348170a927c` | `1,14%` | Cùng lỗi kiểm tra `dm_orders_all` |
| `q_a7563e07926b` | `[106656000, 420904000]` | Model chọn `number` cho hai dòng kết quả; cần `list` theo thứ tự hình thức thanh toán trong câu |
| `q_7c0a4b5f5196` | `14` | Hỏi đếm ngày đạt ngưỡng nhưng contract chọn doanh thu tổng; model lặp SQL thay vì chốt |
| `q_3d7beff53344` | `893622300` | SQL tính từ đầu năm đúng kỳ nhưng bị cổng final chặn |

## Đối chiếu toàn bộ 13 câu sai

| Câu | Nộp | Nhãn | Quan sát từ log và kiểm chứng |
|---|---:|---:|---|
| `q_3234281bf31d` | 0 | 19 | Đếm đơn hủy trên bảng đã lọc hủy |
| `q_6d57349294ec` | 0 | 41 | Cùng lỗi nguồn bảng |
| `q_255991af3110` | 0 | 37 | Cùng lỗi nguồn bảng |
| `q_ece3c46e24d3` | −0,50% | +2,83% | So tháng đang chạy với cả tháng trước |
| `q_b5c0715af111` | 118 | 1.668 | Chọn một sản phẩm thay cho tất cả món tên kết thúc bằng `(L)` |
| `q_4bbd72655e8e` | 15.470.000 | 182 | Cộng doanh thu thay vì số đĩa |
| `q_b3e5c2fb4852` | 134.177.600 | 129.523.500 | Dò chữ “kê đơn” trong tên thay vì cột cờ của danh mục thuốc |
| `q_2674d035b36f` | 0 | 590 | So timestamp đầy đủ với giờ ca |
| `q_81cc91f86744` | 271.124.500 | 190.133.805 | Chọn doanh thu cho câu lãi gộp. Cả 6.130 dòng hợp lệ trong kỳ đều có chi phí, số lượng và thành tiền để tính theo rulebook |
| `q_43bd6fa219e8` | 0 | 24 | Truy vấn công nợ trên lớp đơn hàng rỗng do thiếu ánh xạ `biz_123` |
| `q_250cf7d222c0` | 2025-06-24 | 2029-05-12 | Truy vấn chọn hạn nhỏ hơn hoặc bằng ngày hỏi, bỏ điều kiện lô còn tồn chưa quá hạn |
| `q_b0ff97ca1db9` | 32.659.000 | 771 | Cộng tiền cho câu đếm ly |
| `q_ddd87e1d24ed` | `list: []` | `number: 5` | Dùng `product_id` rỗng và `NOT IN`, sau đó đổi type sang list. Cả 340 dòng bán trong kỳ thiếu `product_id` nhưng có 55 `product_key` khác nhau. Phép đếm bằng `product_key` và `NOT EXISTS` cho đúng 5 |

20 phép tính kiểm chứng trên CPU khớp nhãn. Các phép tính dùng phạm vi đơn hủy đúng, kỳ từ đầu năm, cùng số ngày tháng trước, tách giờ ca, chọn đúng metric, tổng số ly size L, lãi gộp, cờ thuốc kê đơn hoặc khóa mặt hàng phù hợp. Một phép gọi bổ sung cho xếp hạng khách tại `biz_123` xác nhận lỗi ánh xạ ở trên. Đây là các lời gọi được chỉnh để chẩn đoán, tách riêng khỏi điểm inference đã ghi.

## Thứ tự sửa theo bằng chứng

1. Kiểm tra **đại lượng, phạm vi và điều kiện lọc** trước khi chốt. Sửa cặp trường hợp đối xứng đã thấy trong log, gồm câu hợp lệ bị ép sang sai metric và phép tính sai đại lượng được chấp nhận. Dùng tình huống “từ đầu năm”, cùng kỳ tháng trước, số đơn hủy, số lượng theo đơn vị, tỷ trọng và đầu ra nhiều thành phần để kiểm thử.
2. Hoàn thiện ánh xạ schema và trường nghiệp vụ. Kiểm chứng trên nguồn SQLite có `MaHoaDon`, `MaKhachHang`, `TriGia` và `SoTienThu`, rồi dùng cột phân loại thực tế cho nhóm hàng, thuốc kê đơn, ca bán và công nợ. Kết quả rỗng do nguồn chưa ánh xạ cần được phân biệt với số 0 thực tế.
3. Cho đường SQL hợp lệ cơ hội chốt bằng kiểm tra ngữ nghĩa phù hợp với câu hỏi. Chỉ dẫn sửa lỗi cần nêu đúng thao tác, chẳng hạn dùng `list` cho hai kết quả hoặc dùng bảng mọi trạng thái cho tỷ lệ hủy. Hai lượt sinh chạm trần token và 19 ca dừng lặp cho thấy cần tập trung vào chất lượng phản hồi sửa lỗi.

Với trajectory, có 49 log đúng theo nhãn và 39 trong số đó không ghi lỗi ở các bước công cụ. Đây là ứng viên cần rà ngữ nghĩa và tách theo doanh nghiệp. Nếu dùng bộ 100 câu làm mốc so sánh trước và sau fine-tune, giữ các câu này và biến thể của chúng ngoài tập học. Kết quả hiện tại xác định được một số lỗi ở lớp thực thi và kiểm tra mà fine-tune model không trực tiếp sửa được.
