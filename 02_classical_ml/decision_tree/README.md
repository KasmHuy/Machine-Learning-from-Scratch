# decision_tree

## Concept
1. Trong một cây quyết định, có hai loại nút chính:
    - Nút quyết định (Decision Node): Nút dùng để đặt câu hỏi (chia nhánh). Nó sẽ có các thông tin về tính chất dữ liệu để rẽ nhánh sang Trái hoặc Phải.
    - Nút lá (Leaf Node): Nút ở cuối cùng của nhánh, không chia thêm nữa. Nó chứa kết quả dự đoán cuối cùng.
2. Trong DecisionTree, bao gồm thuộc tính cơ bản là độ sâu tối đa và số mẫu tối thiểu:
    - Max-depth: Việc không giới hạn cây sẽ tiếp tục chia nhánh mãi cho đến khi mọi điểm dữ liệu bị tách riêng rẽ. Điều này dẫn đến hiện tượng Overfitting (Học vẹt) — cây chạy cực tốt trên dữ liệu học nhưng cực tệ trên dữ liệu thực tế mới.
    - Min Samples: Đây là điều kiện về số lượng dữ liệu tối thiểu phải có tại một Nút thì nút đó mới được phép tiếp tục chẻ đôi (chia nhánh).
3. Áp dụng chia để trị tách thành các bộ dữ liệu con left và right và tìm kiếm nhị phân để lược bỏ bớt các dữ liệu dư thừa trong quá trình dựng cây.
4. Áp dụng đệ quy để dựng cây và xây dựng từng Node.

## Key learnings
- **Bản chất của thuật toán là sự đơn giản hóa:** Mọi tập dữ liệu phức tạp (nhiều hàng, nhiều cột, nhiều nhãn khác nhau) rốt cuộc đều được cây xử lý bằng cách đặt ra các câu hỏi hệ nhị phân (Đúng/Sai) để chẻ đôi dữ liệu.
- **Phân biệt hai trạng thái của Left/Right:** - *Lúc xây cây (Fit):* `left` và `right` là các mảng dữ liệu được lọc bằng mặt nạ logic (`True/False`).
    - *Lúc cây hoàn thành (Node):* `left` và `right` là các sợi dây liên kết (con trỏ) trỏ từ nút cha xuống nút con để dẫn đường cho dữ liệu mới.
- **Tốc độ nhờ luật loại trừ:** Việc bóc tách dữ liệu liên tục thành hai nửa giúp mô hình có tốc độ tìm kiếm câu trả lời nhanh như chớp (tương tự thuật toán tìm kiếm nhị phân). Mỗi câu hỏi được trả lời đồng nghĩa với việc loại bỏ được một nửa lượng dữ liệu không liên quan.
- **Tầm quan trọng của tiêu chí đo lường (Gini/Entropy):** Hiểu rằng toán học (Gini hay Entropy) đóng vai trò là "thước đo độ sạch". Mô hình dựa vào đó để chấm điểm xem câu hỏi nào (cặp `feat` và `thres` nào) mang lại lượng thông tin lớn nhất (Information Gain) nhằm ưu tiên cắt trước.
- **Tư duy kỹ sư AI > Cú pháp code:** Cú pháp lập trình (NumPy, vòng lặp, đệ quy) chỉ là công cụ và hoàn toàn có thể được thay thế bằng các thư viện ăn liền như Scikit-Learn. Việc thấu hiểu tường tận bản chất logic vận hành bên dưới mới là giá trị cốt lõi giúp cấu hình và tinh chỉnh mô hình chính xác (tránh học vẹt - overfitting).