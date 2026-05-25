# linear_algebra

## Concept
- PCA áp dụng các phép biến đổi của đại số tuyến tính để giảm số chiều dữ liệu những vẫn giữ lượng lớn thông tin
- PCA đi tìm một góc nhìn (một trục tọa độ mới) sao cho khi ép dẹp dữ liệu lên trục đó, các điểm dữ liệu vẫn tản ra xa nhau nhất có thể (phương sai lớn nhất), chứ không bị chụm lại một cục.
- Ý nghĩa cốt lõi đằng sau đống công thức về nhân tử Lagrange:
Bạn muốn tìm một vector hướng $v$ sao cho khi bạn nhân ma trận hiệp biến $\Sigma$ với $v$, kết quả thu được là một vector mới cùng phương (chỉ bị kéo dài hoặc thu ngắn lại) so với chính nó. $$\text{Ma trận hiệp biến} \times \text{Vector hướng} = \text{Độ dài thay đổi} \times \text{Vector hướng đó}$$
- Áp dụng QR cho việc tìm nghiệm lambda, thu đc nghiệm trên đường chéo chính của A cuối cùng sau khi lặp.
## Key learnings
- Quá trình Gram-Schrmidt
- Thuật toán QR
- Ma trận trực giao, ma trận tam giác trên
- PCA
- Giảm chiều dữ liệu
- ĐỊnh thức ma trận
- orthonormal basis
