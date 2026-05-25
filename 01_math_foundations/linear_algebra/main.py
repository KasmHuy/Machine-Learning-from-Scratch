# PCA Reducing data dimension
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class PCAFromScratch:
    def __init__(self, n_components):
        """
        PCA cơ bản từ scratch
        """
        self.n_components = n_components
        self.mean = None
        self.eigenValues = None
        self.eigenVectors = None
    
    def fit(self, X, debug = False, epsilon = 10e-9, max_iter = 1000): 
        """
        X là ma trận dữ liệu (Số mẫu x Số đặc trưng)
        """
        # 1. Tính mean đầu vào
        self.mean = np.mean(X, axis=0)
        if debug: print(f'Mean = {self.mean}')

        # 2. Đưa về center
        X_centered = X - self.mean
        if debug: print(f'Ma trận tâm = {X_centered}')

        # 3. Tính ma trận hiệp biến chuẩn đặc trưng (N_features x N_features)
        # SỬA LỖI: Chuyển vị đúng vị trí (.T) và chia cho (Số mẫu - 1)
        A = (X_centered.T @ X_centered) * (1.0 / (X.shape[0] - 1)) 
        if debug: print(f'Ma trận hiệp biến = {A}')

        # 4. Tìm nghiệm lambda bằng thuật toán QR
        A_k = A.astype(float) 
        V = np.eye(A_k.shape[0]) # Kích thước chuẩn bằng số đặc trưng gốc

        converged = False
        for i in range(max_iter):
            Q, R = np.linalg.qr(A_k)
            A_next = np.dot(R, Q)
            V = np.dot(V, Q) # Nhân tích lũy đúng chuẩn hình học
            
            low_elem = A_next[np.tril_indices(A_next.shape[0], k=-1)]
            
            if np.all(np.abs(low_elem) < epsilon):
                converged = True
                if debug: 
                    print(f"Hội tụ ở vòng thứ {i+1}")
                    print(f'Ma trận cuối (Tam giác): \n{A_next}') # SỬA LỖI: Đổi từ A_k thành A_next
                    print(f'Ma trận Q cuối: \n{Q}')
                    print(f'Ma trận R cuối: \n{R}')
                    print(f'Ma trận Vectơ riêng V tổng hợp: \n{V}')
                break
            else:
                if debug: print(f"Vòng lặp {i+1}: Chưa hội tụ hết")
            
            A_k = A_next

        if not converged and debug: 
            print("Cảnh báo: Thuật toán dừng lại do chạm ngưỡng số vòng lặp tối đa!")

        # 5. LƯU LẠI THÀNH PHẦN CHÍNH (Bộ lọc chuyển đổi không gian cho hàm transform)
        # Trích xuất n_components cột đầu tiên của V ứng với các lambda lớn nhất
        self.components = V[:, :self.n_components]
        
        # Trả về đối tượng self để có thể gọi liên tiếp dạng: pca.fit(X).transform(X)
        return self

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def view3D(self, X, y=None, title="Không gian gốc 3D (Trích 3 đặc trưng đầu)"):
        """
        Trực quan hóa dữ liệu ở không gian 3D ban đầu.
        Do dữ liệu có thể nhiều hơn 3 chiều, hàm này mặc định bốc 3 cột đầu tiên để vẽ.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Nếu không có nhãn màu y, mặc định vẽ cùng 1 màu xanh lục rực rỡ
        colors = y if y is not None else 'forestgreen'
        cmap = 'viridis' if y is not None else None
        
        # Thực hiện vẽ scatter 3D
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap=cmap, s=60, edgecolors='k', alpha=0.8)
        
        # Trang trí đồ thị cho hiện đại và chuyên nghiệp
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Đặc trưng 1", fontsize=10, labelpad=8)
        ax.set_ylabel("Đặc trưng 2", fontsize=10, labelpad=8)
        ax.set_zlabel("Đặc trưng 3", fontsize=10, labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if y is not None:
            cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
            cbar.set_label('Phân lớp (Classes)', fontsize=10)
            
        plt.tight_layout()
        plt.show()

    def view2D(self, X_projected, y=None, title="Không gian nén 2D sau khi chạy PCA"):
        """
        Trực quan hóa dữ liệu phẳng sau khi đã được nén xuống 2 chiều (PC1 và PC2).
        """
        plt.figure(figsize=(9, 7))
        
        colors = y if y is not None else 'royalblue'
        cmap = 'viridis' if y is not None else None
        
        # Vẽ các điểm dữ liệu trên mặt phẳng 2D thành phần chính
        sc = plt.scatter(X_projected[:, 0], X_projected[:, 1], c=colors, cmap=cmap, s=70, edgecolors='k', alpha=0.8)
        
        # Tạo các đường trục tọa độ (0,0) đi qua trung tâm để dễ nhìn phân bố
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        
        # Trang trí thẩm mỹ đồ thị
        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Thành phần chính 1 (PC1)", fontsize=11)
        plt.ylabel("Thành phần chính 2 (PC2)", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        if y is not None:
            cbar = plt.colorbar(sc)
            cbar.set_label('Phân lớp (Classes)', fontsize=10)
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # --- THỬ NGHIỆM 1: Với ma trận X 4x4 góc của bạn ---
    X_simple = np.array([
        [1, -2, 3, 4],
        [1,  0, 1, -1],
        [3, -5, 7, 9],
        [1,  5, 9, -3]
    ])
    
    print("-Test")
    pca_simple = PCAFromScratch(n_components=2)
    pca_simple.fit(X_simple, debug=False)
    X_simple_2d = pca_simple.transform(X_simple)
    
    # Vẽ đồ thị cho ma trận nhỏ của bạn
    pca_simple.view3D(X_simple, title="Dữ liệu 4x4 gốc (Trích 3D)")
    pca_simple.view2D(X_simple_2d, title="Dữ liệu 4x4 nén xuống 2D")
    
    # --- THỬ NGHIỆM 2: Với dữ liệu hoa Iris thực tế (150 mẫu, 4 đặc trưng, 3 nhãn hoa) ---

    iris = load_iris()
    X_iris = iris.data  # Kích thước: 150 mẫu x 4 đặc trưng
    y_iris = iris.target # Nhãn của 3 loài hoa (0, 1, 2)
    
    # Khởi tạo mô hình nén xuống 2 chiều
    pca_iris = PCAFromScratch(n_components=2)
    
    # Học cấu trúc không gian của hoa Iris
    pca_iris.fit(X_iris)
    
    # Hạ chiều toàn bộ dữ liệu hoa Iris
    X_iris_2d = pca_iris.transform(X_iris)
    
    # Gọi hàm vẽ 3D gốc và 2D sau khi nén có truyền thêm nhãn hoa màu sắc `y_iris`
    pca_iris.view3D(X_iris, y=y_iris, title="Iris Dataset - Không gian 3D ban đầu")
    pca_iris.view2D(X_iris_2d, y=y_iris, title="Iris Dataset - Hạ chiều xuống 2D bằng QR-PCA")
    