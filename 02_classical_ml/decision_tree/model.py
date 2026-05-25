import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # 1. Nhóm thuộc tính dành cho Nút quyết định (Decision Node)
        self.feature = feature        # Chỉ số (index) của cột đặc trưng dùng để chia dữ liệu
        self.threshold = threshold    # Giá trị ngưỡng (mốc) để phân tách dữ liệu

        # 2. Nhóm thuộc tính liên kết (Pointers)
        self.left = left              # Trỏ đến Node con bên trái (khi giá trị <= threshold)
        self.right = right            # Trỏ đến Node con bên phải (khi giá trị > threshold)

        # 3. Nhóm thuộc tính dành cho Nút lá (Leaf Node)
        self.value = value            # Giá trị dự đoán cuối cùng (Nhãn lớp hoặc giá trị số)

    def is_leaf_node(self):
        """Hàm hỗ trợ nhanh để kiểm tra xem node này có phải là nút lá hay không"""
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth = 50, min_samples=2, criterion='gini'):
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.root        = None
        self.criterion = criterion

    def _gini(self, y):
        classes , counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2) # công thức gini
    
    def _entropy(self, y):
        classes , counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return - sum(probs * np.log2(probs))
    
    def _information_gain(self, y, y_left, y_right):
        # 1. Tính trọng số (tỷ lệ phần tử) của mỗi nhánh con
        p_left = len(y_left) / len(y)
        p_right = len(y_right) / len(y)
        
        # 2. Tính Information Gain (sử dụng hàm entropy bạn đã viết ở trên)
        # Lưu ý: Bạn có thể đổi self.entropy thành self._gini tùy thuộc vào cấu hình cây
        gain = self.entropy(y) - (p_left * self.entropy(y_left) + p_right * self.entropy(y_right))
        return gain
    def _impurity(self, y):
        """Hàm trung gian tự động chọn thuật toán dựa trên cấu hình"""
        if self.criterion == 'entropy':
            return self._entropy(y)
        else:
            return self._gini(y)
    
    def _best_split(self, X, y, debug = False):
        best_gain, best_feat, best_thres = -1, None, None
        parent = self._impurity(y)
        n = len(y)

        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask  = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left  = y[left_mask]
                y_right = y[right_mask]

                # Tính toán Information Gain chung cho cả 2 loại tiêu chí
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh
    
    def fit(self, X, y):
        """Hàm public: Người dùng sẽ gọi hàm này để bắt đầu train cây"""
        # Kích hoạt hàm xây cây đệ quy từ độ sâu bằng 0 (Nút gốc)
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth=0):
        """Hàm private: Xây dựng cây bằng thuật toán đệ quy chia để trị"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # --- BƯỚC 2.1: KIỂM TRA ĐIỀU KIỆN DỪNG (STOP CONDITIONS) ---
        # Nếu đạt độ sâu tối đa, HOẶC số mẫu quá ít, HOẶC dữ liệu đã sạch (chỉ còn 1 nhãn)
        if (depth >= self.max_depth or 
            n_samples < self.min_samples or 
            n_labels == 1):
            
            # Tính toán Nhãn xuất hiện nhiều nhất tại nút này (Majority Vote)
            # np.bincount(y).argmax() sẽ trả về số (nhãn) có tần suất cao nhất
            leaf_value = np.bincount(y).argmax()
            
            # Trả về một Nút lá (chỉ chứa value, các biến feature, left, right đều bằng None)
            return Node(value=leaf_value)

        # --- BƯỚC 2.2: TÌM ĐIỂM CHIA TỐT NHẤT ---
        best_feat, best_thresh = self._best_split(X, y)

        # Nếu không tìm được điểm chia nào hợp lý, biến nút này thành nút lá luôn
        if best_feat is None:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        # --- BƯỚC 2.3: CHẺ DỮ LIỆU VÀ ĐỆ QUY SANG 2 NHÁNH CON ---
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        # Tự gọi lại chính nó để xây dựng nhánh bên trái (tăng độ sâu lên 1)
        left_child = self._build(X[left_mask], y[left_mask], depth + 1)
        
        # Tự gọi lại chính nó để xây dựng nhánh bên phải (tăng độ sâu lên 1)
        right_child = self._build(X[right_mask], y[right_mask], depth + 1)

        # --- BƯỚC 2.4: TRẢ VỀ NÚT QUYẾT ĐỊNH ---
        # Nối nút hiện tại với 2 nút con vừa tạo ra ở trên
        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

