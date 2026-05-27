import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, max_depth=50, min_samples=2, criterion='gini', debug=False):
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.root        = None
        self.criterion   = criterion
        self.debug       = debug # Bật/Tắt chế độ hiển thị log chi tiết

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def _impurity(self, y):
        if self.criterion == 'entropy':
            return self._entropy(y)
        else:
            return self._gini(y)
    
    def _information_gain(self, y, y_left, y_right):
        p_left = len(y_left) / len(y)
        p_right = len(y_right) / len(y)
        gain = self._impurity(y) - (p_left * self._impurity(y_left) + p_right * self._impurity(y_right))
        return gain

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None

        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask  = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left  = y[left_mask]
                y_right = y[right_mask]

                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh, best_gain
    
    def fit(self, X, y):
        if self.debug:
            print(f"================ BẮT ĐẦU HUẤN LUYỆN CÂY ({self.criterion.upper()}) ================")
            print(f"Kích thước tập Train: X={X.shape}, y={y.shape}")
            print(f"Cấu hình: max_depth={self.max_depth}, min_samples={self.min_samples}\n")
        
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_labels, label_counts = np.unique(y, return_counts=True)
        n_labels = len(unique_labels)
        
        # Thụt lề thụ động dựa theo độ sâu để log phân cấp như sơ đồ cây
        indent = "  " * depth

        # --- BƯỚC 2.1: KIỂM TRA ĐIỀU KIỆN DỪNG ---
        if (depth >= self.max_depth or n_samples < self.min_samples or n_labels == 1):
            leaf_value = np.bincount(y).argmax()
            
            if self.debug:
                reason = []
                if depth >= self.max_depth: reason.append(f"Đạt độ sâu tối đa ({depth}>={self.max_depth})")
                if n_samples < self.min_samples: reason.append(f"Số mẫu quá ít ({n_samples}<{self.min_samples})")
                if n_labels == 1: reason.append("Dữ liệu tinh khiết (chỉ còn 1 nhãn)")
                
                print(f"{indent} [NÚT LÁ TẦNG {depth}] Tạo nút lá. Lý do: {', '.join(reason)}")
                print(f"{indent}   -> Số lượng mẫu tại lá: {n_samples}, Phân bố: {dict(zip(unique_labels, label_counts))}")
                print(f"{indent}   -> Dự đoán nhãn (Bầu chọn đa số): {leaf_value}\n")
                
            return Node(value=leaf_value)

        # --- BƯỚC 2.2: TÌM ĐIỂM CHIA TỐT NHẤT ---
        best_feat, best_thresh, best_gain = self._best_split(X, y)

        if best_feat is None or best_gain <= 0:
            leaf_value = np.bincount(y).argmax()
            if self.debug:
                print(f"{indent}[NÚT LÁ TẦNG {depth}] Không tìm thấy điểm chia hợp lý giúp tăng Gain. Tạo nút lá. Dự đoán: {leaf_value}\n")
            return Node(value=leaf_value)

        if self.debug:
            print(f"{indent} [NÚT QUYẾT ĐỊNH TẦNG {depth}] Chọn đặc trưng cột số {best_feat} với ngưỡng <= {best_thresh:.4f}")
            print(f"{indent}   -> Gain đạt được: {best_gain:.4f} | Tổng số mẫu hiện tại: {n_samples}")

        # --- BƯỚC 2.3: CHẺ DỮ LIỆU VÀ ĐỆ QUY SANG 2 NHÁNH CON ---
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        if self.debug:
            print(f"{indent}   ├─ Rẽ TRÁI (Yes): Chuyển đi {left_mask.sum()} mẫu")
            print(f"{indent}   └─ Rẽ PHẢI (No) : Chuyển đi {right_mask.sum()} mẫu\n")

        left_child = self._build(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build(X[right_mask], y[right_mask], depth + 1)

        # --- BƯỚC 2.4: TRẢ VỀ NÚT QUYẾT ĐỊNH ---
        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

def plot_tree(node, feature_names=None, class_names=None, x=0, y=1, dx=1, dy=0.15, ax=None):
    """Hàm đệ quy để vẽ cấu trúc cây quyết định tự chế lên giao diện đồ họa"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

    if node is None:
        return

    # Nếu là nút lá (Leaf Node)
    if node.is_leaf_node():
        label = class_names[node.value] if class_names is not None else f"Class {node.value}"
        bbox_props = dict(boxstyle="round,pad=0.6", fc="#a1f5a4", ec="green", lw=2)
        ax.text(x, y, f"LÁ\nDự đoán:\n{label}", ha="center", va="center", bbox=bbox_props, fontsize=9)
        return

    # Nếu là nút quyết định (Decision Node)
    feat_name = feature_names[node.feature] if feature_names is not None else f"X[{node.feature}]"
    node_text = f"{feat_name}\n<= {node.threshold:.2f}"
    
    bbox_props = dict(boxstyle="square,pad=0.5", fc="#bce1f7", ec="blue", lw=1.5)
    ax.text(x, y, node_text, ha="center", va="center", bbox=bbox_props, fontsize=9)

    # Tính toán tọa độ và vẽ nhánh con bên trái (Màu đỏ/True)
    if node.left:
        x_left = x - dx
        y_left = y - dy
        ax.plot([x, x_left], [y - 0.03, y_left + 0.03], c="gray", lw=1.5, ls="--")
        ax.text(x - dx/2, y - dy/2, "Yes", color="green", ha="right", va="bottom", fontsize=8)
        plot_tree(node.left, feature_names, class_names, x_left, y_left, dx/1.4, dy, ax)

    # Tính toán tọa độ và vẽ nhánh con bên phải (Màu xanh/False)
    if node.right:
        x_right = x + dx
        y_right = y - dy
        ax.plot([x, x_right], [y - 0.03, y_right + 0.03], c="gray", lw=1.5, ls="--")
        ax.text(x + dx/2, y - dy/2, "No", color="red", ha="left", va="bottom", fontsize=8)
        plot_tree(node.right, feature_names, class_names, x_right, y_right, dx/1.4, dy, ax)


