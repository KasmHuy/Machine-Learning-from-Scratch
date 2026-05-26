import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import model

if __name__ == "__main__":
    # 1. Chuẩn bị dữ liệu Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Huấn luyện cây quyết định (Giới hạn độ sâu bằng 3 để mô hình vẽ ra vừa vặn khung hình)
    my_tree = model.DecisionTree(max_depth=3, min_samples=2, criterion='gini', debug=True)
    my_tree.fit(X_train, y_train)
    
    # 3. Tính độ chính xác của mô hình
    preds = my_tree.predict(X_test)
    accuracy = np.sum(preds == y_test) / len(y_test)
    print(f"Độ chính xác trên tập Test: {accuracy * 100:.2f}%")

    # 4. Kích hoạt vẽ cây quyết định bằng matplotlib
    print("Đang sinh đồ thị cây quyết định...")
    model.plot_tree(my_tree.root, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.tight_layout()
    plt.show()  # Lệnh hiển thị cửa sổ đồ thị
    
    