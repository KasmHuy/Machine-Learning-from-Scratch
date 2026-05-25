# TODO: training/demo script
import numpy as np
if __name__ == "__main__":
    X_train = np.array([
        [34, 85],  # Trời nóng, ẩm cao -> Không đi        0
        [31, 80],  # Trời nóng, ẩm cao -> Không đi        0
        [26, 70],  # Trời mát, ẩm vừa  -> Có đi           1
        [22, 65],  # Trời se lạnh, ẩm thấp -> Có đi       1
        [20, 90],  # Trời lạnh, ẩm quá cao -> Không đi    0
        [24, 60],  # Trời mát, ẩm thấp -> Có đi           1
        [15, 50],  # Trời rất lạnh, ẩm thấp -> Không đi   0
        [28, 75],  # Trời ấm, ẩm vừa -> Có đi             1
    ])

    # y là nhãn (labels): 0 = Không đi chơi, 1 = Có đi chơi
    y_train = np.array([0, 0, 1, 1, 0, 1, 0, 1])

    print(np.unique(y_train, return_counts =True)) ## trả về [các loại nhãn] và [số lượng thuộc loại của các nhãn khác nhau] tồn tại trong y