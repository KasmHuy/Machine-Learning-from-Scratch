"""
shared/data_utils.py
Các utility dùng chung cho tất cả projects.
Không import bất kỳ thư viện ngoài numpy/pandas.
"""
import numpy as np


def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Split data thành train/test.
    WHY scratch: hiểu random permutation thay vì dùng sklearn như blackbox.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize(X):
    """Min-max normalize về [0, 1]. Dùng cho non-Gaussian data."""
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8)


def standardize(X):
    """Z-score standardize. Dùng khi data gần Gaussian (thường dùng hơn)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + 1e-8)


def accuracy(y_true, y_pred):
    """Simple accuracy score."""
    return np.mean(y_true == y_pred)
