"""
02_classical_ml/logistic_regression/model.py

WHY this exists:
    Logistic Regression là nền tảng của classification.
    Tự implement để hiểu: sigmoid, binary cross-entropy loss,
    và gradient descent — 3 thứ này tái hiện trong mọi neural net sau này.
"""
import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, n_epochs=1000):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        """σ(z) = 1 / (1 + e^{-z}) — ép output về (0,1) như xác suất."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _bce_loss(self, y_true, y_pred):
        """L = -1/n * Σ [y*log(ŷ) + (1-y)*log(1-ŷ)]"""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            y_pred = self._sigmoid(X @ self.weights + self.bias)
            self.losses.append(self._bce_loss(y, y_pred))

            error = y_pred - y
            self.weights -= self.lr * (X.T @ error) / n_samples
            self.bias    -= self.lr * np.mean(error)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {self.losses[-1]:.4f}")
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
