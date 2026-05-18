"""
Run: python main.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import numpy as np
from sklearn.datasets import load_iris
from shared.data_utils import train_test_split, standardize, accuracy
from shared.viz_utils import plot_loss_curve, plot_decision_boundary
from model import LogisticRegression


def main():
    iris = load_iris()
    mask = iris.target < 2
    X = standardize(iris.data[mask, :2])
    y = iris.target[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LogisticRegression(lr=0.1, n_epochs=500)
    model.fit(X_train, y_train)

    print(f"\nTrain accuracy: {accuracy(y_train, model.predict(X_train)):.4f}")
    print(f"Test  accuracy: {accuracy(y_test,  model.predict(X_test)):.4f}")

    plot_loss_curve(model.losses)
    plot_decision_boundary(model, X, y)


if __name__ == "__main__":
    main()
