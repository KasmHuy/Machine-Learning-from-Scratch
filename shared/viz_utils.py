"""
shared/viz_utils.py
Visualization helpers dùng chung.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(losses, title="Training Loss"):
    """Vẽ loss theo epoch. Gọi sau mỗi training loop."""
    plt.figure(figsize=(8, 4))
    plt.plot(losses, color="#e74c3c", linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Visualize decision boundary cho 2D feature space.
    model phải có method .predict(X) → array of labels.
    """
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',
                          s=40, cmap="RdYlBu")
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=None):
    """Hiển thị confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    if class_names:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.show()
