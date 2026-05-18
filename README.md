# ML from Scratch

> Build every model by hand. Understand before you use.

## Philosophy

- **No black boxes** — every line of code has a comment explaining *why*, not just *what*
- **Numpy first** — no framework magic until you've written it by hand
- **Verify everything** — compare against sklearn/PyTorch to confirm your implementation is correct
- **Explain before moving on** — don't proceed to the next project until you can explain the current one in plain language

## Rules

|  | Allowed | Not allowed |
|---|---|---|
| **Phase 1–2** | `numpy`, `matplotlib` | `sklearn` models, `torch.nn` |
| **Phase 3** | `numpy` → then `pytorch` | `torch.nn.Linear` in numpy version |
| **Phase 4** | `pytorch` | pre-built transformer libraries |
| **Always** | `sklearn.datasets` (load data) | `sklearn` models as replacement |
| **Always** | `sklearn.metrics` (verify result) | copying implementations |

## Progress Tracker

### 01 — Math Foundations
- [ ] Linear Algebra — matrix ops, dot product, SVD
- [ ] Calculus — gradient, chain rule by hand
- [ ] Probability — MLE, distributions, Bayes

### 02 — Classical ML
- [ ] Linear Regression — MSE loss, gradient descent
- [ ] Logistic Regression — sigmoid, binary cross-entropy
- [ ] Decision Tree — information gain, Gini split
- [ ] K-Means — centroid update, unsupervised loop

### 03 — Neural Networks
- [ ] MLP (numpy) — forward + backprop, NO autograd
- [ ] MLP (PyTorch) — same model, compare with numpy version
- [ ] CNN (PyTorch) — convolution, pooling, spatial features

### 04 — Transformer
- [ ] Positional Encoding — why it's needed, visualize
- [ ] Self-Attention — from numpy, understand Q/K/V
- [ ] Transformer Encoder — full block, train on small task

## Stack

| Phase | Tools |
|-------|-------|
| 01 Math | Python, NumPy |
| 02 Classical ML | NumPy, Matplotlib |
| 03 Neural Networks | NumPy → PyTorch |
| 04 Transformer | PyTorch |
| Verify | scikit-learn (datasets + metrics only) |

## Repo Structure

```
ml-from-scratch/
│
├── 01_math_foundations/
│   ├── linear_algebra/
│   ├── calculus/
│   └── probability/
│
├── 02_classical_ml/
│   ├── linear_regression/
│   ├── logistic_regression/
│   ├── decision_tree/
│   └── kmeans/
│
├── 03_neural_networks/
│   ├── mlp_numpy/           # backprop tự viết, không dùng autograd
│   ├── mlp_pytorch/         # same model — so sánh với numpy version
│   └── cnn_pytorch/
│
├── 04_transformer/
│   ├── positional_encoding/
│   ├── attention_scratch/   # self-attention từ numpy
│   └── transformer_encoder/
│
└── shared/
    ├── data_utils.py        # train_test_split, normalize, standardize
    └── viz_utils.py         # loss curve, decision boundary, confusion matrix
```

## Learning Log

> Ghi lại sau mỗi project — không phải kết quả, mà là điều bạn hiểu thêm.

| Project | Xong | Điều học được |
|---------|------|---------------|
| Linear Algebra | | |
| Linear Regression | | |
| Logistic Regression | | |
| Decision Tree | | |
| K-Means | | |
| MLP numpy | | |
| MLP PyTorch | | |
| CNN PyTorch | | |
| Positional Encoding | | |
| Self-Attention | | |
| Transformer Encoder | | |

---

*Repo tiếp theo sau khi xong: `rag-from-scratch` — lúc đó sẽ hiểu tại sao embedding hoạt động.*
