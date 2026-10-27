"""Binary logistic regression from scratch with gradient descent."""
import numpy as np


def sigmoid(z):
    """Numerically stable sigmoid (no overflow for large |z|)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegressionGD:
    """Minimizes binary cross-entropy with optional L2 regularization:
    loss = -mean(y*log(p) + (1-y)*log(1-p)) + 0.5 * l2 * ||w||^2,  p = sigmoid(X @ w + b)
    Gradient: dL/dw = X.T @ (p - y) / n + l2 * w,  dL/db = mean(p - y)
    (the bias is not regularized)

    Why l2 matters: on (nearly) separable data the unregularized optimum has
    weights that grow without bound, so "where gradient descent stops" depends
    on the iteration count. L2 gives a single well-defined solution.
    """

    def __init__(self, lr=0.1, n_iters=2000, l2=0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []
        eps = 1e-12  # keeps log() away from 0

        for _ in range(self.n_iters):
            p = sigmoid(X @ self.w + self.b)
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            loss += 0.5 * self.l2 * np.sum(self.w ** 2)
            self.loss_history.append(loss)
            grad = p - y
            self.w -= self.lr * ((X.T @ grad) / n + self.l2 * self.w)
            self.b -= self.lr * grad.mean()
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, dtype=float) @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
