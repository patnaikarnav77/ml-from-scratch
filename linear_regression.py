"""Linear regression from scratch, two ways:
1) Gradient descent (iterative)
2) Normal equation / least squares (closed form)
"""
import numpy as np


class LinearRegressionGD:
    """Minimizes mean squared error with batch gradient descent.

    loss = mean((X @ w + b - y) ** 2)
    dL/dw = (2/n) * X.T @ error      dL/db = (2/n) * sum(error)
    """

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            error = X @ self.w + self.b - y
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)
            if not np.isfinite(loss):  # diverged (learning rate too high)
                break
            self.w -= self.lr * (2 / n) * (X.T @ error)
            self.b -= self.lr * (2 / n) * error.sum()
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b


class LinearRegressionNormal:
    """Closed-form least squares: solves for [w, b] directly, no iterations."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])  # add bias column
        # lstsq is more numerically stable than inverting X.T @ X
        theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self.w = theta[:-1]
        self.b = theta[-1]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b
