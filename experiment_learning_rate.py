"""Experiment: what happens to gradient descent at different learning rates?
Saves lr_experiment.png and prints a summary table."""
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

from linear_regression import LinearRegressionGD, LinearRegressionNormal
from utils import StandardScaler

X, y = load_diabetes(return_X_y=True)
Xs = StandardScaler().fit_transform(X)

best_loss = np.mean((LinearRegressionNormal().fit(Xs, y).predict(Xs) - y) ** 2)
learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
N_ITERS = 200

plt.figure(figsize=(8, 5))
print(f"Optimal MSE (normal equation): {best_loss:.2f}\n")
print(f"{'lr':>8}{'final MSE':>16}  result")

for lr in learning_rates:
    with np.errstate(all="ignore"):
        model = LinearRegressionGD(lr=lr, n_iters=N_ITERS).fit(Xs, y)
    hist = np.array(model.loss_history)
    final = hist[-1]
    if not np.isfinite(final) or final > hist[0] * 10:
        status = "DIVERGED"
    elif final < best_loss * 1.01:
        status = "converged"
    else:
        status = "too slow (not converged yet)"
    shown = "overflow" if status == "DIVERGED" and (not np.isfinite(final) or final > 1e12) else f"{final:.2f}"
    print(f"{lr:>8}{shown:>16}  {status}")
    plt.plot(np.clip(hist, 0, 1e6), label=f"lr={lr}")

plt.axhline(best_loss, color="gray", linestyle="--", label="optimal")
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("MSE (log scale, clipped at 1e6)")
plt.title("Gradient descent: effect of learning rate")
plt.legend()
plt.tight_layout()
plt.savefig("lr_experiment.png", dpi=150)
print("\nSaved lr_experiment.png")
