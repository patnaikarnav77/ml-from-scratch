"""Check every from-scratch model against scikit-learn.
scikit-learn is used ONLY here, as the reference to compare against."""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from decision_tree import DecisionTreeScratch
from linear_regression import LinearRegressionGD, LinearRegressionNormal
from logistic_regression import LogisticRegressionGD
from utils import StandardScaler, accuracy, r2_score

SEED = 42

# ---------------------------------------------------------------- linear
print("=" * 64)
print("LINEAR REGRESSION  (diabetes dataset)")
print("=" * 64)
X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

ref = LinearRegression().fit(Xtr_s, ytr)
gd = LinearRegressionGD(lr=0.1, n_iters=3000).fit(Xtr_s, ytr)
ne = LinearRegressionNormal().fit(Xtr_s, ytr)

print(f"{'model':<22}{'test R2':>10}{'max |pred - sklearn|':>24}")
for name, m in [("sklearn", ref), ("scratch (gradient)", gd), ("scratch (normal eq)", ne)]:
    pred = m.predict(Xte_s)
    diff = np.max(np.abs(pred - ref.predict(Xte_s)))
    print(f"{name:<22}{r2_score(yte, pred):>10.4f}{diff:>24.6f}")

# -------------------------------------------------------------- logistic
print("\n" + "=" * 64)
print("LOGISTIC REGRESSION  (breast cancer dataset)")
print("=" * 64)
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

# sklearn's objective: C * sum(log-loss) + 0.5 * ||w||^2
# Dividing by C*n gives mean(log-loss) + (1 / (2*C*n)) * ||w||^2  ->  l2 = 1 / (C*n)
C = 1.0
ref = LogisticRegression(C=C, max_iter=10000).fit(Xtr_s, ytr)
mine = LogisticRegressionGD(lr=0.1, n_iters=5000, l2=1.0 / (C * len(ytr))).fit(Xtr_s, ytr)

p_ref, p_mine = ref.predict_proba(Xte_s)[:, 1], mine.predict_proba(Xte_s)
print(f"sklearn accuracy:   {accuracy(yte, ref.predict(Xte_s)):.4f}")
print(f"scratch accuracy:   {accuracy(yte, mine.predict(Xte_s)):.4f}")
print(f"prediction agreement: {np.mean(ref.predict(Xte_s) == mine.predict(Xte_s)) * 100:.1f}%")
print(f"max |prob diff|:    {np.max(np.abs(p_ref - p_mine)):.4f}")

# ------------------------------------------------------------------ tree
print("\n" + "=" * 64)
print("DECISION TREE  (breast cancer dataset, max_depth=5)")
print("=" * 64)
ref = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(Xtr, ytr)
mine = DecisionTreeScratch(max_depth=5).fit(Xtr, ytr)

print(f"sklearn accuracy:   {accuracy(yte, ref.predict(Xte)):.4f}")
print(f"scratch accuracy:   {accuracy(yte, mine.predict(Xte)):.4f}")
print(f"prediction agreement: {np.mean(ref.predict(Xte) == mine.predict(Xte)) * 100:.1f}%")
print(f"depth: sklearn={ref.get_depth()}  scratch={mine.depth()}")
