"""Decision tree classifier from scratch (Gini impurity, greedy splits)."""
import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # index of feature to split on
        self.threshold = threshold  # go left if x[feature] <= threshold
        self.left = left
        self.right = right
        self.value = value          # class index (only set on leaves)


class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.root = self._grow(X, y_idx, depth=0)
        return self

    # ---- training -------------------------------------------------------
    def _grow(self, X, y, depth):
        n = len(y)
        # stop: pure node, too deep, or too few samples -> make a leaf
        if depth >= self.max_depth or n < self.min_samples_split or len(np.unique(y)) == 1:
            return _Node(value=self._majority(y))

        feature, threshold = self._best_split(X, y)
        if feature is None:  # no valid split exists
            return _Node(value=self._majority(y))

        mask = X[:, feature] <= threshold
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._grow(X[mask], y[mask], depth + 1),
            right=self._grow(X[~mask], y[~mask], depth + 1),
        )

    def _majority(self, y):
        return int(np.argmax(np.bincount(y, minlength=self.n_classes_)))

    def _best_split(self, X, y):
        """Find (feature, threshold) with lowest weighted Gini.
        Sorts each feature once and uses cumulative class counts, so every
        candidate threshold is scored in O(1) instead of re-scanning the data."""
        n, d = X.shape
        onehot = np.eye(self.n_classes_)[y]
        best_gini, best_feature, best_thr = np.inf, None, None
        n_left = np.arange(1, n)[:, None]
        n_right = n - n_left

        for f in range(d):
            order = np.argsort(X[:, f], kind="stable")
            xs = X[order, f]
            oh = onehot[order]

            left_counts = np.cumsum(oh, axis=0)[:-1]   # split after sample i
            right_counts = oh.sum(axis=0) - left_counts

            gini_left = 1 - np.sum((left_counts / n_left) ** 2, axis=1)
            gini_right = 1 - np.sum((right_counts / n_right) ** 2, axis=1)
            weighted = (n_left[:, 0] * gini_left + n_right[:, 0] * gini_right) / n

            valid = xs[:-1] != xs[1:]  # can't split between identical values
            if not valid.any():
                continue
            weighted = np.where(valid, weighted, np.inf)
            i = int(np.argmin(weighted))
            if weighted[i] < best_gini:
                best_gini = weighted[i]
                best_feature = f
                best_thr = (xs[i] + xs[i + 1]) / 2  # midpoint between neighbors
        return best_feature, best_thr

    # ---- prediction -----------------------------------------------------
    def _predict_one(self, x):
        node = self.root
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        idx = np.array([self._predict_one(x) for x in X])
        return self.classes_[idx]

    def depth(self):
        def d(node):
            return 0 if node.value is not None else 1 + max(d(node.left), d(node.right))
        return d(self.root)
