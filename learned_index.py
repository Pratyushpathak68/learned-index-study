"""Learned Index vs Binary Search: a small replication of
"The Case for Learned Index Structures" (Kraska et al., SIGMOD 2018).

Idea: a sorted array of keys is like a function key -> position (a CDF).
Instead of a B-tree or binary search over the whole array, a model can
*predict* where a key is, and we only search a small window around the
prediction. This file implements a 2-stage Recursive Model Index (RMI):

    stage 1: one linear model picks which stage-2 model to use
    stage 2: M linear models, each predicts the position for its key range,
             and remembers its worst prediction error on the training keys

Lookup = predict position, then binary-search only inside [pos - err, pos + err].
Because every stage-2 model stores its max error, the search is guaranteed
to find the key (correctness is checked in benchmark.py).
"""
import numpy as np


def fit_line(x: np.ndarray, y: np.ndarray):
    """Least-squares line y = a*x + b. Falls back to a constant if x is flat."""
    if len(x) == 0:
        return 0.0, 0.0
    if len(x) == 1 or x[-1] == x[0]:
        return 0.0, float(y.mean())
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


class RMI:
    def __init__(self, keys: np.ndarray, n_models: int):
        self.keys = keys
        self.n = len(keys)
        self.m = n_models
        positions = np.arange(self.n, dtype=np.float64)

        # Stage 1: map key -> model id in [0, m)
        self.a1, self.b1 = fit_line(keys, positions * self.m / self.n)
        model_id = self._stage1(keys)

        # Stage 2: one line per model, plus its error bound
        self.a2 = np.zeros(self.m)
        self.b2 = np.zeros(self.m)
        self.err = np.zeros(self.m, dtype=np.int64)
        order = np.argsort(model_id, kind="stable")
        bounds = np.searchsorted(model_id[order], np.arange(self.m + 1))
        for j in range(self.m):
            idx = order[bounds[j]:bounds[j + 1]]
            if len(idx) == 0:
                continue
            a, b = fit_line(keys[idx], positions[idx])
            pred = np.clip(np.rint(a * keys[idx] + b), 0, self.n - 1)
            self.a2[j], self.b2[j] = a, b
            self.err[j] = int(np.abs(pred - positions[idx]).max())

    def _stage1(self, q: np.ndarray) -> np.ndarray:
        return np.clip((self.a1 * q + self.b1).astype(np.int64), 0, self.m - 1)

    def predict(self, q: np.ndarray):
        j = self._stage1(q)
        pos = np.clip(np.rint(self.a2[j] * q + self.b2[j]), 0, self.n - 1).astype(np.int64)
        e = self.err[j]
        lo = np.maximum(pos - e, 0)
        hi = np.minimum(pos + e + 1, self.n)
        return lo, hi

    def lookup(self, q: np.ndarray):
        lo, hi = self.predict(q)
        return bounded_search(self.keys, q, lo, hi)

    def size_bytes(self) -> int:
        # 2 floats stage 1 + (2 floats + 1 int) per stage-2 model
        return 16 + self.m * (8 + 8 + 8)


def bounded_search(keys, q, lo, hi):
    """Vectorised binary search for each query inside its own [lo, hi) window.
    Returns (positions, steps) where steps = number of halving iterations."""
    lo = lo.copy()
    hi = hi.copy()
    steps = 0
    while True:
        active = lo < hi
        if not active.any():
            break
        mid = (lo + hi) // 2
        go_right = active & (keys[np.minimum(mid, len(keys) - 1)] < q)
        lo = np.where(go_right, mid + 1, lo)
        hi = np.where(active & ~go_right, mid, hi)
        steps += 1
    return lo, steps


def full_binary_search(keys, q):
    """Baseline: the same search over the whole array."""
    zeros = np.zeros(len(q), dtype=np.int64)
    full = np.full(len(q), len(keys), dtype=np.int64)
    return bounded_search(keys, q, zeros, full)


def btree_size_bytes(n: int, fanout: int = 256) -> int:
    """Approximate size of the inner nodes of a B+ tree with 8-byte keys and
    8-byte child pointers. Leaves are pages of `fanout` sorted keys (the data
    itself, not counted); inner nodes index those leaf pages."""
    total, level = 0, -(-n // fanout)  # number of leaf pages
    while level > 1:
        level = -(-level // fanout)  # ceil division
        total += level * fanout * 16
    return total
