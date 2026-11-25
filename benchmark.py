"""Benchmark the learned index (RMI) against binary search.

For each key distribution and data size:
  - build RMIs with different numbers of stage-2 models
  - look up a batch of existing keys
  - check every result is correct (same position as numpy.searchsorted)
  - record lookup time, search steps, average/max error window, index size

Usage:
    python benchmark.py                 # sizes 100K, 1M, 10M
    python benchmark.py --sizes 100000  # quick test
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from learned_index import RMI, btree_size_bytes, full_binary_search

RESULTS = Path(__file__).parent / "results"
N_QUERIES = 200_000
REPEATS = 5
MODEL_COUNTS = [100, 1_000, 10_000, 100_000]


def make_keys(dist: str, n: int, rng) -> np.ndarray:
    if dist == "uniform":
        k = rng.uniform(0, 1e9, n)
    elif dist == "lognormal":       # skewed, like prices or file sizes
        k = rng.lognormal(0, 2, n) * 1e6
    elif dist == "timestamps":      # bursty, like order times: busy days and quiet days
        days = rng.choice(730, n, p=rng.dirichlet(np.ones(730) * 0.3))
        k = days * 86_400 + rng.uniform(0, 86_400, n)
    else:
        raise ValueError(dist)
    return np.unique(k)  # sorted, no duplicates


def best_time(fn):
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return min(times), out


def main(sizes):
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    for dist in ["uniform", "lognormal", "timestamps"]:
        for n in sizes:
            keys = make_keys(dist, n, rng)
            q = keys[rng.integers(0, len(keys), N_QUERIES)]
            truth = np.searchsorted(keys, q)

            t_bs, (pos_bs, steps_bs) = best_time(lambda: full_binary_search(keys, q))
            assert np.array_equal(pos_bs, truth)
            t_np, _ = best_time(lambda: np.searchsorted(keys, q))
            rows.append([dist, len(keys), "binary_search", "", round(t_bs / N_QUERIES * 1e9, 1),
                         steps_bs, len(keys), btree_size_bytes(len(keys))])
            rows.append([dist, len(keys), "numpy_searchsorted_C", "", round(t_np / N_QUERIES * 1e9, 1),
                         "", "", ""])
            print(f"\n{dist:<10} n={len(keys):>10,}  binary search: {t_bs / N_QUERIES * 1e9:7.1f} ns/lookup, "
                  f"{steps_bs} steps")

            for m in MODEL_COUNTS:
                if m > len(keys) // 10:
                    continue
                t0 = time.perf_counter()
                rmi = RMI(keys, m)
                build_s = time.perf_counter() - t0
                t_rmi, (pos, steps) = best_time(lambda: rmi.lookup(q))
                assert np.array_equal(pos, truth), "learned index returned a wrong position"
                lo, hi = rmi.predict(q)
                avg_window = float((hi - lo).mean())
                rows.append([dist, len(keys), "learned_index_rmi", m, round(t_rmi / N_QUERIES * 1e9, 1),
                             steps, round(avg_window, 1), rmi.size_bytes()])
                print(f"  RMI m={m:>7,}: {t_rmi / N_QUERIES * 1e9:7.1f} ns/lookup, {steps:>2} steps, "
                      f"avg window {avg_window:9.1f}, size {rmi.size_bytes() / 1024:8.1f} KB "
                      f"(B+ tree inner nodes ~{btree_size_bytes(len(keys)) / 1024:8.1f} KB), build {build_s:.2f}s")

    with open(RESULTS / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distribution", "n_keys", "method", "n_models", "ns_per_lookup",
                    "search_steps", "avg_search_window", "index_size_bytes"])
        w.writerows(rows)
    print("\nAll lookups verified correct. Saved results/results.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000, 10_000_000])
    main(p.parse_args().sizes)
