"""Draw graphs from results/results.csv.

1. speed.png: lookup time vs number of models, one panel per distribution (largest size)
2. size_vs_speed.png: index size vs lookup time at the largest size
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"


def main():
    rows = list(csv.DictReader(open(RESULTS / "results.csv", newline="")))
    biggest = max(int(r["n_keys"]) for r in rows if r["distribution"] == "uniform")
    dists = ["uniform", "lognormal", "timestamps"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, dist in zip(axes, dists):
        sel = [r for r in rows if r["distribution"] == dist]
        n = max(int(r["n_keys"]) for r in sel)
        sel = [r for r in sel if int(r["n_keys"]) == n]
        rmi = sorted((int(r["n_models"]), float(r["ns_per_lookup"])) for r in sel if r["method"] == "learned_index_rmi")
        bs = next(float(r["ns_per_lookup"]) for r in sel if r["method"] == "binary_search")
        ax.plot([m for m, _ in rmi], [t for _, t in rmi], "o-", label="learned index (RMI)")
        ax.axhline(bs, color="gray", linestyle="--", label="binary search")
        ax.set_xscale("log")
        ax.set_title(f"{dist} ({n / 1e6:.1f}M keys)")
        ax.set_xlabel("number of stage-2 models")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("ns per lookup (lower is better)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "speed.png", dpi=150)
    print("saved results/speed.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dist in dists:
        sel = [r for r in rows if r["distribution"] == dist and r["method"] == "learned_index_rmi"]
        n = max(int(r["n_keys"]) for r in sel)
        pts = sorted((int(r["index_size_bytes"]) / 1024, float(r["ns_per_lookup"])) for r in sel if int(r["n_keys"]) == n)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"RMI, {dist}")
    bt = next(int(r["index_size_bytes"]) for r in rows if r["method"] == "binary_search" and int(r["n_keys"]) == biggest)
    ax.axvline(bt / 1024, color="gray", linestyle=":", label="B+ tree inner nodes (est.)")
    ax.set_xscale("log")
    ax.set_xlabel("index size (KB, log scale)")
    ax.set_ylabel("ns per lookup")
    ax.set_title("Index size vs lookup speed (largest data size)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "size_vs_speed.png", dpi=150)
    print("saved results/size_vs_speed.png")


if __name__ == "__main__":
    main()
