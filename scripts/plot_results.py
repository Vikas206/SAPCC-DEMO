"""
Builds the result charts from data/phase5_results.txt -- the final ablation:
5 concurrency-control strategies x 6 real workloads (YCSB A/B/C/D/F + TPC-C),
SAPCC routed by the actual live-served, Platt-calibrated model over HTTP
(not the oracle stand-in), so these numbers include the real network +
inference cost, not an idealized one.

Strategies, in the order every chart uses them:
  static_occ          -- always-optimistic baseline (no locking)
  static_pessimistic   -- always-pessimistic baseline (locks every key)
  reactive_switch      -- prior-art baseline: try OCC, retry pessimistic on
                          abort (the mechanism in US 6,681,226-style systems)
  ml_only_foresight    -- ablation: per-key frequency predictor, online-
                          learned, NO semantic/text layer (ForeSight-style)
  sapcc                -- this project: calibrated semantic classifier gates
                          OCC vs. pessimistic before execution

Run from the scripts/ directory: python plot_results.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_FILE = DATA_DIR / "phase5_results.txt"

STRATEGIES = ["static_occ", "static_pessimistic", "reactive_switch", "ml_only_foresight", "sapcc"]
LABELS = {
    "static_occ": "Static OCC",
    "static_pessimistic": "Static Pessimistic",
    "reactive_switch": "Reactive Switch\n(prior art)",
    "ml_only_foresight": "ML-Only\n(ForeSight-style)",
    "sapcc": "SAPCC\n(this work)",
}
COLORS = {
    "static_occ": "#9fb8c8",
    "static_pessimistic": "#6b7f8c",
    "reactive_switch": "#4a6fa5",
    "ml_only_foresight": "#3f9179",
    "sapcc": "#c9622f",
}
WORKLOADS = ["YCSB-A", "YCSB-B", "YCSB-C", "YCSB-D", "YCSB-F", "TPCC"]


def load_results():
    rows = []
    with open(RESULTS_FILE) as f:
        for line in f:
            if line.startswith("workload") or not line.strip():
                continue
            parts = line.split()
            rows.append({
                "workload": parts[0],
                "strategy": parts[1],
                "abort_pct": float(parts[2].rstrip("%")),
                "locked_pct": float(parts[3].rstrip("%")),
                "tps": float(parts[4]),
                "p50ms": float(parts[5]),
                "p99ms": float(parts[6]),
            })
    return rows


def grouped_bar(rows, metric, ylabel, title, fname, logy=False, pct=False):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_strat = len(STRATEGIES)
    x = np.arange(len(WORKLOADS))
    width = 0.8 / n_strat

    for i, strat in enumerate(STRATEGIES):
        vals = []
        for wl in WORKLOADS:
            v = next(r[metric] for r in rows if r["workload"] == wl and r["strategy"] == strat)
            vals.append(v)
        offset = (i - (n_strat - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=LABELS[strat].replace("\n", " "),
                       color=COLORS[strat], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(DATA_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def latency_panel(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in [(axes[0], "p50ms", "p50 latency (ms)"), (axes[1], "p99ms", "p99 latency (ms)")]:
        x = np.arange(len(WORKLOADS))
        width = 0.8 / len(STRATEGIES)
        for i, strat in enumerate(STRATEGIES):
            vals = [next(r[metric] for r in rows if r["workload"] == wl and r["strategy"] == strat) for wl in WORKLOADS]
            offset = (i - (len(STRATEGIES) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=LABELS[strat].replace("\n", " "),
                   color=COLORS[strat], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    fig.suptitle("Commit latency under live-served SAPCC (real HTTP round-trip to the\ncalibrated model, not the oracle stand-in)", y=1.03)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "latency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved latency_comparison.png")


def wasted_work_reduction(rows):
    # The core claim: SAPCC vs. the actual prior-art baseline (reactive
    # switching), not vs. the naive static strategies.
    fig, ax = plt.subplots(figsize=(9, 5))
    reactive = {wl: next(r["abort_pct"] for r in rows if r["workload"] == wl and r["strategy"] == "reactive_switch") for wl in WORKLOADS}
    sapcc = {wl: next(r["abort_pct"] for r in rows if r["workload"] == wl and r["strategy"] == "sapcc") for wl in WORKLOADS}
    mlonly = {wl: next(r["abort_pct"] for r in rows if r["workload"] == wl and r["strategy"] == "ml_only_foresight") for wl in WORKLOADS}

    x = np.arange(len(WORKLOADS))
    width = 0.25
    ax.bar(x - width, [reactive[wl] for wl in WORKLOADS], width, label="Reactive Switch (prior art)", color=COLORS["reactive_switch"])
    ax.bar(x, [mlonly[wl] for wl in WORKLOADS], width, label="ML-Only (ForeSight-style)", color=COLORS["ml_only_foresight"])
    ax.bar(x + width, [sapcc[wl] for wl in WORKLOADS], width, label="SAPCC (this work)", color=COLORS["sapcc"])

    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS)
    ax.set_ylabel("Wasted-work rate (aborted/retried txns, %)")
    ax.set_title("Wasted work: predictive gating vs. reactive retry-after-abort")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "wasted_work_reduction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved wasted_work_reduction.png")


def main():
    rows = load_results()
    grouped_bar(rows, "abort_pct", "Abort / wasted-work rate (%)",
                "Wasted work by strategy across real workloads", "abort_rate_comparison.png", pct=True)
    grouped_bar(rows, "tps", "Throughput (commits/sec)",
                "Throughput by strategy across real workloads (log scale)", "throughput_comparison.png", logy=True)
    latency_panel(rows)
    wasted_work_reduction(rows)


if __name__ == "__main__":
    main()
