"""
Analysis script for the 13-experiment sweep.
Generates summary tables and visualizations.

Usage:
    python notebooks/analysis.py
    # Outputs to notebooks/ directory
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent


def load_all_experiments():
    """Load all experiment results from data/ directory."""
    experiments = {}
    for fpath in sorted(DATA_DIR.glob("*.jsonl")):
        label = fpath.stem
        with open(fpath) as f:
            results = [json.loads(l) for l in f if l.strip()]
        # Separate baseline from guided
        baseline = [r for r in results if r.get("experiment") == "baseline"]
        guided = [r for r in results if "target_sim" in r and r.get("experiment") != "baseline"]
        if guided:
            # Filter degenerate outputs
            valid = [r for r in guided if r.get("non_rep", 1.0) > 0.5]
            experiments[label] = {
                "baseline": baseline,
                "guided": valid,
                "degenerate_count": len(guided) - len(valid),
                "agg": _aggregate(valid),
            }
    return experiments


def _aggregate(results):
    """Group by topic."""
    agg = defaultdict(list)
    for r in results:
        agg[r["experiment"]].append(r)
    return dict(agg)


def print_summary_table(experiments):
    """Print a formatted summary table."""
    print(f"\n{'='*95}")
    print(f"{'Experiment':<32} {'sim_mean':>9} {'good%':>6} {'space':>7} {'ocean':>7} {'horror':>7} {'cook':>7}")
    print(f"{'='*95}")

    for label, data in experiments.items():
        guided = data["guided"]
        if not guided:
            continue
        sims = [r["target_sim"] for r in guided]
        good = sum(1 for r in guided if r["target_sim"] > 0.15 and r.get("coherence", 0) > 0.3)
        pct = 100 * good / len(guided)
        mean_sim = sum(sims) / len(sims)

        topics = ""
        for t in ["space", "ocean", "horror", "cooking"]:
            trials = data["agg"].get(t, [])
            if trials:
                s = sum(x.get("target_sim", 0) for x in trials) / len(trials)
                topics += f" {s:>6.3f}"
            else:
                topics += f" {'--':>6}"

        print(f"{label:<32} {mean_sim:>9.4f} {pct:>5.0f}%{topics}")


def plot_sim_by_experiment(experiments, output_path):
    """Bar chart: mean target_sim by experiment."""
    labels = []
    sims = []
    colors = []

    baseline_sim = None
    for label, data in experiments.items():
        guided = data["guided"]
        if not guided:
            continue
        mean_sim = sum(r["target_sim"] for r in guided) / len(guided)
        if label == "e00_baseline":
            baseline_sim = mean_sim
        labels.append(label.replace("_", "\n"))
        sims.append(mean_sim)
        good = sum(1 for r in guided if r["target_sim"] > 0.15 and r.get("coherence", 0) > 0.3)
        pct = good / len(guided)
        if pct >= 0.7:
            colors.append("#2ecc71")
        elif pct >= 0.5:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(labels)), sims, color=colors, edgecolor="black", linewidth=0.5)

    if baseline_sim is not None:
        ax.axhline(y=baseline_sim, color="blue", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Baseline ({baseline_sim:.3f})")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("Mean target_sim", fontsize=11)
    ax.set_title("Semantic Steering Effectiveness Across 13 Experiments", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(sims) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_topic_heatmap(experiments, output_path):
    """Heatmap: per-topic sim_mean for each experiment."""
    topics = ["space", "ocean", "horror", "cooking"]
    exp_labels = []
    matrix = []

    for label, data in experiments.items():
        if not data["guided"]:
            continue
        row = []
        for t in topics:
            trials = data["agg"].get(t, [])
            if trials:
                row.append(sum(x.get("target_sim", 0) for x in trials) / len(trials))
            else:
                row.append(0.0)
        matrix.append(row)
        exp_labels.append(label)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.5)

    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels([t.capitalize() for t in topics], fontsize=11)
    ax.set_yticks(range(len(exp_labels)))
    ax.set_yticklabels(exp_labels, fontsize=8)

    # Add text annotations
    for i in range(len(exp_labels)):
        for j in range(len(topics)):
            value = matrix[i, j]
            color = "white" if value < 0.2 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Topic Steering Effectiveness (target_sim)", fontsize=13)
    fig.colorbar(im, ax=ax, label="target_sim", shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_alpha_sweep(experiments, output_path):
    """Line chart: effect of alpha on target_sim."""
    alpha_configs = {
        "e00_baseline": (5.0, "constant", 0.1845),
        "e03_cosine_a10": (10.0, "cosine", None),
        "e04_cosine_a15": (15.0, "cosine", None),
    }

    alphas = []
    sims = []
    cohs = []

    for label, (alpha, sched, _) in alpha_configs.items():
        data = experiments.get(label)
        if not data or not data["guided"]:
            continue
        valid = [r for r in data["guided"] if r.get("non_rep", 1.0) > 0.5]
        if valid:
            alphas.append(alpha)
            sims.append(sum(r["target_sim"] for r in valid) / len(valid))
            cohs.append(sum(r.get("coherence", 0) for r in valid) / len(valid))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(alphas, sims, "bo-", linewidth=2, markersize=8, label="target_sim")
    ax1.set_xlabel("Alpha (guidance strength)", fontsize=11)
    ax1.set_ylabel("target_sim", fontsize=11, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(alphas, cohs, "rs--", linewidth=2, markersize=8, label="coherence")
    ax2.set_ylabel("coherence", fontsize=11, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.axvspan(8, 12, alpha=0.15, color="green", label="Effective range")
    ax1.set_title("Alpha vs Steering Effectiveness (cosine schedule)", fontsize=12)

    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("Loading experiments...")
    experiments = load_all_experiments()
    print(f"  Loaded {len(experiments)} experiments")

    print_summary_table(experiments)

    print("\nGenerating visualizations...")
    plot_sim_by_experiment(experiments, OUTPUT_DIR / "experiment_comparison.png")
    plot_topic_heatmap(experiments, OUTPUT_DIR / "topic_heatmap.png")
    plot_alpha_sweep(experiments, OUTPUT_DIR / "alpha_sweep.png")

    print("\nDone. Outputs in notebooks/ directory.")


if __name__ == "__main__":
    main()
