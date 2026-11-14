"""Plot Top 1 and Top 5 against N from results/tables/summary.csv, one line per loss.

The line is the mean of the seeds, the band goes from the lowest to the highest seed.
The baseline without training is a dashed horizontal line.

Example: python -m whoisit.plot
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window, only the file
import matplotlib.pyplot as plt  # noqa: E402

SUMMARY_FILE = Path("results/tables/summary.csv")
BASELINE_FILE = Path("results/baseline_imagenet.txt")
OUT_DIR = Path("results/figures")
# fixed color per loss, so a loss keeps its color when the other one is missing
LOSSES = {"ce": ("Cross-entropy", "#2a78d6"), "arcface": ("ArcFace", "#eb6834")}
METRICS = {"top1": "Top 1", "top5": "Top 5"}
GRAY, TEXT, MUTED = "#8a8985", "#0b0b0b", "#52514e"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY_FILE)
    parser.add_argument("--baseline", type=Path, default=BASELINE_FILE)
    parser.add_argument("--split", default="val", help="val or test, the rows of summary.csv to plot")
    parser.add_argument("--out", type=Path, help="default: results/figures/results_<split>.png")
    return parser.parse_args(argv)


def load_summary(path, split):
    with path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == split]
    if not rows:
        raise ValueError(f"no rows with split {split} in {path}")
    return rows


def load_baseline(path):
    # lines like "top 1: 12.22%" in the file written by scripts/baseline.py
    text = path.read_text(encoding="utf-8")
    return {f"top{k}": float(re.search(rf"top {k}: ([\d.]+)%", text).group(1)) for k in (1, 5)}


def plot(rows, baseline, split, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    ns = sorted({int(row["n"]) for row in rows})
    for ax, (metric, title) in zip(axes, METRICS.items()):
        for loss, (label, color) in LOSSES.items():
            points = sorted((int(r["n"]), float(r[f"{metric}_mean"]), float(r[f"{metric}_min"]),
                             float(r[f"{metric}_max"])) for r in rows if r["loss"] == loss)
            if not points:
                continue
            n, mean, low, high = zip(*points)
            ax.fill_between(n, low, high, color=color, alpha=0.2, linewidth=0)
            ax.plot(n, mean, color=color, linewidth=2, marker="o", markersize=6, label=label)
        if baseline:
            ax.axhline(baseline[metric], color=GRAY, linewidth=1.5, linestyle="--",
                       label="No training (ImageNet, test people)")
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlim(min(ns) / 1.2, max(ns) * 1.2)
        ax.set_ylim(0, 100)
        ax.set_title(title, color=TEXT, loc="left")
        ax.set_xlabel("Training photos per person (N)", color=MUTED)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(f"Accuracy on {'validation' if split == 'val' else 'test'} people (%)", color=MUTED)
    # legend in both panels, so each panel can be read alone (e.g. on the poster)
    for ax in axes:
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "Line: mean of the seeds. Band: lowest to highest seed.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    out = args.out or OUT_DIR / f"results_{args.split}.png"
    try:
        rows = load_summary(args.summary, args.split)
    except ValueError as error:
        raise SystemExit(f"error: {error}")
    baseline = load_baseline(args.baseline) if args.baseline.exists() else None
    plot(rows, baseline, args.split, out)
    print("saved", out)


if __name__ == "__main__":
    main()
