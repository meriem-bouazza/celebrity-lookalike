"""Plot validation Top 1 against the epoch, one panel per N, one line per loss.

It shows why ArcFace gets its own settings: it reaches its best epoch earlier and then
gets worse. The numbers come from the "epochs" list in each result.json.

Example: python -m whoisit.plot_curves
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# same colors and names as the results plot; importing plot.py also sets the Agg backend
from whoisit.plot import LOSSES, MUTED, OUT_DIR, TEXT
from whoisit.summarize import RESULT_FILE, RUNS_DIR, check_same, load_runs, settings

OUT_FILE = OUT_DIR / "val_curves.png"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--n", type=int, nargs="+", default=[2, 20], help="one panel per N")
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    return parser.parse_args(argv)


def make_line(runs, loss, n):
    group = {name: run for name, run in runs.items()
             if run["options"]["loss"] == loss and run["options"]["n"] == n}
    if not group:
        raise ValueError(f"no runs with loss {loss} and n {n}")
    lengths = {name: len(run["epochs"]) for name, run in group.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"runs with loss {loss} and n {n} have a different number of epochs: "
                         + ", ".join(f"{name} has {count}" for name, count in lengths.items()))
    # also the other settings must be the same, like in summary.csv
    check_same(group, settings, f"runs with loss {loss} and n {n} differ in more than the seed")

    top1 = np.array([[row["top1"] * 100 for row in run["epochs"]] for run in group.values()])
    mean = top1.mean(axis=0)
    # argmax takes the first epoch if two are equal
    best = int(np.argmax(mean))
    return {"epochs": np.arange(1, top1.shape[1] + 1), "mean": mean, "low": top1.min(axis=0),
            "high": top1.max(axis=0), "seeds": len(group), "best_epoch": best + 1,
            "best_top1": float(mean[best])}


def plot(lines, ns, out):
    fig, axes = plt.subplots(1, len(ns), figsize=(4.5 * len(ns), 4), squeeze=False)
    for ax, n in zip(axes[0], ns):
        for loss, (label, color) in LOSSES.items():
            line = lines[loss, n]
            ax.fill_between(line["epochs"], line["low"], line["high"], color=color, alpha=0.2,
                            linewidth=0)
            ax.plot(line["epochs"], line["mean"], color=color, linewidth=2,
                    label=f"{label} (best: epoch {line['best_epoch']})")
            # small dot on the best epoch of the mean line
            ax.plot(line["best_epoch"], line["best_top1"], "o", color=color, markersize=6)
        ax.set_title(f"N = {n}", color=TEXT, loc="left")
        ax.set_xlabel("Epoch", color=MUTED)
        # a bit of room at both ends, so a dot on the first or last epoch is not cut
        ax.set_xlim(0.5, max(lines[loss, n]["epochs"][-1] for loss in LOSSES) + 0.5)
        # only whole epochs on the axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    axes[0][0].set_ylabel("Top 1 on validation people (%)", color=MUTED)

    seeds = {line["seeds"] for line in lines.values()}
    count = f"{seeds.pop()} seeds" if len(seeds) == 1 else "the seeds"
    fig.text(0.01, 0.01, f"Validation people. Line: mean of {count}. Band: lowest to highest seed.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"error: no {RESULT_FILE} found in {args.runs_dir}")
    try:
        lines = {(loss, n): make_line(runs, loss, n) for n in args.n for loss in LOSSES}
    except ValueError as error:
        raise SystemExit(f"error: {error}")

    plot(lines, args.n, args.out)
    for (loss, n), line in lines.items():
        print(f"{loss} n={n}: best epoch {line['best_epoch']} ({line['best_top1']:.2f} %), "
              f"last epoch {line['epochs'][-1]} ({line['mean'][-1]:.2f} %)")
    print("saved", args.out)


if __name__ == "__main__":
    main()
