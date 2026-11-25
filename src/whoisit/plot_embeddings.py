"""Picture of the embedding space: photos of the same person should be close together.

A few random test people, one panel per model, one color per person. t-SNE makes the
embeddings 2D. The printed silhouette score uses the real embeddings, not the 2D picture.

Example (in the container): python -m whoisit.plot_embeddings
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from whoisit.data.images import FaceDataset
from whoisit.error_analysis import embed
from whoisit.final_eval import ALIGNED_DIR, CHECKPOINT, load_model, load_test_photos
from whoisit.models.backbone import Backbone
# importing plot.py also sets the Agg backend (no window, only the file)
from whoisit.plot import LOSSES, MUTED, OUT_DIR, TEXT
from whoisit.summarize import RESULT_FILE, RUNS_DIR

RUNS = [RUNS_DIR / "ce_n20_seed42_ep20", RUNS_DIR / "arcface_n20_seed42_ep20"]
OUT_FILE = OUT_DIR / "embeddings_tsne.png"
PEOPLE, PHOTOS = 20, 10
PERPLEXITY = 15
TSNE_SEED = 42  # fixed, --seed only changes the people
# 20 colors are hard to tell apart, so neighbours in the color list also get another shape
MARKERS = ["o", "s", "^", "D"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", default=RUNS, help="run folders, one panel each")
    parser.add_argument("--aligned-dir", type=Path, default=ALIGNED_DIR)
    parser.add_argument("--seed", type=int, default=42, help="picks the people")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    return parser.parse_args(argv)


def pick_photos(photos, seed):
    # only people with enough aligned photos; sorted, so the seed always gives the same people
    counts = photos.groupby("person").size()
    people = sorted(counts[counts >= PHOTOS].index)
    if len(people) < PEOPLE:
        raise ValueError(f"only {len(people)} test people have {PHOTOS} aligned photos, need {PEOPLE}")
    chosen = np.random.default_rng(seed).choice(people, PEOPLE, replace=False)
    picked = photos[photos["person"].isin(chosen)].sort_values(["person", "image"])
    return picked.groupby("person").head(PHOTOS).reset_index(drop=True)


def load_imagenet(device):
    # the same model as scripts/baseline.py: ResNet-18 from ImageNet without the last layer
    return Backbone(pretrained=True).resnet.to(device)


def load_titles(folders):
    # check all runs before the first model is loaded
    titles = []
    for folder in folders:
        for path in [folder / RESULT_FILE, folder / CHECKPOINT]:
            if not path.exists():
                raise ValueError(f"{path} not found")
        options = json.loads((folder / RESULT_FILE).read_text(encoding="utf-8"))["options"]
        titles.append(f"{LOSSES[options['loss']][0]}, N = {options['n']}")
    # two runs with the same loss and n: add the folder, so the panels can be told apart
    return [f"{title}\n{folder.parent.name}/{folder.name}" if titles.count(title) > 1 else title
            for title, folder in zip(titles, folders)]


def normalize(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def separation(embeddings, labels):
    # 1 = every person is a clear group, 0 = groups overlap, below 0 = closer to other people
    return float(silhouette_score(embeddings, labels, metric="cosine"))


def to_2d(embeddings):
    return TSNE(n_components=2, metric="cosine", perplexity=PERPLEXITY, init="pca",
                random_state=TSNE_SEED).fit_transform(embeddings)


def plot(panels, labels, out):
    # panels: list of (title, 2D points); the labels are the same in every panel
    labels = np.asarray(labels)
    colors = plt.get_cmap("tab20").colors
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.5), squeeze=False)
    for ax, (title, points) in zip(axes[0], panels):
        for person in np.unique(labels):
            mask = labels == person
            ax.scatter(points[mask, 0], points[mask, 1], s=32, color=colors[person % len(colors)],
                       marker=MARKERS[person % len(MARKERS)], edgecolors="white", linewidths=0.5)
        ax.set_title(title, color=TEXT, loc="left", fontsize=11)
        # the t-SNE axes have no meaning, so no numbers
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#e6e5e0")
    fig.text(0.01, 0.01, f"{PEOPLE} test people, {PHOTOS} photos each. One color per person. "
             "t-SNE of the embeddings.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    meta = args.aligned_dir / "meta.csv"
    if not meta.exists():
        raise SystemExit(f"error: {meta} not found")
    try:
        titles = load_titles(args.runs)
        photos = pick_photos(load_test_photos(args.aligned_dir), args.seed)
    except ValueError as error:
        raise SystemExit(f"error: {error}")

    data = FaceDataset(photos, image_dir=args.aligned_dir / "images", train=False)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    print(f"picked: {data.num_people} people, {len(data)} photos, device {args.device}")

    models = [("No training (ImageNet)", "imagenet", lambda: load_imagenet(args.device))]
    models += [(title, folder.name, lambda folder=folder: load_model(folder / CHECKPOINT, args.device))
               for title, folder in zip(titles, args.runs)]
    panels = []
    for title, name, load in models:
        embeddings = normalize(embed(load(), loader, args.device))
        print(f"{name}: silhouette score {separation(embeddings, data.labels):.3f}")
        panels.append((title, to_2d(embeddings)))

    plot(panels, data.labels, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
