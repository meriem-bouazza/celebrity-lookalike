"""Baseline without training: ImageNet ResNet-18 embeddings on the CelebA test people.

This is the lower bound for our results: a network that has never been trained on faces.
Steps: align the test photos (``whoisit.data.preprocess``, resumable), embed them with
ResNet-18 without its last layer, and measure Top 1 and Top 5 (``whoisit.evaluate``).

Usage (in the container, about 4 minutes on the GPU for 10000 photos):
    python scripts/baseline.py --device cuda
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from smoke_test import build_model, embed
from whoisit.data.align import load_image
from whoisit.data.preprocess import preprocess
from whoisit.evaluate import top_k_accuracy


def run_baseline(photos: Path, image_root: Path, aligned_dir: Path, device: str = "cpu",
                 pretrained: bool = True, batch_size: int = 256) -> dict:
    """Align and embed the photos in ``photos`` (columns image,person) and return the Top-k accuracy."""
    # preprocess wants the columns image_path,identity
    aligned_dir.mkdir(parents=True, exist_ok=True)
    metadata = aligned_dir / "input.csv"
    with open(photos, newline="", encoding="utf-8") as f_in, \
            open(metadata, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["image_path", "identity"])
        writer.writerows((row["image"], row["person"]) for row in csv.DictReader(f_in))
    stats = preprocess(metadata, aligned_dir, image_root, device=device)

    with open(aligned_dir / "meta.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    identities = np.array([r["identity"] for r in rows])
    # A person whose other photos all failed detection can never be found: leave them out.
    names, counts = np.unique(identities, return_counts=True)
    keep = np.isin(identities, names[counts >= 2])
    rows, identities = [r for r, k in zip(rows, keep) if k], identities[keep]

    model = build_model(pretrained)
    parts = []
    for start in range(0, len(rows), batch_size):
        faces = np.stack([load_image(aligned_dir / r["image_path"]) for r in rows[start:start + batch_size]])
        parts.append(embed(model, faces))
    accuracy = top_k_accuracy(np.concatenate(parts), identities)
    return {"photos": stats["total"], "aligned": stats["aligned"], "people": len(set(identities)),
            "used": len(rows), "top1": accuracy[1], "top5": accuracy[5]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline without training (ImageNet ResNet-18).")
    parser.add_argument("--photos", type=Path, default=Path("splits/test_photos.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/celeba/img_align_celeba"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/aligned/celeba_test"))
    parser.add_argument("--out", type=Path, default=Path("results/baseline_imagenet.txt"))
    parser.add_argument("--device", default="cpu", help="cpu or cuda (only for face detection)")
    args = parser.parse_args()

    r = run_baseline(args.photos, args.image_root, args.aligned_dir, args.device)
    report = "\n".join([
        "model: ResNet-18, ImageNet weights, no training on faces",
        f"test photos: {r['photos']}, faces found: {r['aligned']}",
        f"used: {r['used']} photos of {r['people']} people (people with 2+ found faces)",
        f"top 1: {r['top1']:.2%}",
        f"top 5: {r['top5']:.2%}",
    ])
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
