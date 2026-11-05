"""Builds the demo gallery: LFW people with 2 or more photos, up to 5 photos each, as embeddings.

Part 1 (fallback): the embeddings come from a ready-made face model (facenet-pytorch
InceptionResnetV1, trained on VGGFace2), so we have a demo even if our own training fails.
Part 2: one photo in, the 3 most similar gallery people out.

Input is the output folder of ``whoisit.data.preprocess`` (aligned 112 x 112 crops + meta.csv).
Output is one ``.npz`` file with ``embeddings`` (N, 512), ``identities`` (N,) and ``image_paths`` (N,).

Usage:
    python -m whoisit.data.preprocess --metadata data/lfw_meta.csv --out-dir data/aligned/lfw
    python -m whoisit.gallery --aligned-dir data/aligned/lfw --out data/gallery/pretrained.npz
    python -m whoisit.gallery --out data/gallery/pretrained.npz --photo me.jpg

The first run downloads the VGGFace2 weights (~110 MB) into the torch cache.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from whoisit.data.align import detect_and_align, load_image
from whoisit.evaluate import top_k_accuracy

MIN_PHOTOS, MAX_PHOTOS = 2, 5  # docs/decisions.md, "2026-10-29: Datasets"
PRETRAINED_SIZE = 160  # InceptionResnetV1 was trained on 160 x 160 crops


def select_gallery(meta_rows: list[dict], min_photos: int = MIN_PHOTOS,
                   max_photos: int = MAX_PHOTOS) -> list[dict]:
    """Keep people with at least ``min_photos`` aligned photos, and their first ``max_photos`` by name."""
    by_identity = defaultdict(list)
    for row in meta_rows:
        by_identity[row["identity"]].append(row)
    chosen = []
    for identity in sorted(by_identity):
        rows = sorted(by_identity[identity], key=lambda r: r["image_path"])
        if len(rows) >= min_photos:
            chosen.extend(rows[:max_photos])
    return chosen


def load_pretrained(device: str = "cpu", pretrained: str | None = "vggface2") -> torch.nn.Module:
    """InceptionResnetV1 that returns L2-normalised 512-d embeddings (None = untrained, for tests)."""
    return InceptionResnetV1(pretrained=pretrained).eval().to(device)


@torch.no_grad()
def embed_pretrained(model: torch.nn.Module, faces: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Turn aligned faces (N, 112, 112, 3) uint8 into embeddings (N, 512) with norm 1."""
    x = torch.from_numpy(faces).permute(0, 3, 1, 2).float().to(device)
    x = torch.nn.functional.interpolate(x, size=PRETRAINED_SIZE, mode="bilinear", align_corners=False)
    return model((x - 127.5) / 128.0).cpu().numpy()  # same scaling as facenet's fixed_image_standardization


def build_gallery(aligned_dir: Path, out: Path, device: str = "cpu", batch_size: int = 128) -> dict:
    """Select the gallery photos from ``aligned_dir/meta.csv``, embed them and save them to ``out``."""
    with open(aligned_dir / "meta.csv", newline="", encoding="utf-8") as f:
        rows = select_gallery(list(csv.DictReader(f)))
    if not rows:
        raise ValueError(f"no person with {MIN_PHOTOS}+ photos in {aligned_dir / 'meta.csv'}")

    model = load_pretrained(device)
    parts = []
    for start in range(0, len(rows), batch_size):
        faces = np.stack([load_image(aligned_dir / r["image_path"]) for r in rows[start:start + batch_size]])
        parts.append(embed_pretrained(model, faces, device))
    embeddings = np.concatenate(parts)
    identities = np.array([r["identity"] for r in rows])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, embeddings=embeddings, identities=identities,
             image_paths=np.array([r["image_path"] for r in rows]))
    # Sanity check: every gallery person has 2+ photos, so the model should find the same person.
    return {"people": len(set(identities)), "photos": len(rows), **top_k_accuracy(embeddings, identities)}


def top_people(query: np.ndarray, embeddings: np.ndarray, identities: np.ndarray,
               k: int = 3) -> list[tuple[str, float]]:
    """The k most similar people to ``query`` (D,), each scored by their best photo (cosine similarity)."""
    similarity = embeddings @ query  # cosine, because all rows have norm 1
    best = {}
    for identity, score in zip(identities, similarity):
        best[identity] = max(best.get(identity, -np.inf), float(score))
    return sorted(best.items(), key=lambda item: -item[1])[:k]


def look_up(photo: Path, gallery: Path, device: str = "cpu", k: int = 3) -> list[tuple[str, float]]:
    """Detect and align the face in ``photo`` and return its k most similar gallery people."""
    face = detect_and_align(load_image(photo), device=device)
    if face is None:
        raise ValueError(f"no face found in {photo}")
    data = np.load(gallery)
    query = embed_pretrained(load_pretrained(device), face[None], device)[0]
    return top_people(query, data["embeddings"], data["identities"], k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the demo gallery with the ready-made face model.")
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/aligned/lfw"), help="output of preprocess")
    parser.add_argument("--out", type=Path, default=Path("data/gallery/pretrained.npz"))
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--photo", type=Path, help="look up this photo in the gallery --out instead of building it")
    args = parser.parse_args()

    if args.photo:
        for rank, (identity, score) in enumerate(look_up(args.photo, args.out, args.device), 1):
            print(f"{rank}. {identity.replace('_', ' ')}  (similarity {score:.2f})")
        return

    stats = build_gallery(args.aligned_dir, args.out, args.device)
    print(f"gallery: {stats['people']} people, {stats['photos']} photos -> {args.out}")
    print(f"top 1: {stats[1]:.2%}  top 5: {stats[5]:.2%}  (each photo against the rest of the gallery)")


if __name__ == "__main__":
    main()
