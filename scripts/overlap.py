"""Which CelebA training people are probably also in the LFW demo gallery?

If a gallery person was also a training person, the demo looks better than it is.
CelebA has no names, so we compare faces: one mean embedding (centroid) per person with the
ready-made face model. A pair is a likely match if the cosine similarity of the centroids is at
least ``--threshold``. The script first measures on LFW alone how well this threshold works.

Usage (in the container; the LFW gallery must exist, see README "Run"):
    PYTHONPATH=src python scripts/overlap.py --device cuda
"""

import argparse
from pathlib import Path

import numpy as np

from whoisit.data.dataset import load_train_people, pick_train_photos
from whoisit.data.preprocess import preprocess
from whoisit.data.splits import load_identities
from whoisit.gallery import build_gallery

PHOTOS_PER_PERSON = 5  # same as the gallery (MAX_PHOTOS), so both centroids are equally reliable


def centroids(embeddings: np.ndarray, identities) -> tuple[np.ndarray, np.ndarray]:
    """One unit-length mean embedding per person. Returns ``(names, centroids)``."""
    names, index = np.unique(identities, return_inverse=True)
    sums = np.zeros((len(names), embeddings.shape[1]))
    np.add.at(sums, index, embeddings)
    return names, sums / np.linalg.norm(sums, axis=1, keepdims=True)


def calibrate(embeddings: np.ndarray, identities, threshold: float) -> tuple[int, int, float]:
    """How the threshold behaves on one dataset with names.

    Returns ``(pairs, wrong, found)``: number of pairs of different names, how many of them reach
    the threshold (false matches, or one person under two names), and the share of people with
    4+ photos whose two halves reach it (same person found).
    """
    identities = np.asarray(identities)
    names, c = centroids(embeddings, identities)
    upper = np.triu_indices(len(names), 1)
    wrong = int(((c @ c.T)[upper] >= threshold).sum())
    halves = []
    for name in names:
        rows = embeddings[identities == name]
        if len(rows) >= 4:
            a, b = rows[:len(rows) // 2].mean(0), rows[len(rows) // 2:].mean(0)
            halves.append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    found = float((np.array(halves) >= threshold).mean()) if halves else float("nan")
    return len(upper[0]), wrong, found


def likely_matches(emb_a, ids_a, emb_b, ids_b, threshold: float) -> list[tuple[str, str, float]]:
    """Pairs (person in a, person in b, similarity) with centroid similarity >= threshold, best first."""
    names_a, ca = centroids(emb_a, ids_a)
    names_b, cb = centroids(emb_b, ids_b)
    similarity = ca @ cb.T
    rows, cols = np.nonzero(similarity >= threshold)
    pairs = [(str(names_a[i]), str(names_b[j]), float(similarity[i, j])) for i, j in zip(rows, cols)]
    return sorted(pairs, key=lambda p: -p[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Likely overlap between CelebA training people and the LFW gallery.")
    parser.add_argument("--gallery", type=Path, default=Path("data/gallery/pretrained.npz"))
    parser.add_argument("--image-root", type=Path, default=Path("data/celeba/img_align_celeba"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/aligned/celeba_train5"))
    parser.add_argument("--celeba-gallery", type=Path, default=Path("data/gallery/celeba_train5.npz"))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=Path("results/overlap_lfw_celeba.txt"))
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    # CelebA training people, 5 photos each -> aligned -> embeddings with the ready-made model
    photos = pick_train_photos(load_identities(), load_train_people(), PHOTOS_PER_PERSON)
    args.aligned_dir.mkdir(parents=True, exist_ok=True)
    photos.rename(columns={"image": "image_path", "person": "identity"}).to_csv(
        args.aligned_dir / "input.csv", index=False)
    preprocess(args.aligned_dir / "input.csv", args.aligned_dir, args.image_root, device=args.device)
    build_gallery(args.aligned_dir, args.celeba_gallery, args.device)

    lfw, celeba = np.load(args.gallery), np.load(args.celeba_gallery)
    pairs, wrong, found = calibrate(lfw["embeddings"], lfw["identities"], args.threshold)
    matches = likely_matches(celeba["embeddings"], celeba["identities"],
                             lfw["embeddings"], lfw["identities"], args.threshold)
    n_celeba = len(set(celeba["identities"]))
    lines = [
        f"threshold: cosine similarity of the person centroids >= {args.threshold}",
        f"check on LFW: {wrong} of {pairs} pairs of different names reach it, "
        f"{found:.1%} of people with 4+ photos are found again (two halves)",
        f"CelebA training people: {n_celeba} ({PHOTOS_PER_PERSON} photos each), LFW gallery people: "
        f"{len(set(lfw['identities']))}",
        f"likely matches: {len(matches)} pairs, {len({m[0] for m in matches})} CelebA people",
        "",
        "celeba_id,lfw_name,similarity",
        *[f"{a},{b},{s:.3f}" for a, b, s in matches],
    ]
    report = "\n".join(lines)
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
