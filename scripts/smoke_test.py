"""End-to-end smoke test: align a few faces, embed them with ImageNet ResNet-18, rank by cosine.

No training is involved. The test only checks that the pipeline pieces fit together:
    raw photo -> detect_and_align (112 x 112) -> ResNet-18 without fc (512-d) -> L2 norm
    -> cosine similarity -> top-k neighbours per image.

Checks: embedding shape (N_identities * N_images, 512), every norm 1.0 +- 1e-5, and two
complete runs (alignment + embedding) give identical embeddings.

Usage:
    python scripts/smoke_test.py --metadata data/lfw_meta.csv

The first run downloads the torchvision ImageNet weights for ResNet-18 (~45 MB) into the
torch cache (not into the repository).
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

from whoisit.data.align import detect_and_align, load_image

# torchvision's ImageNet normalisation (same values as in ResNet18_Weights.IMAGENET1K_V1).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def select_images(
    metadata: Path, image_root: Path | None, n_identities: int, n_images: int, seed: int
) -> list[tuple[str, Path]]:
    """Pick ``n_identities`` random identities with at least ``n_images`` photos each.

    Args:
        metadata: CSV with columns ``image_path,identity``.
        image_root: Base folder for relative paths (None = paths as given).
        n_identities, n_images: How many identities and photos per identity.
        seed: Seed for the choice of identities, so every run picks the same images.

    Returns:
        List of ``(identity, path)``, grouped by identity; per identity the candidate photos
        in file order (more than ``n_images`` if available, as backup for failed detections).
    """
    by_identity: dict[str, list[Path]] = defaultdict(list)
    with open(metadata, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = Path(row["image_path"]) if image_root is None else image_root / row["image_path"]
            by_identity[row["identity"]].append(path)

    candidates = sorted(name for name, paths in by_identity.items() if len(paths) >= n_images)
    if len(candidates) < n_identities:
        raise ValueError(f"only {len(candidates)} identities have >= {n_images} images")
    chosen = random.Random(seed).sample(candidates, n_identities)
    return [(name, path) for name in chosen for path in sorted(by_identity[name])]


def align_images(
    candidates: list[tuple[str, Path]], n_images: int
) -> tuple[list[str], list[str], np.ndarray]:
    """Align the first ``n_images`` photos per identity in which a face is found.

    Returns:
        ``(identities, file_names, faces)`` with faces of shape (N, 112, 112, 3) uint8.
    """
    identities, names, faces = [], [], []
    count: dict[str, int] = defaultdict(int)
    for identity, path in candidates:
        if count[identity] == n_images:
            continue
        face = detect_and_align(load_image(path))
        if face is None:
            print(f"  no face found, skipped: {path.name}")
            continue
        identities.append(identity)
        names.append(path.name)
        faces.append(face)
        count[identity] += 1
    missing = [name for name, c in count.items() if c < n_images]
    if missing:
        raise RuntimeError(f"not enough detectable faces for: {missing}")
    return identities, names, np.stack(faces)


def build_model(pretrained: bool = True) -> torch.nn.Module:
    """ResNet-18 with ImageNet weights and the classification layer removed (output 512-d)."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = torch.nn.Identity()
    return model.eval()


@torch.no_grad()
def embed(model: torch.nn.Module, faces: np.ndarray) -> np.ndarray:
    """Turn aligned faces (N, 112, 112, 3) uint8 into L2-normalised embeddings (N, 512).

    The 112 x 112 crops are fed directly; ResNet's adaptive average pooling handles the
    smaller input size (no resizing to 224).
    """
    x = torch.from_numpy(faces).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    features = model((x - mean) / std)
    return torch.nn.functional.normalize(features, dim=1).numpy()


def top_k(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Indices (N, k) of the k most similar *other* images per image (cosine similarity)."""
    similarity = embeddings @ embeddings.T  # cosine, because rows have norm 1
    np.fill_diagonal(similarity, -np.inf)  # an image must not find itself
    return np.argsort(-similarity, axis=1, kind="stable")[:, :k]


def run_once(
    metadata: Path, image_root: Path | None, n_identities: int, n_images: int, seed: int,
    pretrained: bool = True,
) -> tuple[list[str], list[str], np.ndarray]:
    """One full pass: select, align and embed. Returns ``(identities, names, embeddings)``."""
    candidates = select_images(metadata, image_root, n_identities, n_images, seed)
    identities, names, faces = align_images(candidates, n_images)
    torch.manual_seed(seed)  # only matters for pretrained=False (random initial weights)
    return identities, names, embed(build_model(pretrained), faces)


def check(embeddings: np.ndarray, embeddings_again: np.ndarray, expected_rows: int) -> list[str]:
    """Return a list of failed checks (empty list = all good)."""
    errors = []
    if embeddings.shape != (expected_rows, 512):
        errors.append(f"shape {embeddings.shape}, expected ({expected_rows}, 512)")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        errors.append(f"norms not 1.0 +- 1e-5 (min {norms.min():.6f}, max {norms.max():.6f})")
    if not np.array_equal(embeddings, embeddings_again):
        errors.append(f"second run differs (max abs diff {np.abs(embeddings - embeddings_again).max():.2e})")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test: align, embed (ImageNet ResNet-18), top-k.")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with image_path,identity (raw photos)")
    parser.add_argument("--image-root", type=Path, default=None, help="base folder for relative image paths")
    parser.add_argument("--identities", type=int, default=5)
    parser.add_argument("--images", type=int, default=4, help="images per identity")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)  # avoids thread-dependent summation order on CPU
    run_args = (args.metadata, args.image_root, args.identities, args.images, args.seed)
    print("run 1 ...")
    identities, names, embeddings = run_once(*run_args)
    print("run 2 ...")
    _, _, embeddings_again = run_once(*run_args)

    print(f"\ntop-{args.top} per image (* = same identity):")
    for i, neighbours in enumerate(top_k(embeddings, args.top)):
        hits = [f"{'*' if identities[j] == identities[i] else ' '}{names[j]} ({embeddings[i] @ embeddings[j]:.3f})"
                for j in neighbours]
        print(f"  {names[i]:<32} -> " + "  ".join(hits))

    errors = check(embeddings, embeddings_again, args.identities * args.images)
    print(f"\nshape {embeddings.shape}, norms in "
          f"[{np.linalg.norm(embeddings, axis=1).min():.6f}, {np.linalg.norm(embeddings, axis=1).max():.6f}]")
    if errors:
        print("FAILED:\n  " + "\n  ".join(errors))
        return 1
    print("all checks passed: shape, norms, two runs identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
