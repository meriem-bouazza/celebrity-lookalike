"""Top-1 / Top-5 retrieval on identity-disjoint test images (cosine similarity).

Every test photo is the query once; all other test photos are the gallery
(docs/decisions.md, "2026-10-31: Experiment setup").
"""

import numpy as np


def top_k(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Indices (N, k) of the k most similar *other* images per image (cosine similarity)."""
    similarity = embeddings @ embeddings.T  # cosine, because rows have norm 1
    np.fill_diagonal(similarity, -np.inf)  # an image must not find itself
    return np.argsort(-similarity, axis=1, kind="stable")[:, :k]


def top_k_accuracy(embeddings: np.ndarray, labels, ks: tuple[int, ...] = (1, 5)) -> dict[int, float]:
    """Share of queries whose k most similar other photos include the same person.

    Args:
        embeddings: Array (N, D), one row per test photo. Rows are L2-normalised here.
        labels: N person ids, one per row.
        ks: Which Top-k values to compute.

    Returns:
        ``{k: accuracy}`` with accuracies between 0 and 1, e.g. ``{1: 0.62, 5: 0.81}``.
    """
    labels = np.asarray(labels)
    if len(labels) != len(embeddings):
        raise ValueError(f"{len(embeddings)} embeddings but {len(labels)} labels")
    # A person with one photo can never be found, which would lower the score for no reason.
    _, counts = np.unique(labels, return_counts=True)
    if (counts < 2).any():
        raise ValueError("every person needs at least 2 photos")

    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    neighbours = top_k(normed, max(ks))
    hits = labels[neighbours] == labels[:, None]
    return {k: float(hits[:, :k].any(axis=1).mean()) for k in ks}
