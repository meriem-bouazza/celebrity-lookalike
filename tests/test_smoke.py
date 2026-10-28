"""Tests for scripts/smoke_test.py, offline (untrained ResNet-18, no weight download)."""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from smoke_test import check, run_once, top_k  # noqa: E402


def test_top_k_excludes_the_image_itself():
    embeddings = np.eye(3, dtype=np.float32)
    embeddings[1] = embeddings[0]  # image 1 is a copy of image 0
    neighbours = top_k(embeddings, k=1)
    assert neighbours[0, 0] == 1 and neighbours[1, 0] == 0


def test_check_reports_wrong_norm_and_differences():
    good = np.eye(2, 512, dtype=np.float32)
    assert check(good, good.copy(), expected_rows=2) == []
    assert len(check(good * 2, good, expected_rows=2)) == 2  # norm wrong + runs differ


def test_pipeline_on_lfw(tmp_path, lfw_people):
    # 2 identities x 3 images keeps the test fast; the real smoke test uses 5 x 4.
    metadata = tmp_path / "meta.csv"
    with open(metadata, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "identity"])
        for name in list(lfw_people)[:3]:
            writer.writerows((str(path), name) for path in lfw_people[name][:4])

    identities, _, embeddings = run_once(metadata, None, n_identities=2, n_images=3, seed=0,
                                         pretrained=False)
    _, _, again = run_once(metadata, None, n_identities=2, n_images=3, seed=0, pretrained=False)

    assert len(set(identities)) == 2
    assert check(embeddings, again, expected_rows=6) == []
