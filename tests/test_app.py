"""Test for app/app.py, offline (untrained face model, tiny fake gallery)."""

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import app  # noqa: E402

from whoisit.data.align import load_image  # noqa: E402
from whoisit.gallery import embed_pretrained, load_pretrained  # noqa: E402


@pytest.fixture
def match(tmp_path, monkeypatch):
    # 4 gallery photos of 3 people; embeddings only need norm 1
    rng = np.random.default_rng(0)
    paths = [f"images/p{i}.jpg" for i in range(4)]
    for path in paths:
        (tmp_path / path).parent.mkdir(exist_ok=True)
        cv2.imwrite(str(tmp_path / path), rng.integers(0, 256, (112, 112, 3), dtype=np.uint8))
    embeddings = rng.normal(size=(4, 512))
    np.savez(tmp_path / "gallery.npz", embeddings=embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True),
             identities=np.array(["Ann_A", "Ann_A", "Bob_B", "Cy_C"]), image_paths=np.array(paths))
    monkeypatch.setitem(app.MODELS, "pretrained",
                        (lambda device: load_pretrained(device, None), embed_pretrained, tmp_path / "gallery.npz"))
    return app.make_match(aligned_dir=tmp_path)


def test_match_returns_three_different_people(match, lfw_people):
    photo = load_image(next(iter(lfw_people.values()))[0])
    result = match(photo)
    captions = [caption for _, caption in result]
    assert len(result) == 3
    assert {c.split(" (")[0] for c in captions} == {"Ann A", "Bob B", "Cy C"}
    assert all(image.shape == (112, 112, 3) for image, _ in result)


def test_match_explains_missing_face(match):
    with pytest.raises(gr.Error):
        match(np.zeros((200, 200, 3), dtype=np.uint8))
    with pytest.raises(gr.Error):
        match(None)
