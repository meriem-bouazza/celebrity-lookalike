"""Test for app/app.py, offline (untrained face model, tiny fake gallery)."""

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import app  # noqa: E402

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
    np.savez(tmp_path / "pretrained.npz", embeddings=embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True),
             identities=np.array(["Ann_A", "Ann_A", "Bob_B", "Cy_C"]), image_paths=np.array(paths))
    monkeypatch.setitem(app.MODELS, "pretrained", (lambda device: load_pretrained(device, None), embed_pretrained))
    return app.make_match(gallery_dir=tmp_path, aligned_dir=tmp_path)


def test_match_returns_three_different_people(match, lfw_people):
    result = match(str(next(iter(lfw_people.values()))[0]))
    captions = [caption for _, caption in result]
    assert len(result) == 3
    assert {c.split(" (")[0] for c in captions} == {"Ann A", "Bob B", "Cy C"}
    assert all(image.shape == (112, 112, 3) for image, _ in result)


def test_gallery_and_aligned_dir_default_to_lfw():
    args = app.parse_args([])
    assert (args.gallery_dir, args.aligned_dir) == (Path("data/gallery"), Path("data/aligned/lfw"))
    args = app.parse_args(["--gallery-dir", "data/gallery/celeb", "--aligned-dir", "data/aligned/celeb"])
    assert (args.gallery_dir, args.aligned_dir) == (Path("data/gallery/celeb"), Path("data/aligned/celeb"))


def test_main_stops_when_gallery_folder_has_no_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py", "--gallery-dir", str(tmp_path / "celeb")])
    with pytest.raises(SystemExit, match="pretrained.npz"):
        app.main()


def test_match_explains_missing_face(match, tmp_path):
    cv2.imwrite(str(tmp_path / "empty.jpg"), np.zeros((200, 200, 3), dtype=np.uint8))
    with pytest.raises(gr.Error):
        match(str(tmp_path / "empty.jpg"))
    with pytest.raises(gr.Error):
        match(None)


def test_match_deletes_previous_photo_when_next_arrives(match, lfw_people, tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(app, "UPLOAD_DIR", upload_dir)
    first, second = upload_dir / "first.jpg", upload_dir / "second.jpg"
    for photo in (first, second):
        photo.write_bytes(Path(next(iter(lfw_people.values()))[0]).read_bytes())
    match(str(first))
    assert first.exists()  # kept while shown, so a model switch can send it again
    assert len(match(str(second))) == 3
    assert not first.exists()
    assert second.exists()


def test_match_keeps_files_outside_upload_folder(match, lfw_people, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "UPLOAD_DIR", tmp_path / "uploads")
    photo = Path(next(iter(lfw_people.values()))[0])
    match(str(photo))
    assert photo.exists()


def test_examples_are_lfw_people_outside_the_gallery(tmp_path):
    photos = {"A_A": 1, "B_B": 2, "C_C": 1, "D_D": 1}
    rows = ["image_path,identity"]
    for name, count in photos.items():
        for i in range(count):
            path = tmp_path / f"{name}_{i}.jpg"
            path.touch()
            rows.append(f"{path},{name}")
    (tmp_path / "meta.csv").write_text("\n".join(rows))
    examples = app.example_photos(tmp_path / "meta.csv", k=2)
    assert examples == [str(tmp_path / "A_A_0.jpg"), str(tmp_path / "C_C_0.jpg")]
