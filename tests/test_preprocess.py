"""Tests for the batch preprocessing script (task 4.1), on a few LFW images."""

import csv

import numpy as np
from PIL import Image

from whoisit.data.preprocess import preprocess


def _write_metadata(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "identity"])
        writer.writerows(rows)


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _make_inputs(tmp_path, lfw_people):
    """3 real faces of 2 persons + 1 image without a face, with paths relative to tmp_path/src."""
    src = tmp_path / "src"
    rows = []
    name_a, name_b = list(lfw_people)[:2]
    for name, image in [(name_a, lfw_people[name_a][0]), (name_a, lfw_people[name_a][1]),
                        (name_b, lfw_people[name_b][0])]:
            target = src / name / image.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(image.read_bytes())
            rows.append((f"{name}/{image.name}", name))
    gradient = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    Image.fromarray(gradient).convert("RGB").save(src / "no_face.jpg")
    rows.append(("no_face.jpg", "nobody"))
    return src, rows


def test_preprocess_writes_images_meta_and_failed(tmp_path, lfw_people):
    src, rows = _make_inputs(tmp_path, lfw_people)
    _write_metadata(tmp_path / "in.csv", rows)
    out = tmp_path / "out"

    stats = preprocess(tmp_path / "in.csv", out, image_root=src)

    assert stats == {"total": 4, "aligned": 3, "failed": 1, "new": 4, "detection_rate": 0.75}
    meta = _read(out / "meta.csv")
    assert [m["source_path"] for m in meta] == [r[0] for r in rows[:3]]
    for m in meta:
        image = Image.open(out / m["image_path"])
        assert image.size == (112, 112)
        assert 0.9 <= float(m["confidence"]) <= 1.0
    failed = _read(out / "failed.csv")
    assert failed == [{"source_path": "no_face.jpg", "identity": "nobody", "reason": "no_face"}]


def test_preprocess_resumes_without_redoing_work(tmp_path, lfw_people):
    src, rows = _make_inputs(tmp_path, lfw_people)
    out = tmp_path / "out"

    # First run sees only 2 images (simulates an interrupted run), second run sees all.
    _write_metadata(tmp_path / "in.csv", rows[:2])
    preprocess(tmp_path / "in.csv", out, image_root=src)
    _write_metadata(tmp_path / "in.csv", rows)
    stats = preprocess(tmp_path / "in.csv", out, image_root=src)

    assert stats["new"] == 2
    assert stats["total"] == 4
    assert len(_read(out / "meta.csv")) == 3  # no duplicate rows


def test_unreadable_file_is_recorded(tmp_path):
    _write_metadata(tmp_path / "in.csv", [("does_not_exist.jpg", "x")])
    stats = preprocess(tmp_path / "in.csv", tmp_path / "out", image_root=tmp_path)
    assert stats["failed"] == 1
    assert _read(tmp_path / "out" / "failed.csv")[0]["reason"].startswith("unreadable")
