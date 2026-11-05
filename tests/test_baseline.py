"""Test for scripts/baseline.py, offline (untrained ResNet-18) on a few LFW photos."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline import run_baseline  # noqa: E402


def test_baseline_on_lfw(tmp_path, lfw_dir, lfw_people):
    # image,person like splits/test_photos.csv, plus one person with a single photo
    names = list(lfw_people)[:3]
    photos = tmp_path / "photos.csv"
    with open(photos, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "person"])
        for name in names[:2]:
            writer.writerows((p.relative_to(lfw_dir).as_posix(), name) for p in lfw_people[name][:3])
        writer.writerow((lfw_people[names[2]][0].relative_to(lfw_dir).as_posix(), names[2]))

    result = run_baseline(photos, lfw_dir, tmp_path / "aligned", pretrained=False)

    assert result["photos"] == 7
    assert result["people"] == 2  # the person with one photo is left out
    assert result["used"] == result["aligned"] - 1
    assert 0 <= result["top1"] <= result["top5"] <= 1
