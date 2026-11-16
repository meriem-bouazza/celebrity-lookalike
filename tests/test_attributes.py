"""Tests for the attribute counts, with a tiny made-up attribute file and meta.csv."""
import csv

import pytest

pytest.importorskip("torch")  # final_eval.py needs torch
from whoisit import attributes  # noqa: E402

NAMES = ["Eyeglasses", "Smiling", "Wearing_Hat"]
# person 1: 3 photos, person 2: 2 photos, person 3: only 1 photo, so it is left out
TEST_PHOTOS = [("1", "a.jpg"), ("1", "b.jpg"), ("1", "c.jpg"), ("2", "d.jpg"), ("2", "e.jpg"),
               ("3", "f.jpg")]
VALUES = {
    "a.jpg": "1 1 -1", "b.jpg": "1 -1 -1", "c.jpg": "-1 1 1",
    "d.jpg": "1 1 -1", "e.jpg": "-1 1 -1",
    "f.jpg": "1 1 1",
    "x.jpg": "1 1 1",  # not a test photo
}


def write_meta(folder, photos=TEST_PHOTOS, source=lambda name: name):
    folder.mkdir(parents=True)
    with open(folder / "meta.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "identity", "source_path"])
        for person, name in photos:
            writer.writerow([f"images/{person}/{name}", person, source(name)])


def write_attr(path, values=VALUES, count=None):
    lines = [str(len(values) if count is None else count), " ".join(NAMES)]
    # the real file uses more than one space between the values
    lines += [f"{name}  {value.replace(' ', '  ')}" for name, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def files(tmp_path):
    write_meta(tmp_path / "aligned")
    write_attr(tmp_path / "attr.txt")
    return tmp_path


def run_main(tmp_path):
    out = tmp_path / "tables" / "test_attributes.csv"
    attributes.main(["--attr-file", str(tmp_path / "attr.txt"), "--out", str(out),
                     "--aligned-dir", str(tmp_path / "aligned")])
    with out.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_counts(files):
    rows = {row["attribute"]: row for row in run_main(files)}
    # 5 test photos: a b c d e
    assert rows["Smiling"] == {"attribute": "Smiling", "photos": "4", "percent": "80.00", "people": "2"}
    assert rows["Eyeglasses"] == {"attribute": "Eyeglasses", "photos": "3", "percent": "60.00", "people": "1"}
    assert rows["Wearing_Hat"] == {"attribute": "Wearing_Hat", "photos": "1", "percent": "20.00", "people": "0"}


def test_biggest_first(files):
    assert [row["attribute"] for row in run_main(files)] == ["Smiling", "Eyeglasses", "Wearing_Hat"]


def test_same_count_is_sorted_by_name(tmp_path):
    write_meta(tmp_path / "aligned", photos=[("1", "a.jpg"), ("1", "b.jpg")])
    # Wearing_Hat and Eyeglasses both have 2 photos
    write_attr(tmp_path / "attr.txt", values={"a.jpg": "1 -1 1", "b.jpg": "1 -1 1"})
    assert [row["attribute"] for row in run_main(tmp_path)] == ["Eyeglasses", "Wearing_Hat", "Smiling"]


def test_uses_the_source_file_name(tmp_path):
    # the aligned photo is always .jpg, the attribute file has the original name
    write_meta(tmp_path / "aligned", source=lambda name: "img_celeba/" + name.replace(".jpg", ".png"))
    values = {name.replace(".jpg", ".png"): value for name, value in VALUES.items()}
    write_attr(tmp_path / "attr.txt", values=values)
    rows = {row["attribute"]: row for row in run_main(tmp_path)}
    assert rows["Smiling"]["photos"] == "4"


def test_missing_test_photo_gives_an_error(tmp_path):
    write_meta(tmp_path / "aligned")
    write_attr(tmp_path / "attr.txt", values={k: v for k, v in VALUES.items() if k != "d.jpg"})
    with pytest.raises(SystemExit, match="d.jpg"):
        run_main(tmp_path)
    assert not (tmp_path / "tables").exists()


def test_wrong_photo_count_gives_an_error(tmp_path):
    write_meta(tmp_path / "aligned")
    write_attr(tmp_path / "attr.txt", count=99)
    with pytest.raises(SystemExit, match="99"):
        run_main(tmp_path)


def test_bad_value_gives_an_error(tmp_path):
    write_meta(tmp_path / "aligned")
    write_attr(tmp_path / "attr.txt", values={**VALUES, "a.jpg": "1 0 -1"})
    with pytest.raises(SystemExit, match="bad line"):
        run_main(tmp_path)


def test_missing_attribute_file_gives_an_error(tmp_path):
    write_meta(tmp_path / "aligned")
    with pytest.raises(SystemExit, match="not found"):
        run_main(tmp_path)


def test_table_is_printed(files, capsys):
    run_main(files)
    output = capsys.readouterr().out
    assert "test: 2 people, 5 photos" in output
    assert "Wearing_Hat" in output
