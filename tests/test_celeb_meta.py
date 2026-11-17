"""Tests for collecting the raw photos of the celebrity gallery, with small fake folders."""
import csv
from pathlib import Path

import pytest

from whoisit.data import celeb_meta


def make_photos(folder, names):
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_bytes(name.encode())


@pytest.fixture
def setup(tmp_path):
    pins = tmp_path / "pins"
    # 12 photos, so only the first 10 by file name are taken; the file names have spaces
    make_photos(pins / "pins_Henry Cavil", [f"Henry Cavil{i:02d}_0.jpg" for i in range(12)])
    make_photos(pins / "pins_Zendaya", ["Zendaya1.jpg", "Zendaya0.jpg", "notes.txt"])
    lfw = tmp_path / "lfw"
    make_photos(lfw / "Tom_Hanks", [f"Tom_Hanks_{i:04d}.jpg" for i in range(1, 13)])
    make_photos(lfw / "Halle_Berry", ["Halle_Berry_0001.jpg"])
    make_photos(lfw / "George_W_Bush", ["George_W_Bush_0001.jpg"])  # not chosen

    (tmp_path / "pins.csv").write_text(
        "folder,name\npins_Henry Cavil,Henry_Cavill\npins_Zendaya,Zendaya\n", encoding="utf-8")
    (tmp_path / "lfw.txt").write_text("Tom_Hanks\n\nHalle_Berry\n", encoding="utf-8")
    # no one excluded, only the comment line
    (tmp_path / "exclude.txt").write_text("# CelebA training people\n", encoding="utf-8")
    return tmp_path


def run_main(tmp_path):
    celeb_meta.main(["--pins-dir", str(tmp_path / "pins"), "--lfw-dir", str(tmp_path / "lfw"),
                     "--pins-names", str(tmp_path / "pins.csv"), "--lfw-names", str(tmp_path / "lfw.txt"),
                     "--exclude", str(tmp_path / "exclude.txt"), "--out-dir", str(tmp_path / "raw")])
    with open(tmp_path / "raw" / "meta.csv", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_copies_the_chosen_photos_and_writes_meta(setup, capsys):
    rows = run_main(setup)
    raw = setup / "raw"

    by_person = {}
    for row in rows:
        by_person.setdefault(row["identity"], []).append(row["image_path"])
        # the path is relative to the raw folder and the photo is there
        assert (raw / row["image_path"]).read_bytes()
    assert sorted(by_person) == ["Halle_Berry", "Henry_Cavill", "Tom_Hanks", "Zendaya"]
    # first 10 by file name, spaces replaced
    assert by_person["Henry_Cavill"] == [f"Henry_Cavill/Henry_Cavil{i:02d}_0.jpg" for i in range(10)]
    assert by_person["Tom_Hanks"] == [f"Tom_Hanks/Tom_Hanks_{i:04d}.jpg" for i in range(1, 11)]
    # only .jpg files
    assert by_person["Zendaya"] == ["Zendaya/Zendaya0.jpg", "Zendaya/Zendaya1.jpg"]
    assert not (raw / "George_W_Bush").exists()
    assert not any(" " in path.name for path in raw.rglob("*"))
    # the content is copied unchanged
    assert (raw / "Henry_Cavill/Henry_Cavil00_0.jpg").read_bytes() == b"Henry Cavil00_0.jpg"

    output = capsys.readouterr().out
    assert "pins: 2 people, 12 photos" in output
    assert "lfw: 2 people, 11 photos" in output
    assert "total: 4 people, 23 photos" in output


def test_excluded_people_are_left_out(setup, capsys):
    # one from Pins and one from LFW, with a comment and an empty line
    (setup / "exclude.txt").write_text("# training people\nHenry_Cavill\n\nTom_Hanks\n", encoding="utf-8")
    rows = run_main(setup)

    assert sorted({row["identity"] for row in rows}) == ["Halle_Berry", "Zendaya"]
    assert not (setup / "raw" / "Henry_Cavill").exists()
    assert not (setup / "raw" / "Tom_Hanks").exists()
    output = capsys.readouterr().out
    assert "excluded: 2 people" in output
    assert "pins: 1 people, 2 photos" in output
    assert "lfw: 1 people, 1 photos" in output
    assert "total: 2 people, 3 photos" in output


def test_no_one_excluded(setup, capsys):
    run_main(setup)
    assert "excluded: 0 people" in capsys.readouterr().out


def test_exclude_name_not_in_gallery_lists_gives_an_error(setup):
    # the Pins folder name instead of the name, and a typo
    (setup / "exclude.txt").write_text("Henry_Cavil\nZendaya\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not in the gallery lists: Henry_Cavil$"):
        run_main(setup)
    assert not (setup / "raw").exists()


def test_exclude_name_twice_gives_an_error(setup):
    (setup / "exclude.txt").write_text("Zendaya\nZendaya\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="appear twice: Zendaya"):
        run_main(setup)


def test_missing_exclude_file_gives_an_error(setup):
    (setup / "exclude.txt").unlink()
    with pytest.raises(SystemExit, match="exclude.txt not found"):
        run_main(setup)


def test_pins_folder_missing_in_table_gives_an_error(setup):
    make_photos(setup / "pins" / "pins_Rihanna", ["Rihanna0.jpg"])
    with pytest.raises(SystemExit, match="pins_Rihanna"):
        run_main(setup)


def test_table_folder_that_does_not_exist_gives_an_error(setup):
    with open(setup / "pins.csv", "a", encoding="utf-8") as file:
        file.write("pins_Nobody,Nobody\n")
    with pytest.raises(SystemExit, match="pins_Nobody"):
        run_main(setup)


def test_pins_folder_twice_gives_an_error(setup):
    with open(setup / "pins.csv", "a", encoding="utf-8") as file:
        file.write("pins_Zendaya,Zendaya_2\n")
    with pytest.raises(SystemExit, match="twice"):
        run_main(setup)


def test_same_name_in_pins_and_lfw_gives_an_error(setup):
    make_photos(setup / "lfw" / "Zendaya", ["Zendaya_0001.jpg"])
    (setup / "lfw.txt").write_text("Tom_Hanks\nZendaya\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="appear twice: Zendaya"):
        run_main(setup)
    assert not (setup / "raw").exists()


def test_same_name_twice_in_lfw_list_gives_an_error(setup):
    (setup / "lfw.txt").write_text("Tom_Hanks\nTom_Hanks\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="appear twice: Tom_Hanks"):
        run_main(setup)


def test_lfw_name_that_is_not_a_folder_gives_an_error(setup):
    (setup / "lfw.txt").write_text("Tom_Hanks\nTom_Hank\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Tom_Hank$"):
        run_main(setup)


def test_full_out_dir_gives_an_error(setup):
    (setup / "raw").mkdir()
    (setup / "raw" / "old.jpg").write_bytes(b"old")
    with pytest.raises(SystemExit, match="not empty"):
        run_main(setup)
    assert (setup / "raw" / "old.jpg").read_bytes() == b"old"


def test_real_config_files_have_no_double_names():
    # the checked-in lists: every name once, and Bill Gates, Keanu Reeves, Tom Cruise only from Pins
    repo = Path(__file__).resolve().parents[1]
    with open(repo / celeb_meta.PINS_NAMES, newline="", encoding="utf-8") as file:
        pins = [row["name"] for row in csv.DictReader(file)]
    lfw = (repo / celeb_meta.LFW_NAMES).read_text(encoding="utf-8").split()
    assert len(pins) == 105 and len(lfw) == 34
    celeb_meta.check_unique(pins + lfw)
    assert {"Bill_Gates", "Keanu_Reeves", "Tom_Cruise"} <= set(pins)
    assert not {"Bill_Gates", "Keanu_Reeves", "Tom_Cruise"} & set(lfw)

    # the 20 likely training people are all real gallery names, so 139 - 20 = 119 people stay
    exclude = celeb_meta.load_exclude(repo / celeb_meta.EXCLUDE, set(pins + lfw))
    assert len(exclude) == 20
    assert len(set(pins + lfw) - exclude) == 119
