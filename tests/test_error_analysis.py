"""Tests for the error analysis, with fake photos, tiny fake checkpoints and a made-up attribute file."""
import csv
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from whoisit import error_analysis  # noqa: E402
from whoisit.evaluate import top_k_accuracy  # noqa: E402
from whoisit.models.backbone import Backbone  # noqa: E402

NAMES = ["Blurry", "Eyeglasses", "Male", "Smiling", "Wearing_Hat", "Young"]
# 3 people with 3 photos each; person 4 has only 1 photo and is left out like in final_eval.py
PHOTOS = [(p, i) for p in ("1", "2", "3") for i in range(3)] + [("4", 0)]


def photo_name(person, i):
    return f"{person}{i}.jpg"


def make_data(tmp_path):
    aligned = tmp_path / "aligned"
    rows = []
    for person, i in PHOTOS:
        (aligned / "images" / person).mkdir(parents=True, exist_ok=True)
        path = f"images/{person}/{photo_name(person, i)}"
        Image.effect_noise((112, 112), 50).convert("RGB").save(aligned / path)
        rows.append([path, person, photo_name(person, i)])
    with open(aligned / "meta.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "identity", "source_path"])
        writer.writerows(rows)

    # photo i of each person: 0 has all attributes, 1 and 2 have none, so every group has photos
    lines = [str(len(PHOTOS)), " ".join(NAMES)]
    for person, i in PHOTOS:
        value = "1" if i == 0 else "-1"
        lines.append(f"{photo_name(person, i)}  " + "  ".join([value] * len(NAMES)))
    (tmp_path / "attr.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_run(runs, loss="ce", n=2, seed=42, checkpoint=True, extra_options=None):
    folder = runs / f"{loss}_n{n}_seed{seed}_ep2"
    folder.mkdir(parents=True)
    options = {"loss": loss, "n": n, "seed": seed, "epochs": 2, **(extra_options or {})}
    result = {"options": options, "final": {"top1": 0.5, "top5": 0.7}}
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if checkpoint:
        torch.manual_seed(seed)
        (folder / "checkpoints").mkdir()
        torch.save(Backbone(pretrained=False).state_dict(), folder / "checkpoints" / "model.pt")
    return folder


@pytest.fixture
def setup(tmp_path):
    make_data(tmp_path)
    return tmp_path


def run_main(tmp_path):
    error_analysis.main(["--runs-dir", str(tmp_path / "runs"), "--attr-file", str(tmp_path / "attr.txt"),
                         "--aligned-dir", str(tmp_path / "aligned"), "--device", "cpu",
                         "--batch-size", "4", "--out-runs", str(tmp_path / "tables/runs.csv"),
                         "--out", str(tmp_path / "tables/mean.csv")])
    return (pd.read_csv(tmp_path / "tables/runs.csv", dtype=str),
            pd.read_csv(tmp_path / "tables/mean.csv", dtype=str))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_top1_hits_mean_is_top_k_accuracy(seed):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(60, 8))
    labels = np.repeat(np.arange(15), 4)
    hits = error_analysis.top1_hits(embeddings, labels)
    assert hits.mean() == top_k_accuracy(embeddings, labels)[1]


def test_top1_hits_by_hand():
    # photo 2 is closest to photo 1 (other person), all others find their own person
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]])
    labels = [0, 0, 1, 1, 2]
    hits = error_analysis.top1_hits(embeddings, labels)
    assert list(hits) == [True, True, False, False, False]


def test_split_uses_the_same_hits_for_both_groups():
    hits = np.array([True, False, True, True])
    has = pd.DataFrame({"Eyeglasses": [True, True, False, False]})
    assert error_analysis.split_by_attribute(hits, has) == [{
        "attribute": "Eyeglasses", "with_photos": 2, "with_top1": 0.5,
        "without_photos": 2, "without_top1": 1.0}]


def test_split_without_a_group_gives_an_error():
    with pytest.raises(ValueError, match="Smiling"):
        error_analysis.split_by_attribute(np.array([True, False]), pd.DataFrame({"Smiling": [True, True]}))


def test_mean_table():
    rows = [{"loss": "ce", "n": 2, "seed": seed, "attribute": "Eyeglasses", "with_photos": 5,
             "with_top1": w, "without_photos": 95, "without_top1": wo}
            for seed, w, wo in [(42, 0.10, 0.20), (43, 0.20, 0.25), (44, 0.30, 0.30)]]
    table = error_analysis.mean_table(rows)
    assert table.to_dict("records") == [{
        "loss": "ce", "n": 2, "attribute": "Eyeglasses", "seeds": 3, "with_photos": 5,
        "without_photos": 95, "top1_with": "20.00", "top1_without": "25.00",
        "difference_mean": "-5.00", "difference_min": "-10.00", "difference_max": "0.00"}]


def test_both_tables_are_written_and_sorted(setup, capsys):
    for n in (10, 2):
        for seed in (42, 43):
            make_run(setup / "runs", n=n, seed=seed)
    runs, mean = run_main(setup)

    # 4 runs x 5 attributes, sorted by n as a number, then seed, then the order of ATTRIBUTES
    assert len(runs) == 20
    assert list(runs["n"][::5]) == ["2", "2", "10", "10"]
    assert list(runs["seed"][::5]) == ["42", "43", "42", "43"]
    assert list(runs["attribute"][:5]) == error_analysis.ATTRIBUTES
    # 9 photos: photo 0 of each person has every attribute
    assert set(runs["with_photos"]) == {"3"} and set(runs["without_photos"]) == {"6"}

    assert list(mean.columns) == ["loss", "n", "attribute", "seeds", "with_photos", "without_photos",
                                  "top1_with", "top1_without", "difference_mean", "difference_min",
                                  "difference_max"]
    assert len(mean) == 10 and set(mean["seeds"]) == {"2"}
    output = capsys.readouterr().out
    assert "test: 3 people, 9 photos" in output
    assert "difference_mean" in output


def test_mean_in_file_matches_the_runs(setup):
    for seed in (42, 43):
        make_run(setup / "runs", seed=seed)
    runs, mean = run_main(setup)
    glasses = runs[runs["attribute"] == "Eyeglasses"]
    expected = (glasses["with_top1"].astype(float) - glasses["without_top1"].astype(float)).mean()
    row = mean[mean["attribute"] == "Eyeglasses"].iloc[0]
    # the files have 2 decimals, so allow a rounding difference
    assert float(row["difference_mean"]) == pytest.approx(expected, abs=0.011)


def test_warning_if_test_json_does_not_match(setup, capsys):
    folder = make_run(setup / "runs")
    test = {"test": {"people": 3, "photos": 9}, "final": {"top1": 2.0, "top5": 1.0}}
    (folder / "test.json").write_text(json.dumps(test), encoding="utf-8")
    run_main(setup)
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("warning: ce_n2_seed42_ep2:") for line in lines)


def test_no_warning_if_test_json_matches(setup, capsys):
    folder = make_run(setup / "runs")
    # compute the real Top 1 by hand, then write a matching test.json
    data = error_analysis.FaceDataset(error_analysis.load_test_sources(setup / "aligned"),
                                      image_dir=setup / "aligned" / "images", train=False)
    model = error_analysis.load_model(folder / "checkpoints" / "model.pt", "cpu")
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    top1 = error_analysis.top1_hits(error_analysis.embed(model, loader, "cpu"), data.labels).mean()
    test = {"test": {"people": 3, "photos": 9}, "final": {"top1": float(top1), "top5": 1.0}}
    (folder / "test.json").write_text(json.dumps(test), encoding="utf-8")

    run_main(setup)
    # the folder name of the test contains "warning", so look at the start of the lines
    assert not any(line.startswith("warning") for line in capsys.readouterr().out.splitlines())


def test_missing_checkpoint_gives_an_error(setup):
    make_run(setup / "runs", seed=42)
    make_run(setup / "runs", seed=43, checkpoint=False)
    with pytest.raises(SystemExit, match="ce_n2_seed43_ep2"):
        run_main(setup)
    assert not (setup / "tables").exists()


def test_different_settings_give_an_error(setup):
    make_run(setup / "runs", loss="arcface", seed=42, extra_options={"margin": 0.5, "scale": 64.0})
    make_run(setup / "runs", loss="arcface", seed=43, extra_options={"margin": 0.3, "scale": 64.0})
    with pytest.raises(SystemExit, match="margin"):
        run_main(setup)


def test_missing_attribute_gives_an_error(setup):
    make_run(setup / "runs")
    lines = (setup / "attr.txt").read_text(encoding="utf-8").splitlines()
    # remove the column Blurry (the first one) from the file
    lines = [lines[0], " ".join(lines[1].split()[1:])] + [" ".join(line.split()[:1] + line.split()[2:])
                                                         for line in lines[2:]]
    (setup / "attr.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Blurry"):
        run_main(setup)


def test_test_photo_missing_in_attribute_file_gives_an_error(setup):
    make_run(setup / "runs")
    lines = (setup / "attr.txt").read_text(encoding="utf-8").splitlines()
    lines = [str(len(lines) - 3), lines[1]] + [line for line in lines[2:] if not line.startswith("20.jpg")]
    (setup / "attr.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="20.jpg"):
        run_main(setup)
