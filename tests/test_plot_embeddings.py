"""Tests for the picture of the embedding space, with fake photos and tiny fake checkpoints."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from whoisit import plot_embeddings  # noqa: E402
from whoisit.data.align import load_image  # noqa: E402
from whoisit.models.backbone import Backbone  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import smoke_test  # noqa: E402  (baseline.py gets its model from here)

PEOPLE = 4  # fewer than the real 20, so the tests are fast


def make_photos_table(counts):
    return pd.DataFrame([(f"{person}_{i:02d}.jpg", person) for person, count in counts.items()
                         for i in range(count)], columns=["image", "person"])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(plot_embeddings, "PEOPLE", PEOPLE)
    # no ImageNet download in the tests
    monkeypatch.setattr(plot_embeddings, "Backbone", lambda pretrained: Backbone(pretrained=False))
    torch.manual_seed(0)
    # 5 people with 10 photos, and one with only 9 who is never picked
    table = make_photos_table({**{str(p): 10 for p in range(5)}, "few": 9})
    aligned = tmp_path / "aligned"
    with open(aligned.parent / "meta.tmp", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "identity", "source_path"])
        for image, person in zip(table["image"], table["person"]):
            (aligned / "images" / person).mkdir(parents=True, exist_ok=True)
            Image.effect_noise((112, 112), 50).convert("RGB").save(aligned / "images" / person / image)
            writer.writerow([f"images/{person}/{image}", person, image])
    (aligned.parent / "meta.tmp").rename(aligned / "meta.csv")
    return tmp_path


def make_run(runs, loss, n=20, seed=42, checkpoint=True):
    folder = runs / f"{loss}_n{n}_seed{seed}_ep20"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({"options": {"loss": loss, "n": n, "seed": seed}}),
                                        encoding="utf-8")
    if checkpoint:
        (folder / "checkpoints").mkdir()
        torch.save(Backbone(pretrained=False).state_dict(), folder / "checkpoints" / "model.pt")
    return folder


def run_main(tmp_path, folders, *extra):
    out = tmp_path / "figures" / "embeddings.png"
    plot_embeddings.main(["--runs", *map(str, folders), "--aligned-dir", str(tmp_path / "aligned"),
                          "--device", "cpu", "--batch-size", "16", "--out", str(out), *extra])
    return out


def test_pick_photos_is_fixed_by_the_seed(monkeypatch):
    monkeypatch.setattr(plot_embeddings, "PEOPLE", 5)
    table = make_photos_table({**{f"p{i}": 10 for i in range(30)}, "few": 9, "many": 12})
    first = plot_embeddings.pick_photos(table, 42)
    # the order of the rows in meta.csv must not matter
    again = plot_embeddings.pick_photos(table.sample(frac=1, random_state=1), 42)
    pd.testing.assert_frame_equal(first, again)
    assert set(first["person"]) != set(plot_embeddings.pick_photos(table, 43)["person"])
    assert first["person"].nunique() == 5
    assert (first.groupby("person").size() == 10).all()


def test_people_with_too_few_photos_are_not_picked(monkeypatch):
    monkeypatch.setattr(plot_embeddings, "PEOPLE", 2)
    # "many" has 12 photos, only 10 are used
    table = make_photos_table({"a": 10, "few": 9, "many": 12})
    picked = plot_embeddings.pick_photos(table, 0)
    assert picked.groupby("person").size().to_dict() == {"a": 10, "many": 10}
    monkeypatch.setattr(plot_embeddings, "PEOPLE", 3)
    with pytest.raises(ValueError, match="only 2 test people"):
        plot_embeddings.pick_photos(table, 0)


def test_imagenet_panel_is_the_baseline_model(setup):
    # same weights in both models, then the embeddings must be the same as in baseline.py
    ours = plot_embeddings.load_imagenet("cpu")
    baseline = smoke_test.build_model(pretrained=False)
    baseline.load_state_dict(ours.state_dict())

    photos = plot_embeddings.load_test_photos(setup / "aligned").head(8)
    data = plot_embeddings.FaceDataset(photos, image_dir=setup / "aligned/images", train=False)
    ours_embeddings = plot_embeddings.normalize(
        plot_embeddings.embed(ours, DataLoader(data, batch_size=8), "cpu"))
    faces = np.stack([load_image(path) for path in data.paths])
    assert np.allclose(ours_embeddings, smoke_test.embed(baseline, faces), atol=1e-5)


def test_imagenet_panel_asks_for_the_imagenet_weights(monkeypatch):
    asked = []

    def fake(pretrained):
        asked.append(pretrained)
        return Backbone(pretrained=False)
    monkeypatch.setattr(plot_embeddings, "Backbone", fake)
    plot_embeddings.load_imagenet("cpu")
    assert asked == [True]


def clusters(spread):
    # 4 people, 10 photos each, around 4 directions
    rng = np.random.default_rng(0)
    centers = np.repeat(np.eye(4, 16), 10, axis=0)
    return plot_embeddings.normalize(centers + rng.normal(scale=spread, size=(40, 16))), np.repeat(np.arange(4), 10)


def test_separation_is_high_for_clear_groups_and_low_for_noise():
    clear, labels = clusters(0.01)
    noise, _ = clusters(10)
    assert plot_embeddings.separation(clear, labels) > 0.95
    assert plot_embeddings.separation(noise, labels) < 0.1


def test_tsne_is_the_same_every_time():
    embeddings, _ = clusters(0.2)
    first = plot_embeddings.to_2d(embeddings)
    assert first.shape == (40, 2)
    assert np.array_equal(first, plot_embeddings.to_2d(embeddings))


def test_picture_and_scores(setup, capsys):
    folders = [make_run(setup / "runs", "ce"), make_run(setup / "runs", "arcface")]
    out = run_main(setup, folders)

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "picked: 4 people, 40 photos, device cpu"
    scores = [line for line in lines if "silhouette score" in line]
    assert [line.split(":")[0] for line in scores] == [
        "imagenet", "ce_n20_seed42_ep20", "arcface_n20_seed42_ep20"]


def test_same_color_per_person_in_every_panel(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plot_embeddings.plt, "close", figures.append)  # keep the figure to look at it
    # 20 people like in the real picture
    labels = np.repeat(np.arange(20), 10)
    points = np.random.default_rng(0).normal(size=(200, 2))
    plot_embeddings.plot([("A", points), ("B", points[::-1])], labels, tmp_path / "x.png")

    fig = figures[0]
    styles = [[(tuple(c.get_facecolor()[0]), c.get_paths()[0].vertices.tobytes()) for c in ax.collections]
              for ax in fig.axes]
    assert styles[0] == styles[1]
    assert len({color for color, _ in styles[0]}) == 20
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["A", "B"]
    for ax in fig.axes:
        assert ax.get_legend() is None and not ax.images
        assert len(ax.get_xticks()) == 0 and len(ax.get_yticks()) == 0
    assert fig.texts[0].get_text() == ("20 test people, 10 photos each. One color per person. "
                                       "t-SNE of the embeddings.")


def test_titles(setup):
    ce, arcface = make_run(setup / "runs", "ce"), make_run(setup / "runs", "arcface")
    assert plot_embeddings.load_titles([ce, arcface]) == ["Cross-entropy, N = 20", "ArcFace, N = 20"]
    # same loss and n in another folder: the folder is in the title
    tuned = make_run(setup / "m0.3_ep20", "arcface")
    assert plot_embeddings.load_titles([arcface, tuned]) == [
        "ArcFace, N = 20\nruns/arcface_n20_seed42_ep20",
        "ArcFace, N = 20\nm0.3_ep20/arcface_n20_seed42_ep20"]


def test_default_runs():
    assert [path.as_posix() for path in plot_embeddings.parse_args([]).runs] == [
        "results/runs/ce_n20_seed42_ep20", "results/runs/arcface_n20_seed42_ep20"]


def test_missing_checkpoint_or_run_gives_an_error(setup):
    folder = make_run(setup / "runs", "ce", checkpoint=False)
    with pytest.raises(SystemExit, match="model.pt not found"):
        run_main(setup, [folder])
    with pytest.raises(SystemExit, match="result.json not found"):
        run_main(setup, [setup / "runs" / "nothing"])
    assert not (setup / "figures").exists()


def test_missing_photos_give_an_error(setup):
    (setup / "aligned" / "meta.csv").unlink()
    with pytest.raises(SystemExit, match="meta.csv not found"):
        run_main(setup, [make_run(setup / "runs", "ce")])
