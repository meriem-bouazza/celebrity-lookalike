"""Tests for one training run, on the cpu with small fake photos and no download."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from whoisit import train  # noqa: E402
from whoisit.data.dataset import load_val_photos  # noqa: E402
from whoisit.data.images import FaceDataset  # noqa: E402
from whoisit.models.backbone import Backbone  # noqa: E402
from whoisit.models.heads import HEADS, SoftmaxHead, make_head  # noqa: E402

TRAIN_PEOPLE = [13, 99]
RUN = ["--n", "2", "--epochs", "2", "--batch-size", "4", "--seed", "1"]


def save_photo(folder, person, name):
    folder = folder / str(person)
    folder.mkdir(parents=True, exist_ok=True)
    Image.effect_noise((112, 112), 50).convert("RGB").save(folder / name)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    # all paths in the code are relative, so a fake repo is just another working folder
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "Backbone", lambda: Backbone(pretrained=False))
    images = tmp_path / "data/aligned/celeba/images"
    (tmp_path / "data/celeba").mkdir(parents=True)
    (tmp_path / "splits").mkdir()

    identities = [(f"{p}_{i}.jpg", p) for p in TRAIN_PEOPLE for i in range(20)]
    # person 501 keeps 3 photos, 502 keeps 2, 503 has only 1 aligned photo left
    val = [(f"{p}_{i}.jpg", p) for p, count in [(501, 3), (502, 3), (503, 2)] for i in range(count)]
    failed = {"502_2.jpg", "503_1.jpg"}
    aligned = [row for row in identities + val if row[0] not in failed]
    for name, person in aligned:
        save_photo(images, person, name)

    (tmp_path / "data/celeba/identity_CelebA.txt").write_text(
        "".join(f"{name} {person}\n" for name, person in identities + val))
    np.savetxt("splits/train_people.txt", TRAIN_PEOPLE, fmt="%d")
    pd.DataFrame(val, columns=["image", "person"]).to_csv("splits/val_photos.csv", index=False)
    pd.DataFrame({"source_path": [name for name, _ in aligned]}).to_csv(
        "data/aligned/celeba/meta.csv", index=False)
    return tmp_path


def read_result(folder):
    return json.loads((folder / "result.json").read_text(encoding="utf-8"))


def test_val_table_keeps_aligned_photos_and_drops_single_people(tmp_path):
    path = tmp_path / "val.csv"
    rows = [("a1.jpg", 1), ("a2.jpg", 1), ("a3.jpg", 1), ("b1.jpg", 2), ("b2.jpg", 2), ("c1.jpg", 3)]
    pd.DataFrame(rows, columns=["image", "person"]).to_csv(path, index=False)
    aligned = {"a1.jpg", "a3.jpg", "b1.jpg", "c1.jpg"}

    photos = load_val_photos(aligned, path)

    # person 2 and 3 have only one aligned photo left
    assert list(photos["image"]) == ["a1.jpg", "a3.jpg"]
    assert list(photos["person"]) == [1, 1]


def test_run_folder_name():
    assert train.run_folder("ce", 5, 42, 20).as_posix() == "results/runs/ce_n5_seed42_ep20"


def test_unknown_loss_is_rejected():
    with pytest.raises(SystemExit):
        train.parse_args(["--loss", "nope"])


def test_validate_goes_back_to_train_mode_and_keeps_batchnorm(tmp_path):
    for i in range(4):
        save_photo(tmp_path, 7, f"{i}.jpg")
    photos = pd.DataFrame({"image": [f"{i}.jpg" for i in range(4)], "person": 7})
    loader = DataLoader(FaceDataset(photos, image_dir=tmp_path, train=False), batch_size=2)
    model = Backbone(pretrained=False)
    before = model.embedding[1].running_mean.clone()

    top1, top5 = train.validate(model, loader, "cpu")

    assert model.training
    assert torch.equal(before, model.embedding[1].running_mean)
    assert top1 == top5 == 1.0  # only one person, so every neighbour is right


def test_fake_run_writes_the_result(fake_repo):
    train.main(RUN)

    folder = fake_repo / "results/runs/ce_n2_seed1_ep2"
    result = read_result(folder)
    assert result["options"] == {"loss": "ce", "n": 2, "seed": 1, "epochs": 2,
                                 "batch_size": 4, "lr": 3e-4}
    assert result["train"] == {"people": 2, "photos": 4}
    assert result["val"] == {"people": 2, "photos": 5}  # 503 is dropped, 502_2 was not aligned
    assert [row["epoch"] for row in result["epochs"]] == [1, 2]
    for row in result["epochs"]:
        assert row["loss"] > 0 and row["seconds"] >= 0
        assert 0 <= row["top1"] <= row["top5"] <= 1
    assert result["final"] == {"top1": result["epochs"][-1]["top1"],
                               "top5": result["epochs"][-1]["top5"]}
    assert (folder / "checkpoints/model.pt").exists()
    assert not (folder / "result.json.tmp").exists()


def test_skip_when_the_result_exists(fake_repo, monkeypatch, capsys):
    folder = fake_repo / "results/runs/ce_n2_seed1_ep2"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text('{"old": true}')

    def fail():
        raise AssertionError("the run should be skipped")
    monkeypatch.setattr(train, "Backbone", fail)
    train.main(RUN)

    assert "skip" in capsys.readouterr().out
    assert read_result(folder) == {"old": True}


def test_other_epochs_are_not_skipped(fake_repo):
    old = fake_repo / "results/runs/ce_n2_seed1_ep2"
    old.mkdir(parents=True)
    (old / "result.json").write_text('{"old": true}')

    train.main(["--n", "2", "--epochs", "1", "--batch-size", "4", "--seed", "1"])

    assert read_result(old) == {"old": True}
    assert read_result(fake_repo / "results/runs/ce_n2_seed1_ep1")["options"]["epochs"] == 1


def test_force_runs_again(fake_repo):
    folder = fake_repo / "results/runs/ce_n2_seed1_ep2"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text('{"old": true}')

    train.main([*RUN, "--force"])

    assert "old" not in read_result(folder)
    assert "options" in read_result(folder)


def test_seed_changes_the_training_but_not_the_photos(fake_repo, monkeypatch):
    used = []

    def spy(photos, train=True):
        if train:
            used.append(sorted(photos["image"]))
        return FaceDataset(photos, train=train)
    monkeypatch.setattr(train, "FaceDataset", spy)

    train.main(["--n", "2", "--epochs", "1", "--batch-size", "4", "--seed", "1"])
    train.main(["--n", "2", "--epochs", "1", "--batch-size", "4", "--seed", "2"])

    assert used[0] == used[1]
    a = torch.load("results/runs/ce_n2_seed1_ep1/checkpoints/model.pt")
    b = torch.load("results/runs/ce_n2_seed2_ep1/checkpoints/model.pt")
    assert not torch.equal(a["embedding.0.weight"], b["embedding.0.weight"])


def test_new_head_only_needs_an_entry_in_heads(fake_repo, monkeypatch):
    monkeypatch.setitem(HEADS, "other", SoftmaxHead)
    assert isinstance(make_head("other", 3), SoftmaxHead)

    train.main(["--loss", "other", *RUN])

    assert read_result(fake_repo / "results/runs/other_n2_seed1_ep2")["options"]["loss"] == "other"


def test_arcface_run_loss_goes_down_and_result_has_margin_and_scale(fake_repo):
    # one color per person, so the network has something it can learn
    for person, color in zip(TRAIN_PEOPLE, ("red", "blue")):
        for path in (fake_repo / f"data/aligned/celeba/images/{person}").iterdir():
            Image.new("RGB", (112, 112), color).save(path)

    train.main(["--loss", "arcface", "--n", "2", "--epochs", "5", "--batch-size", "4", "--seed", "1"])

    result = read_result(fake_repo / "results/runs/arcface_n2_seed1_ep5")
    assert result["options"]["loss"] == "arcface"
    assert result["options"]["margin"] == 0.5
    assert result["options"]["scale"] == 64.0
    losses = [row["loss"] for row in result["epochs"]]
    assert losses[-1] < losses[0]


def test_margin_and_scale_options_reach_the_head(fake_repo, monkeypatch):
    made = []

    def spy(loss, num_people, **options):
        made.append(options)
        return make_head(loss, num_people, **options)
    monkeypatch.setattr(train, "make_head", spy)

    train.main(["--loss", "arcface", "--margin", "0.3", "--scale", "30", *RUN])
    train.main(["--loss", "ce", "--margin", "0.3", "--scale", "30", *RUN])

    assert made == [{"margin": 0.3, "scale": 30.0}, {}]  # cross-entropy gets nothing
    options = read_result(fake_repo / "results/runs/arcface_n2_seed1_ep2")["options"]
    assert (options["margin"], options["scale"]) == (0.3, 30.0)
    assert "margin" not in read_result(fake_repo / "results/runs/ce_n2_seed1_ep2")["options"]


def test_every_loss_sees_the_same_photos_and_flips(fake_repo, monkeypatch):
    seen = {}

    def run(loss):
        batches = seen[loss] = []

        def record(model, inputs):
            if model.training:  # validation also calls the model
                batches.append(inputs[0].clone())

        def spy():
            model = Backbone(pretrained=False)
            model.register_forward_pre_hook(record)
            return model
        monkeypatch.setattr(train, "Backbone", spy)
        train.main(["--loss", loss, "--n", "5", "--epochs", "2", "--batch-size", "4", "--seed", "1"])

    run("ce")
    run("arcface")

    # 10 photos with batch size 4: 2 batches in each of the 2 epochs
    assert len(seen["ce"]) == len(seen["arcface"]) == 4
    for a, b in zip(seen["ce"], seen["arcface"]):
        assert torch.equal(a, b)
