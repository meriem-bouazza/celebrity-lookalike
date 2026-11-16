"""Tests for the evaluation on the test people, with small fake photos and checkpoints."""
import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from whoisit import final_eval  # noqa: E402
from whoisit.evaluate import top_k_accuracy  # noqa: E402
from whoisit.models.backbone import Backbone  # noqa: E402

# person 3 has only one photo left, so it must be left out like in the baseline
PHOTOS = {"1": 3, "2": 2, "3": 1}


def make_aligned(folder):
    rows = []
    for person, count in PHOTOS.items():
        (folder / "images" / person).mkdir(parents=True)
        for i in range(count):
            path = f"images/{person}/{person}_{i}.jpg"
            Image.effect_noise((112, 112), 50).convert("RGB").save(folder / path)
            rows.append({"image_path": path, "identity": person, "source_path": f"{person}_{i}.jpg"})
    with open(folder / "meta.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["image_path", "identity", "source_path"])
        writer.writeheader()
        writer.writerows(rows)


def make_run(runs, name, checkpoint=True, finished=True):
    folder = runs / name
    folder.mkdir(parents=True)
    if finished:
        (folder / "result.json").write_text('{"final": {"top1": 0.5, "top5": 0.7}}\n', encoding="utf-8")
    if checkpoint:
        (folder / "checkpoints").mkdir()
        torch.save(Backbone(pretrained=False).state_dict(), folder / "checkpoints" / "model.pt")
    return folder


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    make_aligned(tmp_path / "aligned")
    return tmp_path


def run_main(tmp_path, *extra):
    final_eval.main(["--runs-dir", str(tmp_path / "runs"), "--aligned-dir", str(tmp_path / "aligned"),
                     "--batch-size", "2", *extra])


def test_people_with_one_photo_are_left_out(setup):
    photos = final_eval.load_test_photos(setup / "aligned")
    assert sorted(photos["person"].unique()) == ["1", "2"]
    assert len(photos) == 5


def test_writes_test_json(setup):
    folder = make_run(setup / "runs", "ce_n5_seed42_ep20")
    result_before = (folder / "result.json").read_bytes()
    run_main(setup)

    test = json.loads((folder / "test.json").read_text(encoding="utf-8"))
    assert test["test"] == {"people": 2, "photos": 5}
    assert 0 <= test["final"]["top1"] <= test["final"]["top5"] <= 1
    assert test["checkpoint"] == "checkpoints/model.pt"
    # result.json stays as it is
    assert (folder / "result.json").read_bytes() == result_before


def test_numbers_come_from_the_checkpoint(setup):
    folder = make_run(setup / "runs", "ce_n5_seed42_ep20")
    run_main(setup)

    # compute the same numbers by hand with the saved weights and evaluate.py
    model = final_eval.load_model(folder / "checkpoints" / "model.pt", "cpu").eval()
    photos = final_eval.load_test_photos(setup / "aligned")
    data = final_eval.FaceDataset(photos, image_dir=setup / "aligned" / "images", train=False)
    with torch.no_grad():
        embeddings = model(torch.stack([data[i][0] for i in range(len(data))])).numpy()
    expected = top_k_accuracy(embeddings, data.labels)

    test = json.loads((folder / "test.json").read_text(encoding="utf-8"))
    assert test["final"] == {"top1": expected[1], "top5": expected[5]}


def test_load_model_uses_the_saved_weights(setup):
    folder = make_run(setup / "runs", "ce_n5_seed42_ep20")
    saved = torch.load(folder / "checkpoints" / "model.pt", weights_only=True)
    loaded = final_eval.load_model(folder / "checkpoints" / "model.pt", "cpu").state_dict()
    assert all(torch.equal(saved[key], loaded[key]) for key in saved)


def test_existing_test_json_is_kept(setup, capsys):
    folder = make_run(setup / "runs", "ce_n5_seed42_ep20")
    (folder / "test.json").write_text("old\n", encoding="utf-8")
    run_main(setup)
    assert (folder / "test.json").read_text(encoding="utf-8") == "old\n"
    assert "skip ce_n5_seed42_ep20" in capsys.readouterr().out

    run_main(setup, "--force")
    assert "final" in json.loads((folder / "test.json").read_text(encoding="utf-8"))


def test_unfinished_run_is_skipped(setup):
    make_run(setup / "runs", "ce_n5_seed42_ep20")
    unfinished = make_run(setup / "runs", "ce_n5_seed43_ep20", finished=False)
    run_main(setup)
    assert not (unfinished / "test.json").exists()


def test_missing_checkpoint_gives_an_error_after_the_other_runs(setup):
    done = make_run(setup / "runs", "ce_n5_seed42_ep20")
    make_run(setup / "runs", "ce_n5_seed43_ep20", checkpoint=False)

    with pytest.raises(SystemExit, match="ce_n5_seed43_ep20"):
        run_main(setup)
    # the run with a checkpoint is still evaluated
    assert (done / "test.json").exists()
    assert not (setup / "runs" / "ce_n5_seed43_ep20" / "test.json").exists()


def test_no_finished_runs_give_an_error(setup):
    (setup / "runs").mkdir()
    with pytest.raises(SystemExit, match="no result.json"):
        run_main(setup)


def test_two_runs_with_the_same_weights_give_the_same_numbers(setup):
    a = make_run(setup / "runs", "ce_n5_seed42_ep20")
    b = make_run(setup / "runs", "ce_n5_seed43_ep20", checkpoint=False)
    (b / "checkpoints").mkdir()
    (b / "checkpoints" / "model.pt").write_bytes((a / "checkpoints" / "model.pt").read_bytes())
    run_main(setup)

    def final(folder):
        return json.loads((folder / "test.json").read_text(encoding="utf-8"))["final"]
    assert final(a) == final(b)
    assert np.isfinite(list(final(a).values())).all()
