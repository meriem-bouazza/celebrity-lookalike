"""Tests for the results table, with small made-up result.json and test.json files."""
import csv
import json
import subprocess
import sys

import pytest

from whoisit import summarize


def save_run(runs_dir, loss="ce", n=5, seed=42, epochs=20, top1=0.5, top5=0.7,
             extra_options=None, train=None, val=None):
    folder = runs_dir / f"{loss}_n{n}_seed{seed}_ep{epochs}"
    folder.mkdir(parents=True)
    options = {"loss": loss, "n": n, "seed": seed, "epochs": epochs, "batch_size": 128,
               "lr": 0.0003, "lr_schedule": "cosine", **(extra_options or {})}
    result = {"options": options, "train": train or {"people": 10, "photos": 10 * n},
              "val": val or {"people": 2, "photos": 8}, "final": {"top1": top1, "top5": top5}}
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return folder


def save_test(folder, top1=0.4, top5=0.6, test=None):
    result = {"checkpoint": "checkpoints/model.pt", "test": test or {"people": 3, "photos": 9},
              "final": {"top1": top1, "top5": top5}}
    (folder / "test.json").write_text(json.dumps(result), encoding="utf-8")


def run_main(tmp_path):
    out = tmp_path / "tables" / "summary.csv"
    summarize.main(["--runs-dir", str(tmp_path / "runs"), "--out", str(out)])
    with out.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_mean_min_max(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, seed=42, top1=0.3561, top5=0.6)
    save_run(runs, seed=43, top1=0.3, top5=0.51)
    save_run(runs, seed=44, top1=0.4, top5=0.75)

    assert run_main(tmp_path) == [{
        "split": "val", "loss": "ce", "n": "5", "seeds": "3",
        "top1_mean": "35.20", "top1_min": "30.00", "top1_max": "40.00",
        "top5_mean": "62.00", "top5_min": "51.00", "top5_max": "75.00",
    }]


def test_columns_are_in_the_right_order(tmp_path):
    save_run(tmp_path / "runs")
    run_main(tmp_path)
    header = (tmp_path / "tables" / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ("split,loss,n,seeds,top1_mean,top1_min,top1_max,"
                      "top5_mean,top5_min,top5_max")


def test_sorted_by_loss_then_n_as_a_number(tmp_path):
    runs = tmp_path / "runs"
    for loss, n in [("ce", 10), ("ce", 2), ("ce", 20), ("ce", 5), ("arcface", 10), ("arcface", 2)]:
        options = {"margin": 0.5, "scale": 64.0} if loss == "arcface" else None
        save_run(runs, loss=loss, n=n, extra_options=options)

    rows = run_main(tmp_path)
    assert [(row["loss"], row["n"]) for row in rows] == [
        ("arcface", "2"), ("arcface", "10"),
        ("ce", "2"), ("ce", "5"), ("ce", "10"), ("ce", "20"),
    ]


def test_folder_without_result_is_skipped(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, seed=42, top1=0.5)
    (runs / "ce_n5_seed43_ep20" / "checkpoints").mkdir(parents=True)

    rows = run_main(tmp_path)
    assert rows[0]["seeds"] == "1"
    assert rows[0]["top1_mean"] == "50.00"


def test_one_loss_only(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, n=2)
    save_run(runs, n=5)

    rows = run_main(tmp_path)
    assert {row["loss"] for row in rows} == {"ce"}
    assert len(rows) == 2


def test_different_epochs_in_one_row_give_an_error(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, seed=42, epochs=20)
    save_run(runs, seed=42, epochs=3)

    with pytest.raises(SystemExit) as error:
        run_main(tmp_path)
    message = str(error.value)
    assert "epochs" in message
    assert "ce_n5_seed42_ep3" in message and "ce_n5_seed42_ep20" in message
    assert not (tmp_path / "tables" / "summary.csv").exists()


@pytest.mark.parametrize("option, value", [
    ("lr", 0.001), ("lr_schedule", "none"), ("batch_size", 64), ("margin", 0.3), ("scale", 30.0),
])
def test_other_different_options_give_an_error(tmp_path, option, value):
    # margin and scale are missing in the first run, this must also count as different
    runs = tmp_path / "runs"
    save_run(runs, seed=42)
    save_run(runs, seed=43, extra_options={option: value})

    with pytest.raises(SystemExit, match=option):
        run_main(tmp_path)


def test_different_photo_count_gives_an_error(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, seed=42)
    save_run(runs, seed=43, train={"people": 10, "photos": 49})

    with pytest.raises(SystemExit, match="train"):
        run_main(tmp_path)


@pytest.mark.parametrize("option, value", [
    ("epochs", 3), ("lr", 0.001), ("lr_schedule", "none"), ("batch_size", 64),
])
def test_different_shared_option_across_rows_gives_an_error(tmp_path, option, value):
    runs = tmp_path / "runs"
    save_run(runs, n=2)
    # extra_options come last in save_run, so they also overwrite epochs
    save_run(runs, n=5, extra_options={option: value})

    with pytest.raises(SystemExit, match=option):
        run_main(tmp_path)
    assert not (tmp_path / "tables" / "summary.csv").exists()


def test_different_option_across_losses_gives_an_error(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, loss="ce", epochs=20)
    save_run(runs, loss="arcface", epochs=3)

    with pytest.raises(SystemExit, match="epochs"):
        run_main(tmp_path)


def test_different_train_people_across_rows_gives_an_error(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, n=2)
    save_run(runs, n=5, train={"people": 11, "photos": 55})

    with pytest.raises(SystemExit, match="train"):
        run_main(tmp_path)


def test_different_val_across_rows_gives_an_error(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, n=2)
    save_run(runs, n=5, val={"people": 2, "photos": 9})

    with pytest.raises(SystemExit, match="val"):
        run_main(tmp_path)


def test_train_photos_may_differ_across_rows(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, n=2, train={"people": 10, "photos": 20})
    save_run(runs, n=5, train={"people": 10, "photos": 49})

    assert len(run_main(tmp_path)) == 2


def test_different_margin_within_one_loss_gives_an_error(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, loss="arcface", n=2, extra_options={"margin": 0.5, "scale": 64.0})
    save_run(runs, loss="arcface", n=5, extra_options={"margin": 0.3, "scale": 64.0})

    with pytest.raises(SystemExit, match="margin"):
        run_main(tmp_path)


def test_margin_may_differ_between_losses(tmp_path):
    # ce has no margin and no scale, arcface has them
    runs = tmp_path / "runs"
    save_run(runs, loss="ce", n=2)
    save_run(runs, loss="arcface", n=2, extra_options={"margin": 0.5, "scale": 64.0})

    assert len(run_main(tmp_path)) == 2


def test_unknown_n_gives_an_error(tmp_path):
    save_run(tmp_path / "runs", n=7)

    with pytest.raises(SystemExit) as error:
        run_main(tmp_path)
    assert "ce_n7_seed42_ep20" in str(error.value)
    assert not (tmp_path / "tables" / "summary.csv").exists()


def test_summarize_does_not_load_torch():
    # N_VALUES comes from dataset.py, this breaks if dataset.py starts to need torch
    code = "import sys, whoisit.summarize; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_too_few_seeds_give_a_warning_and_the_table(tmp_path, capsys):
    runs = tmp_path / "runs"
    options = {"margin": 0.5, "scale": 64.0}
    save_run(runs, loss="arcface", n=5, seed=42, extra_options=options)
    save_run(runs, loss="arcface", n=5, seed=43, extra_options=options)
    save_run(runs, loss="arcface", n=2, seed=42, extra_options=options)
    for seed in [42, 43, 44]:
        save_run(runs, loss="arcface", n=10, seed=seed, extra_options=options)

    rows = run_main(tmp_path)
    assert [row["seeds"] for row in rows] == ["1", "2", "3"]
    lines = capsys.readouterr().out.splitlines()
    assert "warning: arcface n=5 has only 2 seeds" in lines
    assert "warning: arcface n=2 has only 1 seed" in lines
    assert not any("n=10" in line for line in lines)


def test_test_rows_mean_min_max(tmp_path):
    runs = tmp_path / "runs"
    for seed, top1, top5 in [(42, 0.2561, 0.4), (43, 0.2, 0.31), (44, 0.3, 0.55)]:
        save_test(save_run(runs, seed=seed, top1=0.5, top5=0.7), top1=top1, top5=top5)

    rows = run_main(tmp_path)
    assert [row["split"] for row in rows] == ["val", "test"]
    assert rows[1] == {
        "split": "test", "loss": "ce", "n": "5", "seeds": "3",
        "top1_mean": "25.20", "top1_min": "20.00", "top1_max": "30.00",
        "top5_mean": "42.00", "top5_min": "31.00", "top5_max": "55.00",
    }
    # the val row still comes from result.json
    assert rows[0]["top1_mean"] == "50.00"


def test_sorted_by_split_then_loss_then_n(tmp_path):
    runs = tmp_path / "runs"
    for loss, n in [("ce", 10), ("ce", 2), ("arcface", 20), ("arcface", 5)]:
        options = {"margin": 0.5, "scale": 64.0} if loss == "arcface" else None
        save_test(save_run(runs, loss=loss, n=n, extra_options=options))

    rows = run_main(tmp_path)
    assert [(row["split"], row["loss"], row["n"]) for row in rows] == [
        ("val", "arcface", "5"), ("val", "arcface", "20"), ("val", "ce", "2"), ("val", "ce", "10"),
        ("test", "arcface", "5"), ("test", "arcface", "20"), ("test", "ce", "2"), ("test", "ce", "10"),
    ]


def test_val_rows_stay_the_same_with_test_rows(tmp_path):
    runs = tmp_path / "runs"
    folders = [save_run(runs, n=n, seed=seed, top1=seed / 100)
               for n in [2, 5] for seed in [42, 43, 44]]
    before = tmp_path / "before.csv"
    summarize.main(["--runs-dir", str(runs), "--out", str(before)])

    for folder in folders:
        save_test(folder)
    rows = run_main(tmp_path)
    val_lines = (tmp_path / "tables" / "summary.csv").read_text(encoding="utf-8").splitlines()[:3]
    assert val_lines == before.read_text(encoding="utf-8").splitlines()
    assert [row["split"] for row in rows] == ["val", "val", "test", "test"]


def test_no_test_json_gives_no_test_rows(tmp_path):
    save_run(tmp_path / "runs")
    assert [row["split"] for row in run_main(tmp_path)] == ["val"]


def test_run_without_test_json_has_only_a_val_row(tmp_path, capsys):
    runs = tmp_path / "runs"
    save_test(save_run(runs, seed=42), top1=0.2)
    save_test(save_run(runs, seed=43), top1=0.4)
    save_run(runs, seed=44)

    rows = run_main(tmp_path)
    assert [(row["split"], row["seeds"]) for row in rows] == [("val", "3"), ("test", "2")]
    assert rows[1]["top1_mean"] == "30.00"
    lines = capsys.readouterr().out.splitlines()
    assert "warning: ce n=5 has only 2 seeds in test" in lines
    assert not any(line.startswith("warning") and "in test" not in line for line in lines)


def test_test_json_without_result_json_is_skipped(tmp_path):
    runs = tmp_path / "runs"
    save_run(runs, seed=42)
    unfinished = runs / "ce_n5_seed43_ep20"
    unfinished.mkdir()
    save_test(unfinished)

    assert [row["split"] for row in run_main(tmp_path)] == ["val"]


@pytest.mark.parametrize("same_row", [True, False])
def test_different_test_block_gives_an_error(tmp_path, same_row):
    runs = tmp_path / "runs"
    save_test(save_run(runs, n=5, seed=42))
    other = save_run(runs, n=5 if same_row else 10, seed=43)
    save_test(other, test={"people": 3, "photos": 10})

    with pytest.raises(SystemExit, match="test") as error:
        run_main(tmp_path)
    assert "photos': 10" in str(error.value)
    assert not (tmp_path / "tables" / "summary.csv").exists()


def test_no_results_give_an_error(tmp_path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(SystemExit, match="no result.json"):
        run_main(tmp_path)


def test_table_is_printed(tmp_path, capsys):
    save_run(tmp_path / "runs", top1=0.3561)
    run_main(tmp_path)
    output = capsys.readouterr().out
    assert "top1_mean" in output
    assert "35.61" in output
