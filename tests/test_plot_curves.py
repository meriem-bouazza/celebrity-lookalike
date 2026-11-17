"""Tests for the validation curves, with small made-up result.json files."""
import json

import numpy as np
import pytest

from whoisit import plot_curves

# top1 after each epoch, one list per seed
CE = {42: [0.10, 0.20, 0.30, 0.40], 43: [0.12, 0.22, 0.32, 0.38], 44: [0.11, 0.21, 0.31, 0.42]}
ARCFACE = {42: [0.15, 0.35, 0.30, 0.25], 43: [0.16, 0.33, 0.31, 0.27], 44: [0.14, 0.34, 0.29, 0.26]}


def save_run(runs_dir, loss, n, seed, top1, extra_options=None):
    folder = runs_dir / f"{loss}_n{n}_seed{seed}_ep{len(top1)}"
    folder.mkdir(parents=True)
    options = {"loss": loss, "n": n, "seed": seed, "epochs": len(top1), **(extra_options or {})}
    epochs = [{"epoch": i + 1, "top1": value, "top5": value} for i, value in enumerate(top1)]
    result = {"options": options, "epochs": epochs, "final": {"top1": top1[-1], "top5": top1[-1]}}
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")


def save_all(runs_dir, ns=(2, 20)):
    for n in ns:
        for seed in CE:
            save_run(runs_dir, "ce", n, seed, CE[seed])
            save_run(runs_dir, "arcface", n, seed, ARCFACE[seed])


def run_main(tmp_path, *extra):
    out = tmp_path / "figures" / "val_curves.png"
    plot_curves.main(["--runs-dir", str(tmp_path / "runs"), "--out", str(out), *extra])
    return out


def test_png_is_written(tmp_path, capsys):
    save_all(tmp_path / "runs")
    out = run_main(tmp_path)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "saved" in capsys.readouterr().out


def test_mean_low_high(tmp_path):
    save_all(tmp_path / "runs")
    line = plot_curves.make_line(plot_curves.load_runs(tmp_path / "runs"), "ce", 2)
    assert line["seeds"] == 3
    assert list(line["epochs"]) == [1, 2, 3, 4]
    assert np.allclose(line["mean"], [11, 21, 31, 40])
    assert np.allclose(line["low"], [10, 20, 30, 38])
    assert np.allclose(line["high"], [12, 22, 32, 42])


def test_best_epoch(tmp_path, capsys):
    save_all(tmp_path / "runs")
    runs = plot_curves.load_runs(tmp_path / "runs")
    # cross-entropy is best at the end, ArcFace at epoch 2 (mean 34 %)
    assert plot_curves.make_line(runs, "ce", 2)["best_epoch"] == 4
    arcface = plot_curves.make_line(runs, "arcface", 2)
    assert arcface["best_epoch"] == 2
    assert arcface["best_top1"] == pytest.approx(34)

    run_main(tmp_path)
    assert "arcface n=2: best epoch 2 (34.00 %)" in capsys.readouterr().out


def test_first_epoch_wins_a_tie(tmp_path):
    for seed in (42, 43):
        save_run(tmp_path / "runs", "ce", 2, seed, [0.1, 0.3, 0.3, 0.2])
    line = plot_curves.make_line(plot_curves.load_runs(tmp_path / "runs"), "ce", 2)
    assert line["best_epoch"] == 2


def test_other_n_values(tmp_path):
    save_all(tmp_path / "runs", ns=(5,))
    assert run_main(tmp_path, "--n", "5").exists()


def test_missing_loss_and_n_gives_an_error(tmp_path):
    save_all(tmp_path / "runs", ns=(2,))
    with pytest.raises(SystemExit, match="loss ce and n 20"):
        run_main(tmp_path)
    assert not (tmp_path / "figures").exists()


def test_different_number_of_epochs_gives_an_error(tmp_path):
    save_all(tmp_path / "runs")
    # a short test run in the same line
    save_run(tmp_path / "runs", "arcface", 2, 45, [0.1, 0.2])
    with pytest.raises(SystemExit, match="different number of epochs") as error:
        run_main(tmp_path)
    assert "arcface_n2_seed45_ep2 has 2" in str(error.value)
    assert not (tmp_path / "figures").exists()


def test_other_settings_in_one_line_give_an_error(tmp_path):
    save_all(tmp_path / "runs")
    save_run(tmp_path / "runs", "arcface", 20, 45, ARCFACE[42], extra_options={"margin": 0.3})
    with pytest.raises(SystemExit, match="margin"):
        run_main(tmp_path)


def test_no_results_give_an_error(tmp_path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(SystemExit, match="no result.json"):
        run_main(tmp_path)
