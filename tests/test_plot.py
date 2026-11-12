"""Test for whoisit.plot with made-up numbers (both losses, only cross-entropy, no baseline)."""
import pytest

from whoisit.plot import load_baseline, main

HEADER = "split,loss,n,seeds,top1_mean,top1_min,top1_max,top5_mean,top5_min,top5_max\n"


def write_summary(path, losses):
    lines = [f"val,{loss},{n},3,{n * 2},{n * 2 - 1},{n * 2 + 1},{n * 3},{n * 3 - 1},{n * 3 + 1}\n"
             for loss in losses for n in (2, 5, 10, 20)]
    path.write_text(HEADER + "".join(lines), encoding="utf-8")


@pytest.mark.parametrize("losses", [["ce", "arcface"], ["ce"]])
def test_plot_saves_png(tmp_path, losses):
    write_summary(tmp_path / "summary.csv", losses)
    (tmp_path / "baseline.txt").write_text("top 1: 12.22%\ntop 5: 22.57%\n", encoding="utf-8")
    out = tmp_path / "plot.png"
    main(["--summary", str(tmp_path / "summary.csv"), "--baseline", str(tmp_path / "baseline.txt"),
          "--out", str(out)])
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_baseline_numbers_and_missing_split(tmp_path):
    (tmp_path / "baseline.txt").write_text("top 1: 12.22%\ntop 5: 22.57%\n", encoding="utf-8")
    assert load_baseline(tmp_path / "baseline.txt") == {"top1": 12.22, "top5": 22.57}
    write_summary(tmp_path / "summary.csv", ["ce"])
    with pytest.raises(SystemExit):
        main(["--summary", str(tmp_path / "summary.csv"), "--split", "test", "--out", str(tmp_path / "x.png")])
