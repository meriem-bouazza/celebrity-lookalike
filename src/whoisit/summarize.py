"""Put the results of all finished runs in one table, one row per split, loss and N.

The val rows come from result.json, the test rows from test.json (written by final_eval.py).

Example: python -m whoisit.summarize
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from whoisit.data.dataset import N_VALUES

# not imported from train.py, so this module does not need torch
RUNS_DIR = Path("results/runs")
RESULT_FILE = "result.json"
TEST_FILE = "test.json"
OUT_FILE = Path("results/tables/summary.csv")
COLUMNS = ["split", "loss", "n", "seeds", "top1_mean", "top1_min", "top1_max",
           "top5_mean", "top5_min", "top5_max"]
SHARED_OPTIONS = ["epochs", "lr", "lr_schedule", "batch_size"]
LOSS_OPTIONS = ["margin", "scale"]
MIN_SEEDS = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    return parser.parse_args(argv)


def load_runs(runs_dir):
    # a folder without result.json is not matched: the run is not finished
    return {path.parent.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(runs_dir.glob(f"*/{RESULT_FILE}"))}


def load_test_runs(runs_dir, runs):
    # only runs with a result.json; a run without test.json just has no test row
    test_runs = {}
    for name, run in runs.items():
        path = runs_dir / name / TEST_FILE
        if path.exists():
            test = json.loads(path.read_text(encoding="utf-8"))
            # keep the options of result.json, but take the numbers from test.json
            test_runs[name] = {**run, "test": test["test"], "final": test["final"]}
    return test_runs


def settings(run):
    # everything that must be the same in one row, so all but the seed
    options = {key: value for key, value in run["options"].items() if key != "seed"}
    return {**options, "train": run.get("train"), "val": run.get("val"), "test": run.get("test")}


def shared_settings(run):
    # what must be the same in every run, also for different losses and n
    shared = {key: run["options"].get(key) for key in SHARED_OPTIONS}
    # the photo count of train changes with n, so it is not part of this
    shared["train"] = {key: value for key, value in run.get("train", {}).items()
                       if key != "photos"}
    shared["val"] = run.get("val")
    # only test runs have this block, so all test rows must use the same test photos
    shared["test"] = run.get("test")
    return shared


def loss_settings(run):
    return {key: run["options"].get(key) for key in LOSS_OPTIONS}


def check_same(runs, get_settings, title):
    first_name, first_run = next(iter(runs.items()))
    first = get_settings(first_run)
    problems = []
    for name, run in runs.items():
        other = get_settings(run)
        # go over the keys of both runs, so a missing option also counts as different
        for key in sorted(first.keys() | other.keys()):
            if first.get(key) != other.get(key):
                problems.append(f"  {key}: {first.get(key)} in {first_name}, "
                                f"{other.get(key)} in {name}")
    if problems:
        raise ValueError(f"{title}:\n" + "\n".join(problems))


def check_n_values(runs):
    problems = [f"  n {run['options']['n']} in {name}" for name, run in runs.items()
                if run["options"]["n"] not in N_VALUES]
    if problems:
        raise ValueError(f"n must be one of {N_VALUES}:\n" + "\n".join(problems))


def percent(value):
    return f"{value * 100:.2f}"


def make_row(split, loss, n, runs):
    row = {"split": split, "loss": loss, "n": n, "seeds": len(runs)}
    for metric in ["top1", "top5"]:
        # the mean comes from the exact numbers, we round only at the end
        values = [run["final"][metric] for run in runs.values()]
        row[f"{metric}_mean"] = percent(fmean(values))
        row[f"{metric}_min"] = percent(min(values))
        row[f"{metric}_max"] = percent(max(values))
    return row


def summarize_split(split, runs):
    check_n_values(runs)

    groups = defaultdict(dict)
    by_loss = defaultdict(dict)
    for name, run in runs.items():
        groups[run["options"]["loss"], run["options"]["n"]][name] = run
        by_loss[run["options"]["loss"]][name] = run

    rows = []
    # n is a number in result.json, so 2 comes before 10
    for (loss, n), group in sorted(groups.items()):
        check_same(group, settings,
                   f"{split} runs with loss {loss} and n {n} differ in more than the seed")
        rows.append(make_row(split, loss, n, group))

    check_same(runs, shared_settings, f"{split} runs in the table differ")
    for loss, group in by_loss.items():
        check_same(group, loss_settings, f"{split} runs with loss {loss} differ")
    return rows


def summarize(runs, test_runs=None):
    # val first and test after it, so the val rows stay the same when test rows come in
    rows = summarize_split("val", runs)
    if test_runs:
        rows += summarize_split("test", test_runs)
    return rows


def seed_warnings(rows):
    # the val warning stays as before, the test warning says that it is about test
    return [f"warning: {row['loss']} n={row['n']} has only {row['seeds']} "
            f"seed{'' if row['seeds'] == 1 else 's'}"
            f"{'' if row['split'] == 'val' else ' in ' + row['split']}"
            for row in rows if row["seeds"] < MIN_SEEDS]


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        # always \n, so the file is the same on Windows and Linux
        writer = csv.DictWriter(file, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    table = [COLUMNS] + [[str(row[column]) for column in COLUMNS] for row in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(COLUMNS))]
    for line in table:
        print("  ".join(cell.rjust(width) for cell, width in zip(line, widths)))


def main(argv=None):
    args = parse_args(argv)
    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"error: no {RESULT_FILE} found in {args.runs_dir}")
    try:
        rows = summarize(runs, load_test_runs(args.runs_dir, runs))
    except ValueError as error:
        raise SystemExit(f"error: {error}")
    save_csv(rows, args.out)
    print_table(rows)
    for warning in seed_warnings(rows):
        print(warning)
    print("saved", args.out)


if __name__ == "__main__":
    main()
