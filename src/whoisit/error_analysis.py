"""Top 1 on the test people with and without each attribute (glasses, hat, ...).

For every run we embed all test photos once. Every photo is a query, and the gallery is
always all other test photos. Only the query photos are split by attribute.

Example (in the container): python -m whoisit.error_analysis
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from whoisit.attributes import ATTR_FILE, load_attributes, load_test_sources, photo_attributes
from whoisit.data.images import FaceDataset
from whoisit.evaluate import top_k
from whoisit.final_eval import ALIGNED_DIR, CHECKPOINT, TEST_FILE, load_model
from whoisit.summarize import (RESULT_FILE, RUNS_DIR, check_n_values, check_same, load_runs,
                               loss_settings, percent, settings, shared_settings)

# chosen in docs/decisions.md, "2026-11-16: Attributes for the error analysis"
ATTRIBUTES = ["Eyeglasses", "Wearing_Hat", "Blurry", "Smiling", "Male"]
OUT_RUNS = Path("results/tables/error_analysis_runs.csv")
OUT_MEAN = Path("results/tables/error_analysis.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--attr-file", type=Path, default=ATTR_FILE)
    parser.add_argument("--aligned-dir", type=Path, default=ALIGNED_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-runs", type=Path, default=OUT_RUNS)
    parser.add_argument("--out", type=Path, default=OUT_MEAN)
    return parser.parse_args(argv)


@torch.no_grad()
def embed(model, loader, device):
    # like validate() in train.py, but we need the embeddings, not only the accuracy
    model.eval()
    return np.concatenate([model(images.to(device)).cpu().numpy() for images, _ in loader])


def top1_hits(embeddings, labels):
    # True if the most similar other photo is the same person; the mean is Top 1 of evaluate.py
    labels = np.asarray(labels)
    # normalize like top_k_accuracy, so the neighbours are exactly the same
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return labels[top_k(normed, 1)[:, 0]] == labels


def split_by_attribute(hits, has):
    # has: one row per photo, one column per attribute
    rows = []
    for attribute in has.columns:
        mask = has[attribute].to_numpy()
        if mask.all() or not mask.any():
            raise ValueError(f"all or no test photos have {attribute}, so there is nothing to compare")
        rows.append({"attribute": attribute,
                     "with_photos": int(mask.sum()), "with_top1": float(hits[mask].mean()),
                     "without_photos": int((~mask).sum()), "without_top1": float(hits[~mask].mean())})
    return rows


def check_runs(runs):
    # the same checks as summarize.py, so we never mix runs with other settings
    check_n_values(runs)
    groups, by_loss = {}, {}
    for name, run in runs.items():
        groups.setdefault((run["options"]["loss"], run["options"]["n"]), {})[name] = run
        by_loss.setdefault(run["options"]["loss"], {})[name] = run
    for (loss, n), group in groups.items():
        check_same(group, settings, f"runs with loss {loss} and n {n} differ in more than the seed")
    check_same(runs, shared_settings, "runs differ")
    for loss, group in by_loss.items():
        check_same(group, loss_settings, f"runs with loss {loss} differ")


def check_test_json(folder, hits, data):
    # the overall Top 1 must be the one in test.json (same photos, same model)
    path = folder / TEST_FILE
    if not path.exists():
        print(f"warning: {folder.name} has no {TEST_FILE}, cannot check the overall Top 1")
        return
    test = json.loads(path.read_text(encoding="utf-8"))
    if test["test"] != {"people": data.num_people, "photos": len(data)}:
        print(f"warning: {folder.name}: {TEST_FILE} was made with other photos ({test['test']})")
    # compare the number of right photos, so tiny float differences do not count
    expected = round(test["final"]["top1"] * len(hits))
    if expected != int(hits.sum()):
        print(f"warning: {folder.name}: {int(hits.sum())} right photos, but {TEST_FILE} "
              f"says {expected} (top 1 {hits.mean():.4%} instead of {test['final']['top1']:.4%})")


def sort_rows(table, columns):
    # loss, then n as a number (2 before 10), the attributes in the order of ATTRIBUTES
    order = {name: i for i, name in enumerate(ATTRIBUTES)}
    table = table.assign(order=table["attribute"].map(order))
    return table.sort_values([*columns, "order"]).drop(columns="order").reset_index(drop=True)


def run_table(rows):
    table = pd.DataFrame(rows)
    for column in ["with_top1", "without_top1"]:
        table[column] = table[column].map(percent)
    return sort_rows(table, ["loss", "n", "seed"])


def mean_table(rows):
    rows = pd.DataFrame(rows)
    rows["difference"] = rows["with_top1"] - rows["without_top1"]
    table = []
    for (loss, n, attribute), group in rows.groupby(["loss", "n", "attribute"], sort=False):
        table.append({
            "loss": loss, "n": n, "attribute": attribute, "seeds": len(group),
            # the photos are the same in every run, so one number is enough
            "with_photos": group["with_photos"].iloc[0],
            "without_photos": group["without_photos"].iloc[0],
            "top1_with": percent(group["with_top1"].mean()),
            "top1_without": percent(group["without_top1"].mean()),
            "difference_mean": percent(group["difference"].mean()),
            "difference_min": percent(group["difference"].min()),
            "difference_max": percent(group["difference"].max()),
        })
    return sort_rows(pd.DataFrame(table), ["loss", "n"])


def save_csv(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # always \n, so the file is the same on Windows and Linux
    table.to_csv(path, index=False, lineterminator="\n")


def main(argv=None):
    args = parse_args(argv)
    if not args.attr_file.exists():
        raise SystemExit(f"error: {args.attr_file} not found")
    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"error: no {RESULT_FILE} found in {args.runs_dir}")
    # all runs or none, so the mean table never has fewer seeds by mistake
    missing = [name for name in runs if not (args.runs_dir / name / CHECKPOINT).exists()]
    if missing:
        raise SystemExit(f"error: no checkpoint for {len(missing)} runs: {', '.join(missing)}")

    try:
        check_runs(runs)
        photos = load_test_sources(args.aligned_dir)
        attributes = load_attributes(args.attr_file)
        unknown = [name for name in ATTRIBUTES if name not in attributes.columns]
        if unknown:
            raise ValueError(f"not in {args.attr_file}: {', '.join(unknown)}")
        has = photo_attributes(photos, attributes)[ATTRIBUTES]
    except ValueError as error:
        raise SystemExit(f"error: {error}")

    # the same photos and preprocessing as final_eval.py
    data = FaceDataset(photos, image_dir=args.aligned_dir / "images", train=False)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    print(f"test: {data.num_people} people, {len(data)} photos, device {args.device}")

    rows = []
    for name, run in runs.items():
        folder = args.runs_dir / name
        model = load_model(folder / CHECKPOINT, args.device)
        hits = top1_hits(embed(model, loader, args.device), data.labels)
        check_test_json(folder, hits, data)
        print(f"{name}: top 1 {hits.mean():.2%}")
        info = {"run": name, "loss": run["options"]["loss"], "n": run["options"]["n"],
                "seed": run["options"]["seed"]}
        try:
            rows += [{**info, **row} for row in split_by_attribute(hits, has)]
        except ValueError as error:
            raise SystemExit(f"error: {error}")

    save_csv(run_table(rows), args.out_runs)
    table = mean_table(rows)
    save_csv(table, args.out)
    print(table.to_string(index=False))
    print("saved", args.out_runs)
    print("saved", args.out)


if __name__ == "__main__":
    main()
