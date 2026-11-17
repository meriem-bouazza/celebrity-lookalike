"""Count how many test photos and test people have each CelebA attribute (glasses, hat, ...).

This only counts, it does not evaluate a model. Later the counts tell us which groups are
big enough for the error analysis.

Example (in the container): python -m whoisit.attributes
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

from whoisit.final_eval import ALIGNED_DIR, load_test_photos

ATTR_FILE = Path("data/celeba/list_attr_celeba.txt")
OUT_FILE = Path("results/tables/test_attributes.csv")
# a person needs 2 photos with the attribute, like in our Top 1
MIN_PHOTOS = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr-file", type=Path, default=ATTR_FILE)
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    parser.add_argument("--aligned-dir", type=Path, default=ALIGNED_DIR)
    return parser.parse_args(argv)


def load_attributes(path):
    # line 1: number of photos, line 2: the 40 names, then: file name and 40 values (1 or -1)
    lines = path.read_text(encoding="utf-8").splitlines()
    names = lines[1].split()
    values = {}
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        if len(parts) != len(names) + 1 or any(value not in ("1", "-1") for value in parts[1:]):
            raise ValueError(f"bad line in {path}: {line.strip()}")
        values[parts[0]] = [value == "1" for value in parts[1:]]
    if len(values) != int(lines[0]):
        raise ValueError(f"{path} says {lines[0].strip()} photos, but has {len(values)}")
    return pd.DataFrame.from_dict(values, orient="index", columns=names)


def load_test_sources(aligned_dir):
    # the same test photos as final_eval.py, plus the original CelebA file name
    photos = load_test_photos(aligned_dir)
    with open(aligned_dir / "meta.csv", newline="", encoding="utf-8") as file:
        source = {(row["identity"], Path(row["image_path"]).name): Path(row["source_path"]).name
                  for row in csv.DictReader(file)}
    photos["source"] = [source[person, image] for person, image in zip(photos["person"], photos["image"])]
    return photos


def photo_attributes(photos, attributes):
    # one row per test photo (same order as photos), True if the photo has the attribute
    missing = sorted(set(photos["source"]) - set(attributes.index))
    if missing:
        raise ValueError(f"{len(missing)} test photos are not in the attribute file, "
                         f"for example: {', '.join(missing[:5])}")
    return attributes.loc[photos["source"]].reset_index(drop=True)


def count_attributes(photos, attributes):
    has = photo_attributes(photos, attributes)
    per_person = has.groupby(photos["person"].to_numpy()).sum()
    table = pd.DataFrame({
        "attribute": has.columns,
        "photos": has.sum().to_numpy(),
        "percent": [f"{count / len(has) * 100:.2f}" for count in has.sum()],
        "people": (per_person >= MIN_PHOTOS).sum().to_numpy(),
    })
    # biggest first, the same numbers in the order of the name
    return table.sort_values(["photos", "attribute"], ascending=[False, True]).reset_index(drop=True)


def main(argv=None):
    args = parse_args(argv)
    if not args.attr_file.exists():
        raise SystemExit(f"error: {args.attr_file} not found")
    try:
        photos = load_test_sources(args.aligned_dir)
        table = count_attributes(photos, load_attributes(args.attr_file))
    except ValueError as error:
        raise SystemExit(f"error: {error}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # always \n, so the file is the same on Windows and Linux
    table.to_csv(args.out, index=False, lineterminator="\n")
    print(f"test: {photos['person'].nunique()} people, {len(photos)} photos")
    print(table.to_string(index=False))
    print("saved", args.out)


if __name__ == "__main__":
    main()
