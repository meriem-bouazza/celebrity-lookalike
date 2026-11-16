"""Top 1 and Top 5 of every finished run on the test people. Writes test.json next to result.json.

We look at the test people only once, at the end. So a run that already has a test.json is
skipped, unless --force is given.

Example (in the container): python -m whoisit.final_eval
"""
import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from whoisit.data.images import FaceDataset
from whoisit.models.backbone import Backbone
from whoisit.train import RESULT_FILE, RUNS_DIR, validate

# the same aligned test photos as scripts/baseline.py, so the numbers can be compared
ALIGNED_DIR = Path("data/aligned/celeba_test")
TEST_FILE = "test.json"
CHECKPOINT = Path("checkpoints/model.pt")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--aligned-dir", type=Path, default=ALIGNED_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    # in the container the shared memory (/dev/shm) is only 63 MB, more workers crash
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="evaluate again if test.json exists")
    return parser.parse_args(argv)


def load_test_photos(aligned_dir):
    # meta.csv is written by preprocess.py, only for photos where a face was found
    with open(aligned_dir / "meta.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    photos = pd.DataFrame({"image": [Path(row["image_path"]).name for row in rows],
                           "person": [row["identity"] for row in rows]})
    # like the baseline: a person with only one photo left can never be found
    counts = photos.groupby("person")["image"].transform("size")
    return photos[counts >= 2].reset_index(drop=True)


def load_model(checkpoint, device):
    # no ImageNet download, all weights come from the checkpoint
    model = Backbone(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.to(device)


def save_json(path, data):
    # write to a temp file first, so a crash never leaves half a file
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # a folder without result.json is skipped: the run is not finished
    folders = sorted(path.parent for path in args.runs_dir.glob(f"*/{RESULT_FILE}"))
    if not folders:
        raise SystemExit(f"error: no {RESULT_FILE} found in {args.runs_dir}")

    data = FaceDataset(load_test_photos(args.aligned_dir), image_dir=args.aligned_dir / "images",
                       train=False)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print(f"test: {data.num_people} people, {len(data)} photos, device {device}")

    missing = []
    for folder in folders:
        if (folder / TEST_FILE).exists() and not args.force:
            print(f"skip {folder.name}: {TEST_FILE} exists (use --force to run again)")
            continue
        if not (folder / CHECKPOINT).exists():
            print(f"skip {folder.name}: no {CHECKPOINT.as_posix()}")
            missing.append(folder.name)
            continue
        # the same code as the validation during training, only with the test photos
        top1, top5 = validate(load_model(folder / CHECKPOINT, device), loader, device)
        save_json(folder / TEST_FILE, {
            "checkpoint": CHECKPOINT.as_posix(),
            "test": {"people": data.num_people, "photos": len(data)},
            "final": {"top1": top1, "top5": top5},
        })
        print(f"{folder.name}: test top 1 {top1:.2%}, top 5 {top5:.2%}")

    if missing:
        raise SystemExit(f"error: no checkpoint for {len(missing)} runs: {', '.join(missing)}")


if __name__ == "__main__":
    main()
