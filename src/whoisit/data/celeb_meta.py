"""Collect the raw photos for the new demo gallery (Pins celebrities + chosen LFW people) in one folder.

The container has neither the Pins photos nor LFW funneled, so we copy only the chosen photos to
data/celeb_raw/<name>/<file>.jpg and write data/celeb_raw/meta.csv (image_path,identity) for
preprocess.py. Then only this folder has to be packed and uploaded.

The names come from configs/celeb_gallery_pins.csv (Pins folder -> name) and
configs/celeb_gallery_lfw.txt (one LFW folder name per line).

Example: python -m whoisit.data.celeb_meta
"""
import argparse
import csv
import shutil
from pathlib import Path

PINS_DIR = Path("data/pins/105_classes_pins_dataset")
LFW_DIR = Path("data/sklearn/lfw_home/lfw_funneled")
PINS_NAMES = Path("configs/celeb_gallery_pins.csv")
LFW_NAMES = Path("configs/celeb_gallery_lfw.txt")
OUT_DIR = Path("data/celeb_raw")
# the gallery keeps 5 photos per person; 10 leave room for photos where no face is found
MAX_PHOTOS = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pins-dir", type=Path, default=PINS_DIR)
    parser.add_argument("--lfw-dir", type=Path, default=LFW_DIR)
    parser.add_argument("--pins-names", type=Path, default=PINS_NAMES)
    parser.add_argument("--lfw-names", type=Path, default=LFW_NAMES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-photos", type=int, default=MAX_PHOTOS, help="per person")
    return parser.parse_args(argv)


def load_pins_names(path, pins_dir):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    names = {}
    for row in rows:
        if row["folder"] in names:
            raise ValueError(f"Pins folder '{row['folder']}' is twice in {path}")
        names[row["folder"]] = row["name"]
    folders = {path.name for path in pins_dir.iterdir() if path.is_dir()}
    if folders - names.keys():
        raise ValueError(f"Pins folders missing in {path}: {', '.join(sorted(folders - names.keys()))}")
    if names.keys() - folders:
        raise ValueError(f"folders in {path} not found in {pins_dir}: "
                         f"{', '.join(sorted(names.keys() - folders))}")
    return names


def load_lfw_names(path, lfw_dir):
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [name for name in names if not (lfw_dir / name).is_dir()]
    if missing:
        raise ValueError(f"LFW names in {path} that are not a folder in {lfw_dir}: {', '.join(missing)}")
    return names


def check_unique(names):
    seen, twice = set(), set()
    for name in names:
        (twice if name in seen else seen).add(name)
    if twice:
        raise ValueError(f"names that appear twice: {', '.join(sorted(twice))}")


def first_photos(folder, max_photos):
    return sorted((path for path in folder.iterdir() if path.suffix.lower() == ".jpg"),
                  key=lambda path: path.name)[:max_photos]


def collect(people, out_dir, max_photos):
    # people: list of (source folder, name); returns the rows for meta.csv
    rows = []
    for folder, name in people:
        photos = first_photos(folder, max_photos)
        if not photos:
            raise ValueError(f"no .jpg photos in {folder}")
        (out_dir / name).mkdir(parents=True)
        for photo in photos:
            # Pins file names have spaces, for example "Adriana Lima0_0.jpg"
            target = out_dir / name / photo.name.replace(" ", "_")
            if target.exists():
                raise ValueError(f"two photos of {name} get the same name {target.name}")
            shutil.copyfile(photo, target)
            rows.append((target.relative_to(out_dir).as_posix(), name))
    return rows


def main(argv=None):
    args = parse_args(argv)
    # a new run starts empty, so no old photos stay in the folder by mistake
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"error: {args.out_dir} is not empty, delete it first")
    try:
        pins = load_pins_names(args.pins_names, args.pins_dir)
        lfw = load_lfw_names(args.lfw_names, args.lfw_dir)
        check_unique([*pins.values(), *lfw])
        pins_people = [(args.pins_dir / folder, name) for folder, name in sorted(pins.items(), key=lambda x: x[1])]
        lfw_people = [(args.lfw_dir / name, name) for name in sorted(lfw)]
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pins_rows = collect(pins_people, args.out_dir, args.max_photos)
        lfw_rows = collect(lfw_people, args.out_dir, args.max_photos)
    except ValueError as error:
        raise SystemExit(f"error: {error}")

    with open(args.out_dir / "meta.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["image_path", "identity"])
        writer.writerows(sorted(pins_rows + lfw_rows))

    for source, people, rows in [("pins", pins_people, pins_rows), ("lfw", lfw_people, lfw_rows)]:
        print(f"{source}: {len(people)} people, {len(rows)} photos")
    print(f"total: {len(pins_people) + len(lfw_people)} people, {len(pins_rows) + len(lfw_rows)} photos")
    print("saved", args.out_dir / "meta.csv")


if __name__ == "__main__":
    main()
