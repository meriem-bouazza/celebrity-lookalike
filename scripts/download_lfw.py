"""Download LFW (Labeled Faces in the Wild) for development and tests, and write a metadata CSV.

LFW is only used to develop and test the pipeline (real faces with identities) until the
project dataset is chosen; it is not used for the experiments. Source: http://vis-www.cs.umass.edu/lfw/
(downloaded through scikit-learn). Nothing is written into the repository except under data/,
which is not versioned.

Usage:
    python scripts/download_lfw.py --data-home data/sklearn --out data/lfw_meta.csv
"""

import argparse
import csv
from pathlib import Path

from sklearn.datasets import fetch_lfw_people


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-home", type=Path, default=Path("data/sklearn"), help="download folder")
    parser.add_argument("--out", type=Path, default=Path("data/lfw_meta.csv"), help="metadata CSV to write")
    args = parser.parse_args()

    # Only triggers the download/extraction; the returned arrays are not used.
    fetch_lfw_people(data_home=str(args.data_home), min_faces_per_person=20, download_if_missing=True)
    image_dir = args.data_home / "lfw_home" / "lfw_funneled"

    rows = [
        (path.as_posix(), path.parent.name)
        for path in sorted(image_dir.glob("*/*.jpg"))
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "identity"])
        writer.writerows(rows)
    print(f"{len(rows)} images of {len({r[1] for r in rows})} identities -> {args.out}")


if __name__ == "__main__":
    main()
