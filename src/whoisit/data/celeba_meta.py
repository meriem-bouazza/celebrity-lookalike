"""Write the list of CelebA photos that we align (input for preprocess.py)."""
from pathlib import Path

import pandas as pd

from whoisit.data.dataset import POOL_SIZE, load_train_people, pick_train_photos
from whoisit.data.splits import OUT_DIR, load_identities

OUT_FILE = Path("data/celeba_meta.csv")


def main():
    train = pick_train_photos(load_identities(), load_train_people(), POOL_SIZE, strict=False)
    val = pd.read_csv(OUT_DIR / "val_photos.csv")
    test = pd.read_csv(OUT_DIR / "test_photos.csv")
    photos = pd.concat([train, val, test])
    # preprocess.py wants these column names
    photos = photos.rename(columns={"image": "image_path", "person": "identity"})
    photos.to_csv(OUT_FILE, index=False, lineterminator="\n")
    print(len(train), "train,", len(val), "val,", len(test), "test photos")


if __name__ == "__main__":
    main()
