"""Split the CelebA people into train, validation and test."""

from pathlib import Path

import numpy as np
import pandas as pd

IDENTITY_FILE = Path("data/celeba/identity_CelebA.txt")
OUT_DIR = Path("splits")

SEED = 42
MIN_PHOTOS = 20
N_TRAIN, N_VAL, N_TEST = 2000, 500, 1000
PHOTOS_PER_PERSON = 10  # photos we keep for each val and test person


def load_identities(path=IDENTITY_FILE):
    # one line per photo: photo name, person id
    return pd.read_csv(path, sep=" ", header=None, names=["image", "person"])


def make_split(df, seed=SEED, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
               min_photos=MIN_PHOTOS):
    # only people with enough photos
    counts = df.groupby("person").size()
    people = counts[counts >= min_photos].index.to_numpy()
    people = np.sort(people)
    if len(people) < n_train + n_val + n_test:
        raise ValueError(f"only {len(people)} people with {min_photos}+ photos")

    # shuffle with a fixed seed and cut into three parts
    rng = np.random.default_rng(seed)
    rng.shuffle(people)
    train = people[:n_train]
    val = people[n_train:n_train + n_val]
    test = people[n_train + n_val:n_train + n_val + n_test]
    return np.sort(train), np.sort(val), np.sort(test)


def pick_photos(df, people, n=PHOTOS_PER_PERSON, seed=SEED):
    # choose n photos for each person
    rng = np.random.default_rng(seed)
    images = df.groupby("person")["image"].apply(sorted)
    rows = []
    for person in people:
        chosen = rng.choice(images[person], size=n, replace=False)
        for image in sorted(chosen):
            rows.append((image, person))
    return pd.DataFrame(rows, columns=["image", "person"])


def main():
    df = load_identities()
    train, val, test = make_split(df)
    OUT_DIR.mkdir(exist_ok=True)

    for name, people in [("train", train), ("val", val), ("test", test)]:
        np.savetxt(OUT_DIR / f"{name}_people.txt", people, fmt="%d")
        print(name, len(people), "people")

    for name, people in [("val", val), ("test", test)]:
        photos = pick_photos(df, people)
        photos.to_csv(OUT_DIR / f"{name}_photos.csv", index=False)
        print(name, len(photos), "photos")


if __name__ == "__main__":
    main()