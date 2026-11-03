"""Pick N training photos for each person.

A smaller N is always part of a bigger N, so the results for
2, 5, 10 and 20 photos can be compared.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from whoisit.data.splits import SEED, load_identities

TRAIN_PEOPLE_FILE = Path("splits/train_people.txt")
N_VALUES = (2, 5, 10, 20)


def load_train_people(path=TRAIN_PEOPLE_FILE):
    return np.loadtxt(path, dtype=int)


def pick_train_photos(df, people, n, seed=SEED):
    images = df.groupby("person")["image"].apply(sorted)
    rows = []
    for person in people:
        # every person gets one fixed order, n only decides how many we take
        rng = np.random.default_rng([seed, int(person)])
        order = rng.permutation(images[person])
        if len(order) < n:
            raise ValueError(f"person {person} has only {len(order)} photos")
        for image in sorted(order[:n]):
            rows.append((image, person))
    return pd.DataFrame(rows, columns=["image", "person"])


def main():
    df = load_identities()
    people = load_train_people()
    for n in N_VALUES:
        photos = pick_train_photos(df, people, n)
        print(f"N = {n}: {len(people)} people, {len(photos)} photos")


if __name__ == "__main__":
    main()