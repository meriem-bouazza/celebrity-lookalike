"""Pick N training photos for each person.

A smaller N is always part of a bigger N, so the results for
2, 5, 10 and 20 photos can be compared.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from whoisit.data.splits import SEED, load_identities

TRAIN_PEOPLE_FILE = Path("splits/train_people.txt")
ALIGNED_META = Path("data/aligned/celeba/meta.csv")
N_VALUES = (2, 5, 10, 20)
POOL_SIZE = 25  # photos per person that we align, some of them can fail


def load_train_people(path=TRAIN_PEOPLE_FILE):
    return np.loadtxt(path, dtype=int)


def pick_train_photos(df, people, n, seed=SEED, strict=True):
    images = df.groupby("person")["image"].apply(sorted)
    rows = []
    for person in people:
        # every person gets one fixed order, n only decides how many we take
        rng = np.random.default_rng([seed, int(person)])
        order = rng.permutation(images[person])
        if strict and len(order) < n:
            raise ValueError(f"person {person} has only {len(order)} photos")
        for image in sorted(order[:n]):
            rows.append((image, person))
    return pd.DataFrame(rows, columns=["image", "person"])


def load_aligned(path=ALIGNED_META):
    # names of the photos that were aligned (preprocess.py writes this file)
    return set(pd.read_csv(path, dtype=str)["source_path"])


def pick_aligned_photos(df, people, n, aligned, seed=SEED):
    # pool: up to POOL_SIZE photos per person, without the ones where no face was found
    pool = pick_train_photos(df, people, POOL_SIZE, seed, strict=False)
    pool = pool[pool["image"].isin(aligned)]
    # people with too few aligned photos are left out for every N
    counts = pool.groupby("person").size()
    keep = counts[counts >= max(N_VALUES)].index
    pool = pool[pool["person"].isin(keep)]
    return pick_train_photos(pool, keep, n, seed)


def main():
    df = load_identities()
    people = load_train_people()
    aligned = load_aligned()
    for n in N_VALUES:
        photos = pick_aligned_photos(df, people, n, aligned)
        print(f"N = {n}: {photos['person'].nunique()} people, {len(photos)} photos")


if __name__ == "__main__":
    main()