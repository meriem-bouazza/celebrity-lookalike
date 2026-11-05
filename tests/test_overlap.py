"""Tests for scripts/overlap.py on made-up embeddings."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from overlap import calibrate, likely_matches  # noqa: E402


def people(directions, photos=4, noise=0.05, seed=0):
    # photos of each person = their direction plus a little noise
    rng = np.random.default_rng(seed)
    embeddings = np.repeat(directions, photos, axis=0) + noise * rng.normal(size=(len(directions) * photos, 64))
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def test_shared_person_is_found_and_others_are_not():
    rng = np.random.default_rng(1)
    shared, celeba_only, lfw_only = rng.normal(size=(3, 64))
    celeba = people(np.stack([shared, celeba_only]))
    lfw = people(np.stack([shared, lfw_only]), seed=2)
    matches = likely_matches(celeba, np.repeat(["101", "102"], 4), lfw, np.repeat(["Ann", "Bob"], 4), 0.8)
    assert [(a, b) for a, b, _ in matches] == [("101", "Ann")]


def test_calibrate_counts_wrong_pairs_and_found_people():
    rng = np.random.default_rng(3)
    d = rng.normal(size=(3, 64))
    d[2] = d[0]  # "Sepp" and "Joseph" are the same face under two names
    pairs, wrong, found = calibrate(people(d), np.repeat(["Joseph", "Ann", "Sepp"], 4), 0.8)
    assert (pairs, wrong, found) == (3, 1, 1.0)
