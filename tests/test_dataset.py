import pandas as pd
import pytest

from whoisit.data.dataset import N_VALUES, pick_aligned_photos, pick_train_photos


def fake_df(people=30, photos=25):
    rows = [(f"{p}_{i}.jpg", p) for p in range(people) for i in range(photos)]
    return pd.DataFrame(rows, columns=["image", "person"])


def test_smaller_n_is_part_of_bigger_n():
    df = fake_df()
    people = list(range(30))
    for small, big in zip(N_VALUES, N_VALUES[1:]):
        a = set(pick_train_photos(df, people, small)["image"])
        b = set(pick_train_photos(df, people, big)["image"])
        assert a <= b


def test_n_photos_for_each_person():
    df = fake_df()
    photos = pick_train_photos(df, list(range(30)), 5)
    assert len(photos) == 30 * 5
    assert (photos.groupby("person").size() == 5).all()


def test_same_seed_same_photos():
    df = fake_df()
    a = pick_train_photos(df, [1, 2, 3], 5, seed=42)
    b = pick_train_photos(df, [1, 2, 3], 5, seed=42)
    c = pick_train_photos(df, [1, 2, 3], 5, seed=7)
    assert a.equals(b)
    assert not a.equals(c)


def test_too_few_photos():
    df = fake_df(photos=3)
    with pytest.raises(ValueError):
        pick_train_photos(df, [0], 5)

def test_failed_photos_are_not_used():
    df = fake_df()
    people = list(range(30))
    failed = {f"{p}_{i}.jpg" for p in people for i in range(0, 25, 7)}
    aligned = set(df["image"]) - failed
    previous = set()
    for n in N_VALUES:
        photos = pick_aligned_photos(df, people, n, aligned)
        assert not set(photos["image"]) & failed
        assert (photos.groupby("person").size() == n).all()
        assert previous <= set(photos["image"])
        previous = set(photos["image"])


def test_people_with_less_than_pool_size_photos():
    df = fake_df(people=10, photos=22)
    photos = pick_aligned_photos(df, list(range(10)), 20, set(df["image"]))
    assert len(photos) == 10 * 20


def test_person_with_too_few_aligned_photos_is_left_out():
    df = fake_df(people=5, photos=22)
    failed = {f"0_{i}.jpg" for i in range(3)}  # person 0 keeps only 19 photos
    aligned = set(df["image"]) - failed
    for n in N_VALUES:
        photos = pick_aligned_photos(df, list(range(5)), n, aligned)
        assert set(photos["person"]) == {1, 2, 3, 4}
        assert (photos.groupby("person").size() == n).all()